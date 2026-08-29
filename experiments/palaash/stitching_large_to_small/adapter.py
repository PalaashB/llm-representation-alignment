"""The large->small adapter: the same affine map, fit two ways.

The map is affine on standardised inputs,

    Y_hat = ((X - mu_x) / sd_x) @ W + b

taking the large model's layer-j residual to something injectable at the small
model's block i (so W is dim_large x dim_small — the transpose of the sibling
package's shape).

`--train-method ridge` chooses its parameters by weighted least squares against
the small model's own layer-i residual:

1. Answer rows are up-weighted *and* prompt rows are thinned, so answer
   positions carry the majority of the objective. They are the population the
   adapter faces once decoding starts, and in the published capture they carried
   16% of the weight. `fit` now refuses to ship a map below 50%.
2. Ridge alpha is light. Shrinkage pulls every prediction toward the mean
   residual, which is what turns a stitched answer into a generic continuation.
3. Optional radial norm matching, folded into (W, b) so the shipped map stays
   affine and free at inference.

`--train-method distill` keeps the same map shape and warm-starts from that
ridge solution, then optimises KL to the large model's next-token distribution
through the frozen small-model suffix (see distill.py). It exists because the
ridge target is the wrong state: on a divergent prompt, the small model's own
layer-i residual is the state that produces the wrong answer, so a map that
matches it perfectly reproduces the error this experiment exists to fix.

Quality is reported prompt-level held out and split by row type. The all-token
number is the one that misled the earlier work — 0.9999 all-token against 0.427
on answer tokens, on a map whose generations were unusable — so `answer` is the
number to read, and under distillation it is expected to *fall*, because L2 in
residual space is no longer what is being minimised.
"""

from __future__ import annotations

import json

import numpy as np
import torch

from stitching_large_to_small.config import (
    ADAPTER_TEST_FRAC, ANSWER_WEIGHT, MIN_ANSWER_WEIGHT_FRAC, NORM_MATCH,
    PROMPT_ROW_KEEP, RIDGE_ALPHA, SEED, TRAIN_METHOD, TRAIN_METHODS, Bank, Pair,
    adapter_path, adapters_dir, states_dir, validate_layers,
)
from stitching_large_to_small.data import load_layer_pair, select_fit_rows


def weighted_ridge(X: np.ndarray, Y: np.ndarray, w: np.ndarray, alpha: float):
    """Ridge with per-row weights. Inputs standardised; the penalty scales with
    total weight so alpha means the same at any row count; intercept unpenalised.
    """
    wsum = float(w.sum())
    mu_x = (w[:, None] * X).sum(0) / wsum
    sd_x = np.sqrt((w[:, None] * (X - mu_x) ** 2).sum(0) / wsum) + 1e-6
    Xs = (X - mu_x) / sd_x
    Xa = np.concatenate([Xs, np.ones((len(X), 1), np.float32)], axis=1)

    mu_y = (w[:, None] * Y).sum(0) / wsum
    Yc = Y - mu_y

    reg = np.full(Xa.shape[1], alpha * wsum, np.float32)
    reg[-1] = 0.0
    Xw = Xa * w[:, None]
    A = Xa.T @ Xw + np.diag(reg)
    L = np.linalg.cholesky(A)
    Wa = np.linalg.solve(L.T, np.linalg.solve(L, Xw.T @ Yc))
    return Wa[:-1], Wa[-1] + mu_y, mu_x, sd_x, mu_y


def apply_np(X, W, b, mu_x, sd_x):
    return ((X - mu_x) / sd_x) @ W + b


def norm_gain(Y_hat: np.ndarray, Y: np.ndarray, mu_y: np.ndarray) -> float:
    """Scalar making the predicted deviation from mu_y as long as the true one.
    Medians, so a handful of massive-activation positions do not set the gain
    for every ordinary token."""
    num = np.median(np.linalg.norm(Y - mu_y, axis=1))
    den = np.median(np.linalg.norm(Y_hat - mu_y, axis=1)) + 1e-12
    return float(num / den)


def rescale(W, b, mu_y, s: float):
    return W * s, mu_y + s * (b - mu_y)


def quality(Y_hat: np.ndarray, Y: np.ndarray) -> dict:
    cos = (Y_hat * Y).sum(1) / (np.linalg.norm(Y_hat, axis=1)
                                * np.linalg.norm(Y, axis=1) + 1e-12)
    rel = np.linalg.norm(Y_hat - Y, axis=1) / (np.linalg.norm(Y, axis=1) + 1e-12)
    ss_res = float(((Y_hat - Y) ** 2).sum())
    ss_tot = float(((Y - Y.mean(0)) ** 2).sum())
    return {"r2": float(1 - ss_res / ss_tot), "cosine_mean": float(cos.mean()),
            "rel_l2_mean": float(rel.mean()), "n_rows": int(len(Y))}


def _ridge_map(X, Y, w, ans, alpha, norm_match) -> tuple[dict, float]:
    """Ridge, then the radial gain folded back into (W, b)."""
    W, b, mu_x, sd_x, mu_y = weighted_ridge(X, Y, w, alpha)
    gain = 1.0
    if norm_match and ans.any():
        gain = norm_gain(apply_np(X[ans], W, b, mu_x, sd_x), Y[ans], mu_y)
        W, b = rescale(W, b, mu_y, gain)
    return {"W": W, "b": b, "mu_x": mu_x, "sd_x": sd_x, "mu_y": mu_y}, gain


def apply_map(m: dict, X: np.ndarray) -> np.ndarray:
    return apply_np(X, m["W"], m["b"], m["mu_x"], m["sd_x"])


def _distil(pair: Pair, bank: Bank, i: int, j: int, base_map: dict,
            X: np.ndarray, Y: np.ndarray, pid: np.ndarray, is_answer: np.ndarray,
            lm_small, lm_large, verbose: bool, **kw) -> tuple[dict, dict]:
    """Run the distillation pass on top of a fitted ridge map.

    Imported here rather than at module scope so the numpy-only paths (`fit`
    with ridge, `apply_map`, the reload check) never pay for loading the models
    or for importing the training code.

    Note the row set: distillation uses *every* captured row, not the
    answer-weighted selection the ridge solve uses. It does not need the
    reweighting — its loss is only evaluated at the positions the adapter can
    move, and the other positions enter only as the true residuals the suffix
    attends back to, which is exactly their role at inference.
    """
    from common.model_utils import load_lm, pick_device
    from stitching_large_to_small import distill
    from stitching_large_to_small.data import load_positions, load_teacher_topk

    t_vals, t_idx = load_teacher_topk(pair, bank)
    if len(t_vals) != len(X):
        raise SystemExit(
            f"teacher logits have {len(t_vals)} rows but states have {len(X)} — "
            f"they came from different captures. Re-run capture.")
    position = load_positions(pair, bank)
    if lm_small is None:
        lm_small = load_lm(pair.small_id, pair.small_tag, pick_device())
    return distill.train(pair, bank, i, j, base_map, lm_small, X, Y, pid,
                         is_answer, position, t_vals, t_idx, lm_large=lm_large,
                         verbose=verbose, **kw)


def fit(pair: Pair, i: int, j: int, bank: Bank,
        train_method: str = TRAIN_METHOD, alpha: float = RIDGE_ALPHA,
        answer_weight: float = ANSWER_WEIGHT, norm_match: bool = NORM_MATCH,
        prompt_row_policy=PROMPT_ROW_KEEP, lm_small=None, lm_large=None,
        verbose: bool = True, **distill_kw) -> dict:
    """Fit and save the large->small adapter for one (i, j).

    `train_method="distill"` fits the same map shape against the large model's
    next-token distribution instead of the small model's hidden states,
    warm-started from the ridge solution computed here. Both land in separately
    scoped files, so the head-to-head at fixed geometry is always on disk.
    """
    validate_layers(pair, i, j)
    if train_method not in TRAIN_METHODS:
        raise SystemExit(f"--train-method must be one of {TRAIN_METHODS}, "
                         f"got {train_method!r}")
    X, Y, pid, is_answer = load_layer_pair(pair, i, j, bank)

    # ── row selection: make answer positions the objective ────────────────────
    # Without this the fit is ~95% boilerplate prompt rows (see data.py), and
    # the map optimises a chat template it will never be asked to reproduce.
    keep, sel = select_fit_rows(is_answer, pid, answer_weight, prompt_row_policy)
    Xf, Yf, pidf, ans_f = X[keep], Y[keep], pid[keep], is_answer[keep]
    w = np.where(ans_f == 1, answer_weight, 1.0).astype(np.float32)
    ans = ans_f == 1
    if sel["answer_weight_frac"] <= MIN_ANSWER_WEIGHT_FRAC:
        raise SystemExit(
            f"answer rows carry only {sel['answer_weight_frac']:.1%} of the fit "
            f"objective's weight (need > {MIN_ANSWER_WEIGHT_FRAC:.0%}). "
            f"{sel['n_answer_rows']} answer rows against "
            f"{sel['n_prompt_rows_kept']} prompt rows at answer_weight="
            f"{answer_weight}. Capture more answer positions (--fit-corpus generic, "
            f"a longer corpus answer budget) or raise --answer-weight.")

    # ── honest quality: prompt-level holdout, refit from scratch ──────────────
    pids = np.unique(pidf)
    rng = np.random.default_rng(SEED)
    rng.shuffle(pids)
    n_te = max(1, int(round(len(pids) * ADAPTER_TEST_FRAC)))
    te = np.isin(pidf, pids[:n_te])
    tr = ~te
    if verbose:
        print(f"[fit] {pair.name}/{bank.name}  large L{j} -> small L{i}  "
              f"train={train_method}  rows={len(Xf)} of {len(X)}")
        print(f"  answer rows={sel['n_answer_rows']}  prompt rows kept="
              f"{sel['n_prompt_rows_kept']} of {sel['n_prompt_rows_available']}  "
              f"answer_weight_frac={sel['answer_weight_frac']:.1%}")
    mh, _ = _ridge_map(Xf[tr], Yf[tr], w[tr], ans[tr], alpha, norm_match)
    Yh = apply_map(mh, Xf[te])
    ans_te = ans[te]
    held = {"all": quality(Yh, Yf[te]),
            "answer": quality(Yh[ans_te], Yf[te][ans_te]) if ans_te.any() else None,
            "prompt": quality(Yh[~ans_te], Yf[te][~ans_te]) if (~ans_te).any() else None}

    # ── the shipped map: every selected row ───────────────────────────────────
    m, gain = _ridge_map(Xf, Yf, w, ans, alpha, norm_match)

    distill_meta = {}
    if train_method == "distill":
        m_ridge = dict(m)
        m, distill_meta = _distil(pair, bank, i, j, m, X, Y, pid, is_answer,
                                  lm_small, lm_large, verbose, **distill_kw)
        # Both maps scored on the same rows, and both of them were fit on those
        # rows — so this pair is comparable to *each other* and not to the
        # `held_out` block above, which refits on a subset and is the honest
        # generalisation number. Scoring the distilled map against `held_out`
        # would read as a large improvement caused entirely by the shipped map
        # having seen those rows, which is the kind of accidental flattery this
        # folder exists to stop printing.
        #
        # R2 is expected to *fall* here: the distilled map is no longer
        # optimising L2 in residual space, which is the entire point. The
        # honest before/after for distillation is `distill_val_loss_ridge` vs
        # `distill_val_loss_best` — a KL on prompts held out of the training
        # loop, in the units the objective actually minimises.
        pairwise = {}
        for name, mm in (("ridge", m_ridge), ("distill", m)):
            Yh_x = apply_map(mm, Xf[te])
            pairwise[name] = {
                "all": quality(Yh_x, Yf[te]),
                "answer": quality(Yh_x[ans_te], Yf[te][ans_te]) if ans_te.any() else None}
        held = held | {"after_distill": pairwise["distill"],
                       "before_distill_same_rows": pairwise["ridge"],
                       "after_distill_note":
                           "both maps were fit on these rows; compare the two to "
                           "each other, not to held_out"}

    in_sample = quality(apply_map(m, Xf), Yf)

    adapters_dir(pair, bank).mkdir(parents=True, exist_ok=True)
    path = adapter_path(pair, i, j, bank, train_method)
    np.savez(path, **{k: v.astype(np.float32) for k, v in m.items()})
    meta = {
        "pair": pair.name, "bank": bank.name, "direction": "large->small",
        "small_layer_i": i, "large_layer_j": j,
        "train_method": train_method,
        "dim_in_large": pair.dim_large, "dim_out_small": pair.dim_small,
        "depth_matched_i": pair.depth_matched_i(j),
        "form": "Y_hat = ((X - mu_x) / sd_x) @ W + b",
        "injection_point": f"input to small decoder block {i}",
        "source_point": f"input to large decoder block {j}",
        "ridge_alpha": alpha, "answer_weight": answer_weight,
        "norm_match": norm_match, "norm_gain": gain,
        "n_rows_captured": int(len(X)), "n_rows": int(len(Xf)),
        "n_answer_rows": int(sel["n_answer_rows"]),
        "n_prompt_rows_kept": int(sel["n_prompt_rows_kept"]),
        "n_prompt_rows_available": int(sel["n_prompt_rows_available"]),
        # The number that was 16% for the whole published sweep and appeared
        # nowhere on disk. It is asserted above and recorded here.
        "answer_weight_frac": sel["answer_weight_frac"],
        "prompt_row_policy": sel["prompt_row_policy"],
        "n_prompts": int(len(pids)), "seed": SEED,
        "capture": _capture_provenance(pair, bank),
        "in_sample": in_sample, "held_out": held,
    } | distill_meta
    with open(path.with_suffix(".json"), "w") as f:
        json.dump(meta, f, indent=2)

    if verbose:
        a_, p_ = held["answer"], held["prompt"]
        print(f"  gain={gain:.3f}  "
              f"(depth-matched i for j={j} would be {pair.depth_matched_i(j)})")
        print(f"  held-out R2   all={held['all']['r2']:+.4f}"
              + (f"  answer={a_['r2']:+.4f} (n={a_['n_rows']})" if a_ else "")
              + (f"  prompt={p_['r2']:+.4f}" if p_ else ""))
        print(f"  held-out cos  all={held['all']['cosine_mean']:.4f}"
              + (f"  answer={a_['cosine_mean']:.4f}" if a_ else ""))
        if "after_distill" in held:
            d, r = held["after_distill"], held["before_distill_same_rows"]
            fmt = lambda q: (f"all={q['all']['r2']:+.4f}"
                             + (f"  answer={q['answer']['r2']:+.4f}"
                                if q["answer"] else ""))
            print(f"  same-rows R2  ridge   {fmt(r)}")
            print(f"  same-rows R2  distill {fmt(d)}"
                  "   (a fall is expected — the objective is no longer L2)")
            print(f"  val KL        ridge warm start "
                  f"{distill_meta['distill_val_loss_ridge']:.4f} -> "
                  f"{distill_meta['distill_val_loss_best']:.4f} @ epoch "
                  f"{distill_meta['distill_best_epoch']}  "
                  f"(improved: {distill_meta['distill_improved_on_ridge']})")
        print(f"  wrote {path.name}")
    return meta


def _capture_provenance(pair: Pair, bank: Bank) -> dict:
    """What the map was fit on, copied into the sidecar.

    Captures are overwritten in place, so the sidecar is the only durable record
    of which fit set produced a given map. The published `legacy` adapters were
    fit on a capture with 753 answer rows and one chat template; without this
    field a reader has no way to tell those files apart from a map fit on the
    rebuilt corpus.
    """
    try:
        with open(states_dir(pair, bank) / "meta.json") as f:
            c = json.load(f)
    except (OSError, ValueError):
        return {}
    return {k: c.get(k) for k in
            ("n_prompts", "n_bank_prompts", "n_corpus_prompts", "fit_corpus",
             "vary_templates", "n_rows", "n_answer_rows", "teacher_answer_tokens",
             "corpus_answer_tokens", "teacher_logits", "teacher_topk")}


def load(pair: Pair, i: int, j: int, bank: Bank,
         train_method: str | None = None) -> tuple[dict, dict]:
    path = adapter_path(pair, i, j, bank, train_method)
    if not path.exists():
        raise SystemExit(
            f"{path.name} not found — run `python -m stitching_large_to_small.run fit "
            f"--pair {pair.name} --bank {bank.name} --i {i} --j {j} "
            f"--train-method {train_method or 'ridge'}` first.")
    z = np.load(path)
    with open(path.with_suffix(".json")) as f:
        meta = json.load(f)
    return {k: z[k] for k in ("W", "b", "mu_x", "sd_x", "mu_y")}, meta


def reload_quality(pair: Pair, i: int, j: int, bank: Bank,
                   train_method: str | None = None, lm_small=None) -> dict:
    """Re-load the saved coefficients and re-score them on held-out rows.

    A gating check, not a metric: it catches a map that was saved wrong, or
    loaded into the wrong dtype/shape, before any accuracy number is reported.
    The fit already reported these values; this recomputes them from what is
    actually on disk.

    For a distilled map there is a second, stricter arm. Its parameters live in
    a `torch.nn.ParameterDict` on the accelerator and are written out through
    numpy, so a dtype narrowing, a lost tensor or a transposed weight would
    leave a file that still scores fine on R2 — it is approximately the ridge
    map either way — while decoding differently from the thing whose validation
    loss selected it. Replaying the recorded probe through the frozen suffix is
    the only check that would notice.
    """
    arrays, meta = load(pair, i, j, bank, train_method)
    X, Y, pid, is_answer = load_layer_pair(pair, i, j, bank)
    pids = np.unique(pid)
    rng = np.random.default_rng(SEED)
    rng.shuffle(pids)
    te = np.isin(pid, pids[:max(1, int(round(len(pids) * ADAPTER_TEST_FRAC)))])
    Yh = apply_np(X[te], arrays["W"], arrays["b"], arrays["mu_x"], arrays["sd_x"])
    ans = is_answer[te] == 1
    q_all = quality(Yh, Y[te])
    q_ans = quality(Yh[ans], Y[te][ans]) if ans.any() else None
    # The shipped map is fit on every row including these, so this is an
    # in-sample number for the *shipped* map; the honest held-out figure is in
    # the fit metadata. What is being checked here is that the file round-trips.
    out = {"check": "adapter_reload", "train_method": train_method or "legacy",
           "r2_all": q_all["r2"], "cosine_all": q_all["cosine_mean"],
           "r2_answer": q_ans["r2"] if q_ans else None,
           "cosine_answer": q_ans["cosine_mean"] if q_ans else None,
           "held_out_answer_r2_from_fit":
               (meta.get("held_out", {}).get("answer") or {}).get("r2"),
           "answer_weight_frac": meta.get("answer_weight_frac"),
           "shapes_ok": bool(arrays["W"].shape == (pair.dim_large, pair.dim_small)
                             and arrays["b"].shape == (pair.dim_small,)),
           "finite": bool(all(np.isfinite(v).all() for v in arrays.values())),
           "reproduces_training_logits": None}
    out["passed"] = bool(np.isfinite(q_all["r2"]) and out["shapes_ok"]
                         and out["finite"])
    if meta.get("train_method") != "distill" or "probe_logits_argmax" not in meta:
        return out

    from common.decoding import Stack
    from common.model_utils import load_lm, pick_device
    from stitching_large_to_small.data import load_positions

    position = load_positions(pair, bank)
    row0 = meta["probe_prompt_row0"]
    idx = np.flatnonzero(pid == pid[row0])
    idx = idx[np.argsort(position[idx], kind="stable")]
    a = is_answer[idx] == 1
    if meta.get("distill_train_mode", "warm") == "exit":
        from stitching_large_to_small.config import PRESERVE_PREFIX
        inject = np.ones(len(idx), bool)
        inject[:min(PRESERVE_PREFIX, len(idx))] = False
    else:
        inject = a

    lm_small = lm_small or load_lm(pair.small_id, pair.small_tag, pick_device())
    stack = Stack(lm_small)
    with torch.no_grad():
        to = lambda v: torch.from_numpy(np.ascontiguousarray(v)).to(lm_small.device)
        h = to(Y[idx]).to(torch.float32).unsqueeze(0).clone()
        h[0, to(np.flatnonzero(inject)).long()] = to(
            apply_map(arrays, X[idx][inject])).to(torch.float32)
        pos = to(position[idx]).long().unsqueeze(0)
        hs = stack.run(h.to(stack.dtype), pos, i, stack.n_layers, cache=None)
        got = stack.lm_head(stack.base.norm(hs))[0][-1].float().cpu().numpy()

    want_head = np.array(meta["probe_logits_sha_head"], np.float32)
    diff = float(np.abs(got[:8] - want_head).max())
    # The suffix runs in bf16, which has 8 mantissa bits, so a logit of
    # magnitude m carries a quantisation step of about m * 2^-8 — and the
    # training-time probe went through the autograd graph while this one does
    # not, which is enough to select different GEMM kernels. An absolute
    # threshold is therefore the wrong instrument. The criterion is (a) the
    # argmax must match, because that is what greedy decoding reads, and (b) the
    # logits must agree to a few ULPs at the scale of the logits actually
    # flowing through the stack — not at the scale of the eight sampled head
    # values, which happen to be small. A genuinely mis-saved map moves logits
    # by whole units, not by ULPs, so this still catches what it exists to catch.
    scale = max(1.0, float(np.abs(got).max()))
    tol = max(4 * scale * 2 ** -8, 1e-3)
    out |= {"probe_argmax_saved": meta["probe_logits_argmax"],
            "probe_argmax_reloaded": int(got.argmax()),
            "probe_logit_max_abs_diff": diff,
            "probe_logit_tolerance": tol,
            "reproduces_training_logits":
                bool(int(got.argmax()) == meta["probe_logits_argmax"] and diff <= tol)}
    out["passed"] = bool(out["shapes_ok"] and out["finite"]
                         and out["reproduces_training_logits"])
    return out


class TorchAdapter:
    """The saved affine map on the accelerator, applied in float32."""

    def __init__(self, arrays: dict, meta: dict, device: str):
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(device, torch.float32)
        self.W, self.b = t(arrays["W"]), t(arrays["b"])
        self.mu_x, self.sd_x = t(arrays["mu_x"]), t(arrays["sd_x"])
        self.meta = meta
        self.i, self.j = meta["small_layer_i"], meta["large_layer_j"]

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return ((x.to(torch.float32) - self.mu_x) / self.sd_x) @ self.W + self.b

    @property
    def n_params(self) -> int:
        return self.W.numel() + self.b.numel()
