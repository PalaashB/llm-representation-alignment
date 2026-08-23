"""The small->large adapter: weighted ridge, optionally plus a nonlinear
correction, fit for *generation*.

The base map is affine on standardised inputs,

    Y_hat = ((X - mu_x) / sd_x) @ W + b

and is applied to the small model's residual stream to produce the large model's
layer-j residual stream. Three things differ from the plain ridge fit in
diagnosis/fit_adapter, all of them aimed at surviving a multi-token decode rather
than at estimating a diagnostic quantity:

1. Answer rows are up-weighted (`answer_weight`). They are a minority of the
   rows but 100% of what the map is asked to do once decoding starts.
2. Ridge alpha is an order of magnitude lighter. Shrinkage is not free here:
   it pulls every prediction toward the mean residual, which is exactly the
   "generic continuation" failure mode.
3. Optional radial norm matching (`norm_match`). Ridge is a shrinkage
   estimator, so ||Y_hat - mu_y|| is systematically short of ||Y - mu_y||; the
   large model's later blocks read that as a low-confidence state. A single
   scalar s, estimated on answer rows, rescales the centred prediction. It folds
   into (W, b), so the base map stays purely affine and costs nothing at
   inference.

Two levers widen the map beyond that (both default off, so `linear` with one tap
reproduces the earlier results exactly):

`--n-taps k`  The input is the small model's residual at k depths, concatenated,
    instead of layer i alone. Free at inference — blocks 0..i-1 run on the way to
    i regardless, so an intermediate residual is already in flight and only has
    to be kept. It matters because one layer's residual stream is a lossy summary
    of the computation that produced it: attention and MLP writes overwrite as
    much as they accumulate, and a feature the large model still needs at layer j
    may be legible at i/2 and gone by i.

`--adapter mlp`  A Linear -> GELU -> Linear correction is added to the affine
    map. The reason to expect this to help is that the target is not a linear
    function of the input: the two models' residual bases differ by more than a
    rotation and rescale, and a single matrix has no way to represent "this
    direction means something different depending on what else is present".
    The output layer is zero-initialised, so training starts at exactly the ridge
    solution and the correction can only be learned if it reduces the loss. At
    MLP_HIDDEN=1024 it adds ~2.6M weights against the ~2.8B the large suffix
    already multiplies per token, so it does not move the latency it would have
    to earn back.

Reported quality is prompt-level held out and split by row type, because the
all-token number is the one that misled the first attempt (0.91 all-token vs 0.34
answer-token R² on a map whose generations were unusable).
"""

from __future__ import annotations

import json

import numpy as np
import torch

from stitching_small_to_large.config import (
    ADAPTER_KIND, ADAPTER_TEST_FRAC, ANSWER_WEIGHT, Bank, MIN_ANSWER_WEIGHT_FRAC,
    MLP_BATCH, MLP_EPOCHS, MLP_HIDDEN, MLP_LR, MLP_VAL_FRAC, MLP_WEIGHT_DECAY,
    N_TAPS, NORM_MATCH, PROMPT_ROW_KEEP, Pair, RIDGE_ALPHA, SEED, TRAIN_METHOD,
    adapter_path, adapters_dir, states_dir, validate_layers,
)
from stitching_small_to_large.data import load_layer_pair, select_fit_rows

ARRAY_KEYS = ("W", "b", "mu_x", "sd_x", "mu_y")
MLP_KEYS = ("W1", "b1", "W2", "b2")


# ── the affine part ───────────────────────────────────────────────────────────
def weighted_ridge(X: np.ndarray, Y: np.ndarray, w: np.ndarray, alpha: float):
    """Ridge with per-row weights. Inputs standardised; the penalty is scaled by
    total weight so alpha means the same thing at any row count; the intercept
    is never penalised.

    Returns (W, b, mu_x, sd_x, mu_y) for Y_hat = ((X - mu_x) / sd_x) @ W + b.
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


def norm_gain(Y_hat: np.ndarray, Y: np.ndarray, mu_y: np.ndarray) -> float:
    """Scalar s making the predicted deviation from mu_y as long as the true one.

    Medians rather than means: a handful of huge-norm positions (the sink, and
    whatever massive activations the family carries) would otherwise set the
    gain for every ordinary token.
    """
    num = np.median(np.linalg.norm(Y - mu_y, axis=1))
    den = np.median(np.linalg.norm(Y_hat - mu_y, axis=1)) + 1e-12
    return float(num / den)


def rescale(W, b, mu_y, s: float):
    """Fold the radial gain into the affine map: mu_y + s * (Y_hat - mu_y)."""
    return W * s, mu_y + s * (b - mu_y)


# ── the nonlinear correction ──────────────────────────────────────────────────
def gelu_np(x: np.ndarray) -> np.ndarray:
    """The tanh approximation, matching torch's `gelu(..., approximate="tanh")`
    so the numpy scoring path and the torch inference path compute the same
    function."""
    return 0.5 * x * (1.0 + np.tanh(0.7978845608028654 * (x + 0.044715 * x ** 3)))


def train_mlp(Xs: np.ndarray, R: np.ndarray, w: np.ndarray, val: np.ndarray, *,
              hidden: int = MLP_HIDDEN, epochs: int = MLP_EPOCHS, batch: int = MLP_BATCH,
              lr: float = MLP_LR, weight_decay: float = MLP_WEIGHT_DECAY,
              seed: int = SEED, device: str = "cpu", verbose: bool = True) -> dict:
    """Fit Linear -> GELU -> Linear to predict the ridge map's residual R.

    Xs is already standardised (the ridge map's own standardisation is reused, so
    the two halves see identical inputs). R is normalised by its own scale before
    training and the scale is folded back into the output layer afterwards, which
    keeps one learning rate valid across pairs whose residual streams differ in
    magnitude by an order of magnitude.

    Trained on the residual rather than on Y directly for the same reason the
    output layer starts at zero: the affine solution is known to be decent, and
    the only question is whether anything is left over that a nonlinearity can
    reach.

    `val` marks rows (whole prompts, chosen by the caller) excluded from training
    and used only to choose which epoch to keep. With ~1.1k answer rows against
    5M weights this is not optional — without it the correction memorises the fit
    set and comes out worse than the ridge map it was supposed to improve. The
    returned weights are the best epoch's, not the last.
    """
    torch.manual_seed(seed)
    scale = float(np.sqrt((R ** 2).mean())) + 1e-12
    to = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(device, torch.float32)
    w = w / w.mean()
    tr = ~val
    X_t, R_t, w_t = to(Xs[tr]), to(R[tr] / scale), to(w[tr])
    X_v, R_v, w_v = to(Xs[val]), to(R[val] / scale), to(w[val])

    net = torch.nn.Sequential(
        torch.nn.Linear(Xs.shape[1], hidden),
        torch.nn.GELU(approximate="tanh"),
        torch.nn.Linear(hidden, R.shape[1]),
    ).to(device, torch.float32)
    torch.nn.init.zeros_(net[2].weight)
    torch.nn.init.zeros_(net[2].bias)

    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n = len(X_t)
    g = torch.Generator().manual_seed(seed)

    @torch.no_grad()
    def val_loss(module=None) -> float:
        pred = 0.0 if module is None else module(X_v)
        return float((w_v[:, None] * (pred - R_v) ** 2).mean())

    # Epoch 0 is the ridge map alone (the correction is identically zero), so it
    # is a legitimate candidate: if nothing beats it, the correction is dropped.
    best, best_ep, best_state = val_loss(), 0, None
    ridge_val = best
    for ep in range(1, epochs + 1):
        perm = torch.randperm(n, generator=g).to(device)
        tot = 0.0
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            loss = (w_t[idx, None] * (net(X_t[idx]) - R_t[idx]) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(idx)
        sched.step()
        v = val_loss(net)
        if v < best:
            best, best_ep = v, ep
            best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
        if verbose and (ep % 20 == 0 or ep == 1):
            print(f"  [mlp] epoch {ep:>3}  train {tot / n:.4f}  val {v:.4f}"
                  f"  (best {best:.4f} @ {best_ep})")
    if verbose:
        print(f"  [mlp] hidden={hidden} rows={n} val_rows={int(val.sum())}  "
              f"val MSE {ridge_val:.4f} (ridge alone) -> {best:.4f} @ epoch {best_ep}"
              + ("  — correction dropped, no epoch beat the ridge map"
                 if best_state is None else ""))
    if best_state is None:                  # keep the affine map exactly as it was
        return {}
    p = {k: v.cpu().numpy() for k, v in best_state.items()}
    return {"W1": p["0.weight"].T.astype(np.float32), "b1": p["0.bias"].astype(np.float32),
            # scale folded back in, so apply_map needs no knowledge of it
            "W2": (p["2.weight"].T * scale).astype(np.float32),
            "b2": (p["2.bias"] * scale).astype(np.float32)}


# ── the composed map ──────────────────────────────────────────────────────────
def apply_map(m: dict, X: np.ndarray) -> np.ndarray:
    """Y_hat for a fitted map, affine-only or affine + correction."""
    Xs = (X - m["mu_x"]) / m["sd_x"]
    out = Xs @ m["W"] + m["b"]
    if "W1" in m:
        out = out + gelu_np(Xs @ m["W1"] + m["b1"]) @ m["W2"] + m["b2"]
    return out


def prompt_val_mask(pid: np.ndarray, frac: float = MLP_VAL_FRAC,
                    seed: int = SEED + 1) -> np.ndarray:
    """Row mask selecting whole prompts for the MLP's early-stopping split.

    By prompt, not by row: neighbouring positions in one answer are nearly
    identical, so a row-wise split would put near-copies of training rows in the
    validation set and report no overfitting whatever the model did. Seeded off
    SEED so it cannot coincide with the reported holdout split.
    """
    pids = np.unique(pid)
    rng = np.random.default_rng(seed)
    rng.shuffle(pids)
    return np.isin(pid, pids[:max(1, int(round(len(pids) * frac)))])


def fit_map(X: np.ndarray, Y: np.ndarray, w: np.ndarray, ans: np.ndarray,
            pid: np.ndarray, *, alpha: float, norm_match: bool, kind: str,
            device: str = "cpu", verbose: bool = True, **mlp_kw) -> tuple[dict, float]:
    """Ridge, then norm matching, then (for kind='mlp') the correction.

    Norm matching is applied before the correction rather than after so the
    correction learns against the map that actually ships, and so `norm_gain`
    stays comparable between the linear and mlp variants.

    The ridge solve uses every row; only the MLP holds prompts back, because ridge
    at this alpha is nowhere near able to memorise 8.5k rows and needs no stopping
    rule.
    """
    W, b, mu_x, sd_x, mu_y = weighted_ridge(X, Y, w, alpha)
    gain = 1.0
    if norm_match and ans.any():
        gain = norm_gain(apply_map({"W": W, "b": b, "mu_x": mu_x, "sd_x": sd_x}, X[ans]),
                         Y[ans], mu_y)
        W, b = rescale(W, b, mu_y, gain)
    m = {"W": W, "b": b, "mu_x": mu_x, "sd_x": sd_x, "mu_y": mu_y}
    if kind == "mlp":
        m |= train_mlp((X - mu_x) / sd_x, Y - apply_map(m, X), w, prompt_val_mask(pid),
                       device=device, verbose=verbose, **mlp_kw)
    return m, gain


def quality(Y_hat: np.ndarray, Y: np.ndarray) -> dict:
    cos = (Y_hat * Y).sum(1) / (np.linalg.norm(Y_hat, axis=1) * np.linalg.norm(Y, axis=1) + 1e-12)
    rel = np.linalg.norm(Y_hat - Y, axis=1) / (np.linalg.norm(Y, axis=1) + 1e-12)
    ss_res = float(((Y_hat - Y) ** 2).sum())
    ss_tot = float(((Y - Y.mean(0)) ** 2).sum())
    return {"r2": float(1 - ss_res / ss_tot), "cosine_mean": float(cos.mean()),
            "rel_l2_mean": float(rel.mean()), "n_rows": int(len(Y))}


def _distil(pair: Pair, bank: Bank, i: int, j: int, base_map: dict,
            X: np.ndarray, Y: np.ndarray, pid: np.ndarray, is_answer: np.ndarray,
            lm_large, lm_small, verbose: bool) -> tuple[dict, dict]:
    """Run the distillation pass on top of a fitted ridge map.

    Imported here rather than at module scope so the numpy-only paths (`fit`
    with ridge, `apply_map`, the reload check) never pay for loading the models
    or for importing the training code.

    Note the row set: distillation uses *every* captured row, not the
    answer-weighted selection the ridge solve uses. It does not need the
    reweighting — its loss is only evaluated at answer positions to begin with,
    and prompt positions enter only as the true residuals the suffix attends
    back to, which is exactly their role at inference.
    """
    from common.model_utils import load_lm, pick_device
    from stitching_small_to_large import distill
    from stitching_small_to_large.data import load_teacher_topk

    t_vals, t_idx = load_teacher_topk(pair, bank)
    if len(t_vals) != len(X):
        raise SystemExit(
            f"teacher logits have {len(t_vals)} rows but states have {len(X)} — "
            f"they came from different captures. Re-run capture.")
    with np.load(states_dir(pair, bank) / "meta.npz") as z:
        position = z["position"]
    if lm_large is None:
        lm_large = load_lm(pair.large_id, pair.large_tag, pick_device())
    return distill.train(pair, bank, i, j, base_map, lm_large, X, Y, pid,
                         is_answer, position, t_vals, t_idx, lm_small=lm_small,
                         verbose=verbose)


def fit(pair: Pair, i: int, j: int, bank: Bank, kind: str = ADAPTER_KIND,
        n_taps: int = N_TAPS, train_method: str = TRAIN_METHOD,
        alpha: float = RIDGE_ALPHA,
        answer_weight: float = ANSWER_WEIGHT, norm_match: bool = NORM_MATCH,
        device: str = "cpu", verbose: bool = True,
        prompt_row_policy=PROMPT_ROW_KEEP,
        lm_large=None, lm_small=None, **mlp_kw) -> dict:
    """Fit and save the adapter for one (i, j). Returns its metadata.

    `train_method="distill"` fits the same map shape against the teacher's
    next-token distribution instead of its hidden states, warm-started from the
    ridge solution computed here. Both land in separately scoped files so the
    head-to-head comparison at fixed geometry is always available.
    """
    validate_layers(pair, i, j)
    X, Y, pid, is_answer, taps = load_layer_pair(pair, i, j, bank, n_taps)

    # ── row selection: make answer positions the objective ────────────────────
    # Without this the fit is ~83% boilerplate prompt rows (see data.py), and
    # the map optimises a template it will never be asked to reproduce.
    keep, sel = select_fit_rows(is_answer, pid, answer_weight, prompt_row_policy)
    Xf, Yf, pidf, is_answer_f = X[keep], Y[keep], pid[keep], is_answer[keep]
    w = np.where(is_answer_f == 1, answer_weight, 1.0).astype(np.float32)
    ans = is_answer_f == 1
    if sel["answer_weight_frac"] <= MIN_ANSWER_WEIGHT_FRAC:
        raise SystemExit(
            f"answer rows carry only {sel['answer_weight_frac']:.1%} of the fit "
            f"objective's weight (need > {MIN_ANSWER_WEIGHT_FRAC:.0%}). "
            f"{sel['n_answer_rows']} answer rows against "
            f"{sel['n_prompt_rows_kept']} prompt rows at answer_weight="
            f"{answer_weight}. Capture more answer positions (--fit-corpus generic, "
            f"a longer-answer bank) or raise --answer-weight.")

    # ── honest quality: prompt-level holdout, refit from scratch ──────────────
    pids = np.unique(pidf)
    rng = np.random.default_rng(SEED)
    rng.shuffle(pids)
    n_te = max(1, int(round(len(pids) * ADAPTER_TEST_FRAC)))
    te = np.isin(pidf, pids[:n_te])
    tr = ~te
    if verbose:
        print(f"[fit] {pair.name} bank={bank.name} L{i} -> L{j}  kind={kind} "
              f"train={train_method} taps={list(taps)}  "
              f"rows={len(Xf)} of {len(X)}  d_in={Xf.shape[1]}")
        print(f"  answer rows={sel['n_answer_rows']}  prompt rows kept="
              f"{sel['n_prompt_rows_kept']} of {sel['n_prompt_rows_available']}  "
              f"answer_weight_frac={sel['answer_weight_frac']:.1%}")
    mh, _ = fit_map(Xf[tr], Yf[tr], w[tr], ans[tr], pidf[tr], alpha=alpha,
                    norm_match=norm_match, kind=kind, device=device, verbose=False,
                    **mlp_kw)
    Yh = apply_map(mh, Xf[te])
    ans_te = ans[te]
    held = {"all": quality(Yh, Yf[te]),
            "answer": quality(Yh[ans_te], Yf[te][ans_te]) if ans_te.any() else None,
            "prompt": quality(Yh[~ans_te], Yf[te][~ans_te]) if (~ans_te).any() else None}

    # ── the shipped map: every selected row ───────────────────────────────────
    m, gain = fit_map(Xf, Yf, w, ans, pidf, alpha=alpha, norm_match=norm_match,
                      kind=kind, device=device, verbose=verbose, **mlp_kw)

    distill_meta = {}
    if train_method == "distill":
        m, distill_meta = _distil(pair, bank, i, j, m, X, Y, pid, is_answer,
                                  lm_large, lm_small, verbose)
        # Re-scored against the same held-out rows as the ridge map, so the two
        # numbers in the sidecar are comparable. R2 is expected to *fall*: the
        # distilled map is no longer optimising L2 in residual space, which is
        # the entire point.
        Yh_d = apply_map(m, Xf[te])
        held = held | {"after_distill": {
            "all": quality(Yh_d, Yf[te]),
            "answer": quality(Yh_d[ans_te], Yf[te][ans_te]) if ans_te.any() else None}}

    in_sample = quality(apply_map(m, Xf), Yf)

    adapters_dir(pair, bank).mkdir(parents=True, exist_ok=True)
    path = adapter_path(pair, i, j, bank, kind, n_taps, train_method)
    np.savez(path, **{k: v.astype(np.float32) for k, v in m.items()})
    n_params = int(sum(v.size for k, v in m.items() if k in ARRAY_KEYS + MLP_KEYS
                       and k not in ("mu_x", "sd_x", "mu_y")))
    meta = {
        "pair": pair.name, "bank": bank.name,
        "small_layer_i": i, "large_layer_j": j,
        "kind": kind, "train_method": train_method,
        "n_taps": len(taps), "taps": list(taps),
        "dim_small": pair.dim_small, "dim_in": int(Xf.shape[1]),
        "dim_large": pair.dim_large,
        "depth_matched_j": pair.depth_matched_j(i),
        "form": ("Y_hat = Xs @ W + b + gelu(Xs @ W1 + b1) @ W2 + b2, "
                 "Xs = (X - mu_x) / sd_x" if "W1" in m else
                 "Y_hat = ((X - mu_x) / sd_x) @ W + b"),
        "injection_point": f"input to large decoder block {j}",
        "ridge_alpha": alpha, "answer_weight": answer_weight,
        "norm_match": norm_match, "norm_gain": gain,
        # False when kind == "mlp" but no epoch beat the ridge map on the
        # early-stopping split, so what shipped is the affine map alone.
        "mlp_correction_kept": "W1" in m,
        "mlp_hidden": int(m["W1"].shape[1]) if "W1" in m else None,
        "mlp_val_frac": MLP_VAL_FRAC if kind == "mlp" else None,
        "n_map_params": n_params,
        "n_rows_captured": int(len(X)), "n_rows": int(len(Xf)),
        "n_answer_rows": int(sel["n_answer_rows"]),
        "n_prompt_rows_kept": int(sel["n_prompt_rows_kept"]),
        "n_prompt_rows_available": int(sel["n_prompt_rows_available"]),
        # The number that was 17% for the whole `factual` campaign and appeared
        # nowhere on disk. It is asserted above and recorded here.
        "answer_weight_frac": sel["answer_weight_frac"],
        "prompt_row_policy": sel["prompt_row_policy"],
        "n_prompts": int(len(pids)),
        "in_sample": in_sample, "held_out": held,
    } | distill_meta
    with open(path.with_suffix(".json"), "w") as f:
        json.dump(meta, f, indent=2)

    if verbose:
        a, p = held["answer"], held["prompt"]
        print(f"  gain={gain:.3f}  map params={n_params / 1e6:.1f}M  "
              f"(depth-matched j would be {pair.depth_matched_j(i)})")
        print(f"  held-out R2   all={held['all']['r2']:+.4f}"
              + (f"  answer={a['r2']:+.4f} (n={a['n_rows']})" if a else "")
              + (f"  prompt={p['r2']:+.4f}" if p else ""))
        print(f"  held-out cos  all={held['all']['cosine_mean']:.4f}"
              + (f"  answer={a['cosine_mean']:.4f}" if a else ""))
        if "after_distill" in held:
            d = held["after_distill"]
            print(f"  after distill R2  all={d['all']['r2']:+.4f}"
                  + (f"  answer={d['answer']['r2']:+.4f}" if d["answer"] else "")
                  + "   (a fall here is expected — the objective is no longer L2)")
        print(f"  wrote {path.name}")
    return meta


# ── loading / inference ───────────────────────────────────────────────────────
def load(pair: Pair, i: int, j: int, bank: Bank, kind: str = ADAPTER_KIND,
         n_taps: int = N_TAPS, train_method: str = TRAIN_METHOD) -> tuple[dict, dict]:
    path = adapter_path(pair, i, j, bank, kind, n_taps, train_method)
    if not path.exists():
        raise SystemExit(f"{path.name} not found — run `python -m stitching_small_to_large.run fit --pair "
                         f"{pair.name} --bank {bank.name} --adapter {kind} "
                         f"--n-taps {n_taps} --train-method {train_method} "
                         f"--i {i} --j {j}` first.")
    z = np.load(path)
    with open(path.with_suffix(".json")) as f:
        meta = json.load(f)
    return {k: z[k] for k in z.files}, meta


def check_adapter_reload(pair: Pair, i: int, j: int, bank: Bank,
                         kind: str = ADAPTER_KIND, n_taps: int = N_TAPS,
                         train_method: str = TRAIN_METHOD,
                         lm_large=None) -> dict:
    """The saved map round-trips, and — for a distilled map — still produces the
    logits training ended at.

    The shape/finite half of this catches a file written or reloaded wrong. The
    logits half is specific to distillation and is the one that matters: the
    trained parameters live in a `torch.nn.ParameterDict` on the accelerator and
    are written out through numpy, so a dtype narrowing, a lost tensor, or a
    transposed weight would leave a map that scores fine on R2 (it is still
    approximately the ridge map) while decoding differently from the thing whose
    validation loss was used to select it. Replaying the recorded probe is the
    only check that would notice.
    """
    arrays, meta = load(pair, i, j, bank, kind, n_taps, train_method)
    d_out = pair.dim_large
    out = {"check": "adapter_reload",
           "variant": f"{kind}/t{n_taps}/{train_method}",
           "shapes_ok": bool(arrays["W"].shape[1] == d_out
                             and arrays["b"].shape == (d_out,)),
           "finite": bool(all(np.isfinite(v).all() for v in arrays.values())),
           "reproduces_training_logits": None}
    out["passed"] = out["shapes_ok"] and out["finite"]

    if meta.get("train_method") != "distill" or "probe_logits_argmax" not in meta:
        return out

    from common.decoding import Stack
    from common.model_utils import load_lm, pick_device
    from stitching_small_to_large.data import load_layer_pair

    X, Y, pid, is_answer, _ = load_layer_pair(pair, i, j, bank, n_taps)
    with np.load(states_dir(pair, bank) / "meta.npz") as z:
        position = z["position"]
    row0 = meta["probe_prompt_row0"]
    idx = np.flatnonzero(pid == pid[row0])
    idx = idx[np.argsort(position[idx], kind="stable")]
    ans = is_answer[idx] == 1

    lm_large = lm_large or load_lm(pair.large_id, pair.large_tag, pick_device())
    stack = Stack(lm_large)
    with torch.no_grad():
        to = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(lm_large.device)
        h = to(Y[idx]).to(torch.float32).unsqueeze(0).clone()
        h[0, to(np.flatnonzero(ans)).long()] = to(
            apply_map(arrays, X[idx][ans])).to(torch.float32)
        pos = to(position[idx]).long().unsqueeze(0)
        hs = stack.run(h.to(stack.dtype), pos, j, stack.n_layers, cache=None)
        got = stack.lm_head(stack.base.norm(hs))[0][-1].float().cpu().numpy()

    want_head = np.array(meta["probe_logits_sha_head"], np.float32)
    diff = float(np.abs(got[:8] - want_head).max())
    # The suffix runs in bf16, which has 8 mantissa bits, so a logit of
    # magnitude m carries a quantisation step of about m * 2^-8 — and the
    # training-time probe went through the autograd graph while this one does
    # not, which is enough to select different GEMM kernels. An absolute
    # threshold is therefore the wrong instrument: at |logit| ~ 16 a single ULP
    # is 0.0625, and the first version of this check failed on exactly that
    # while the argmax matched.
    #
    # So the criterion is (a) the argmax must match, because that is what greedy
    # decoding reads, and (b) the logits must agree to a few ULPs at their own
    # scale. A genuinely mis-saved map — wrong dtype, transposed weight, dropped
    # tensor — moves logits by whole units, not by ULPs, so this still catches
    # what the check exists to catch.
    # The scale that sets the error is the magnitude of the logits actually
    # flowing through the stack, not that of the eight sampled head values —
    # bf16 error accumulates across 14 blocks in proportion to the largest
    # intermediate, and the head happens to hold small values. Using the head's
    # own max under-estimates the tolerance by ~4x and fails on 2 ULPs of pure
    # quantisation, which is what the first two versions of this check did.
    scale = max(1.0, float(np.abs(got).max()))
    tol = max(4 * scale * 2 ** -8, 1e-3)
    out |= {"probe_argmax_saved": meta["probe_logits_argmax"],
            "probe_argmax_reloaded": int(got.argmax()),
            "probe_logit_max_abs_diff": diff,
            "probe_logit_tolerance": tol,
            "probe_logit_scale": scale,
            "reproduces_training_logits":
                bool(int(got.argmax()) == meta["probe_logits_argmax"] and diff <= tol)}
    out["passed"] = bool(out["shapes_ok"] and out["finite"]
                         and out["reproduces_training_logits"])
    return out


class TorchAdapter:
    """The saved map on the accelerator. Applied in float32 (a 2048x3072 matmul
    at bf16 loses enough precision to matter against a residual stream the next
    10 blocks read), then cast back by the caller.

    `taps` tells the stitch which small-model layers to collect; their residuals
    arrive concatenated in the same order the fit used.
    """

    def __init__(self, arrays: dict, meta: dict, device: str):
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(device, torch.float32)
        self.p = {k: t(v) for k, v in arrays.items()}
        self.meta = meta
        self.i, self.j = meta["small_layer_i"], meta["large_layer_j"]
        self.kind = meta.get("kind", "linear")
        self.taps = tuple(meta.get("taps") or (self.i,))

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        p = self.p
        xs = (x.to(torch.float32) - p["mu_x"]) / p["sd_x"]
        out = xs @ p["W"] + p["b"]
        if "W1" in p:
            h = torch.nn.functional.gelu(xs @ p["W1"] + p["b1"], approximate="tanh")
            out = out + h @ p["W2"] + p["b2"]
        return out

    @property
    def n_params(self) -> int:
        """Weights multiplied against every decoded token. mu_x/sd_x/mu_y are
        elementwise shifts, not matmuls, so they are excluded — the same
        convention Stack.n_params uses for the embedding table."""
        return sum(v.numel() for k, v in self.p.items()
                   if k not in ("mu_x", "sd_x", "mu_y"))
