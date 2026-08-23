"""The large->small adapter: weighted ridge, fit for generation.

The map is affine on standardised inputs,

    Y_hat = ((X - mu_x) / sd_x) @ W + b

taking the large model's layer-j residual to the small model's layer-i residual
(so W is dim_large x dim_small — the transpose of the sibling package's shape).

Three choices carried over from the sibling, all of which were measured there
rather than assumed:

1. Answer rows are up-weighted. They are a small minority of rows but the entire
   population the adapter faces once decoding starts.
2. Ridge alpha is light. Shrinkage pulls every prediction toward the mean
   residual, which is what turns a stitched answer into a generic continuation.
3. Optional radial norm matching, folded into (W, b) so the shipped map stays
   affine and free at inference.

Quality is reported prompt-level held out and split by row type. The all-token
number is the one that misled the earlier work — 0.91 all-token against 0.34 on
answer tokens, on a map whose generations were unusable — so `answer` is the
number to read.
"""

from __future__ import annotations

import json

import numpy as np
import torch

from stitching_large_to_small.config import (
    ADAPTER_TEST_FRAC, ANSWER_WEIGHT, NORM_MATCH, RIDGE_ALPHA, SEED, Bank, Pair,
    adapter_path, adapters_dir, validate_layers,
)
from stitching_large_to_small.data import load_layer_pair


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


def fit(pair: Pair, i: int, j: int, bank: Bank, alpha: float = RIDGE_ALPHA,
        answer_weight: float = ANSWER_WEIGHT, norm_match: bool = NORM_MATCH,
        verbose: bool = True) -> dict:
    """Fit and save the large->small adapter for one (i, j)."""
    validate_layers(pair, i, j)
    X, Y, pid, is_answer = load_layer_pair(pair, i, j, bank)
    w = np.where(is_answer == 1, answer_weight, 1.0).astype(np.float32)

    # ── honest quality: prompt-level holdout, refit from scratch ──────────────
    pids = np.unique(pid)
    rng = np.random.default_rng(SEED)
    rng.shuffle(pids)
    n_te = max(1, int(round(len(pids) * ADAPTER_TEST_FRAC)))
    te = np.isin(pid, pids[:n_te])
    tr = ~te
    Wh, bh, mxh, sxh, myh = weighted_ridge(X[tr], Y[tr], w[tr], alpha)
    if norm_match:
        a = is_answer[tr] == 1
        Wh, bh = rescale(Wh, bh, myh,
                         norm_gain(apply_np(X[tr][a], Wh, bh, mxh, sxh), Y[tr][a], myh))
    Yh = apply_np(X[te], Wh, bh, mxh, sxh)
    ans_te = is_answer[te] == 1
    held = {"all": quality(Yh, Y[te]),
            "answer": quality(Yh[ans_te], Y[te][ans_te]) if ans_te.any() else None,
            "prompt": quality(Yh[~ans_te], Y[te][~ans_te]) if (~ans_te).any() else None}

    # ── the shipped map: every captured row ───────────────────────────────────
    W, b, mu_x, sd_x, mu_y = weighted_ridge(X, Y, w, alpha)
    gain = 1.0
    if norm_match:
        a = is_answer == 1
        gain = norm_gain(apply_np(X[a], W, b, mu_x, sd_x), Y[a], mu_y)
        W, b = rescale(W, b, mu_y, gain)
    in_sample = quality(apply_np(X, W, b, mu_x, sd_x), Y)

    adapters_dir(pair, bank).mkdir(parents=True, exist_ok=True)
    path = adapter_path(pair, i, j, bank)
    np.savez(path, W=W.astype(np.float32), b=b.astype(np.float32),
             mu_x=mu_x.astype(np.float32), sd_x=sd_x.astype(np.float32),
             mu_y=mu_y.astype(np.float32))
    meta = {
        "pair": pair.name, "bank": bank.name, "direction": "large->small",
        "small_layer_i": i, "large_layer_j": j,
        "dim_in_large": pair.dim_large, "dim_out_small": pair.dim_small,
        "depth_matched_i": pair.depth_matched_i(j),
        "form": "Y_hat = ((X - mu_x) / sd_x) @ W + b",
        "injection_point": f"input to small decoder block {i}",
        "source_point": f"input to large decoder block {j}",
        "ridge_alpha": alpha, "answer_weight": answer_weight,
        "norm_match": norm_match, "norm_gain": gain,
        "n_rows": int(len(X)), "n_answer_rows": int((is_answer == 1).sum()),
        "n_prompts": int(len(pids)), "seed": SEED,
        "in_sample": in_sample, "held_out": held,
    }
    with open(path.with_suffix(".json"), "w") as f:
        json.dump(meta, f, indent=2)

    if verbose:
        a_, p_ = held["answer"], held["prompt"]
        print(f"[fit] {pair.name}/{bank.name}  large L{j} -> small L{i}  "
              f"rows={len(X)}  gain={gain:.3f}  "
              f"(depth-matched i for j={j} would be {pair.depth_matched_i(j)})")
        print(f"  held-out R2   all={held['all']['r2']:+.4f}"
              + (f"  answer={a_['r2']:+.4f} (n={a_['n_rows']})" if a_ else "")
              + (f"  prompt={p_['r2']:+.4f}" if p_ else ""))
        print(f"  held-out cos  all={held['all']['cosine_mean']:.4f}"
              + (f"  answer={a_['cosine_mean']:.4f}" if a_ else ""))
        print(f"  wrote {path.name}")
    return meta


def load(pair: Pair, i: int, j: int, bank: Bank) -> tuple[dict, dict]:
    path = adapter_path(pair, i, j, bank)
    if not path.exists():
        raise SystemExit(
            f"{path.name} not found — run `python -m stitching_large_to_small.run fit "
            f"--pair {pair.name} --bank {bank.name} --i {i} --j {j}` first.")
    z = np.load(path)
    with open(path.with_suffix(".json")) as f:
        meta = json.load(f)
    return {k: z[k] for k in ("W", "b", "mu_x", "sd_x", "mu_y")}, meta


def reload_quality(pair: Pair, i: int, j: int, bank: Bank) -> dict:
    """Re-load the saved coefficients and re-score them on held-out rows.

    A gating check, not a metric: it catches a map that was saved wrong, or
    loaded into the wrong dtype/shape, before any accuracy number is reported.
    The fit already reported these values; this recomputes them from what is
    actually on disk.
    """
    arrays, meta = load(pair, i, j, bank)
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
    return {"check": "adapter_reload", "r2_all": q_all["r2"],
            "cosine_all": q_all["cosine_mean"],
            "r2_answer": q_ans["r2"] if q_ans else None,
            "cosine_answer": q_ans["cosine_mean"] if q_ans else None,
            "held_out_answer_r2_from_fit":
                (meta.get("held_out", {}).get("answer") or {}).get("r2"),
            "shapes_ok": (arrays["W"].shape == (pair.dim_large, pair.dim_small)
                          and arrays["b"].shape == (pair.dim_small,)),
            "finite": bool(all(np.isfinite(v).all() for v in arrays.values())),
            "passed": bool(np.isfinite(q_all["r2"])
                           and arrays["W"].shape == (pair.dim_large, pair.dim_small)
                           and all(np.isfinite(v).all() for v in arrays.values()))}


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
