"""Step 5 — materialise and persist one DM adapter for stitching.

`train_dm.py` answers "how translatable is small layer i into large layer j?"
and deliberately never forms W: it solves the ridge system once per small layer
and pushes the *training targets* through a hat matrix, which yields R² for all
target layers at once but throws the map away. Stitching needs the map, so this
step re-solves the same ridge problem explicitly for a single (i, j) and saves
the coefficients.

The saved map is affine on standardised inputs:

    Y_hat = ((X - mu_x) / sd_x) @ W + b

Fitting differs from the diagnostic in one deliberate way: the diagnostic holds
out 30% of prompts and averages over 6 splits because it is *estimating a
quantity*. Here we want the best usable map, so the saved W is fit on every row
of the chosen set. To keep the reported quality honest, we separately refit on a
prompt-level split and report held-out cosine / relative L2 — those numbers
describe the map, they are not what gets shipped.

Default (i, j) comes from dm_summary.json: i = divergence_layer_small, and
j = best_match_large_layer_last[i], i.e. the depth-matched target the analysis
selected. j is rejected if it is the embedding layer or the post-norm index,
neither of which is a residual stream you can inject into.

Outputs (in results/<pair>/adapters/):
    adapter_i{i}_j{j}.npz   — W, b, mu_x, sd_x, mu_y
    adapter_i{i}_j{j}.json  — provenance + fit/holdout quality
"""

import argparse
import json

import numpy as np

from diagnosis.config import (
    ModelPair, PAIRS, DEFAULT_PAIR,
    RIDGE_ALPHA, SEED, ADAPTER_FIT_SET, ADAPTER_TEST_FRAC,
    states_dir, dm_dir, adapters_dir, adapter_path,
)

FIT_SETS = ("control", "all", "divergent")
SET_LABEL = {"divergent": 0, "control": 1}


def load_layer_pair(pair: ModelPair, i: int, j: int):
    """Load just the two layers we need, plus row metadata (avoids paging in the
    whole state file: x_small alone is 17 layers x N x 2048)."""
    sdir = states_dir(pair)
    with np.load(sdir / "x_small.npz") as x:
        X = x[f"layer_{i:02d}"].astype(np.float32)
    with np.load(sdir / "y_large.npz") as y:
        Y = y[f"layer_{j:02d}"].astype(np.float32)
    meta = np.load(sdir / "meta.npz")
    return X, Y, meta["prompt_index"], meta["set_label"], meta["is_last"]


def solve_ridge(X, Y, alpha=RIDGE_ALPHA):
    """Explicit ridge normal equations, same regularisation convention as
    train_dm: inputs standardised, penalty scaled by n so it stays comparable
    across row counts, bias column unpenalised.

    Returns (W, b, mu_x, sd_x) for  Y_hat = ((X - mu_x) / sd_x) @ W + b.
    """
    n = X.shape[0]
    mu_x = X.mean(0)
    sd_x = X.std(0) + 1e-6
    Xs = (X - mu_x) / sd_x
    Xa = np.concatenate([Xs, np.ones((n, 1), np.float32)], axis=1)

    mu_y = Y.mean(0)
    Yc = Y - mu_y

    d = Xa.shape[1]
    reg = np.full(d, alpha * n, np.float32)
    reg[-1] = 0.0                                  # never penalise the intercept
    A = Xa.T @ Xa + np.diag(reg)
    L = np.linalg.cholesky(A)
    Wa = np.linalg.solve(L.T, np.linalg.solve(L, Xa.T @ Yc))

    W = Wa[:-1]
    b = Wa[-1] + mu_y                              # fold the centring back into the bias
    return W, b, mu_x, sd_x, mu_y


def apply_adapter(X, W, b, mu_x, sd_x):
    return ((X - mu_x) / sd_x) @ W + b


def quality(Y_hat, Y):
    """Per-row cosine and relative L2, plus the R² the diagnostic would report."""
    num = (Y_hat * Y).sum(1)
    den = np.linalg.norm(Y_hat, axis=1) * np.linalg.norm(Y, axis=1) + 1e-12
    cos = num / den
    rel_l2 = np.linalg.norm(Y_hat - Y, axis=1) / (np.linalg.norm(Y, axis=1) + 1e-12)
    ss_res = float(((Y_hat - Y) ** 2).sum())
    ss_tot = float(((Y - Y.mean(0)) ** 2).sum())
    return {
        "cosine_mean": float(cos.mean()), "cosine_median": float(np.median(cos)),
        "rel_l2_mean": float(rel_l2.mean()), "rel_l2_median": float(np.median(rel_l2)),
        "r2": float(1.0 - ss_res / ss_tot), "n_rows": int(Y.shape[0]),
    }


def default_layers(pair: ModelPair) -> tuple[int, int]:
    """(i, j) from the analysis: the hallucination-specific layer and its
    depth-matched target."""
    path = dm_dir(pair) / "dm_summary.json"
    if not path.exists():
        raise SystemExit(f"{path} not found — run the train step first, or pass --i/--j.")
    with open(path) as f:
        s = json.load(f)
    if "divergence_layer_small" not in s or "divergent" not in s:
        raise SystemExit(f"{path} has no divergence layer — pass --i/--j explicitly.")
    i = int(s["divergence_layer_small"])
    j = int(s["divergent"]["best_match_large_layer_last"][i])
    return i, j


def validate_layers(pair: ModelPair, i: int, j: int) -> None:
    if not 0 <= i <= pair.n_layers_small:
        raise SystemExit(f"--i {i} out of range for {pair.small_tag} (0..{pair.n_layers_small})")
    if j <= 0:
        raise SystemExit(f"--j {j} is the embedding layer (or negative); it is not an "
                         f"injectable residual stream.")
    if j >= pair.n_layers_large:
        raise SystemExit(
            f"--j {j} is the post-model.norm hidden state (index {pair.n_layers_large}), "
            f"not a residual stream, so it cannot be injected into. "
            f"Valid targets are 1..{pair.n_layers_large - 1}.")


def run(pair: ModelPair, i: int | None = None, j: int | None = None,
        fit_set: str = ADAPTER_FIT_SET, alpha: float = RIDGE_ALPHA) -> dict:
    if pair.align != "token":
        raise SystemExit(
            f"pair '{pair.name}' uses prompt-level alignment ({pair.align}). Stitching v1 "
            f"is token-aligned only — cross-family pairs have one row per prompt, which is "
            f"far too few to fit a usable {pair.dim_small}->{pair.dim_large} map.")
    if fit_set not in FIT_SETS:
        raise SystemExit(f"--fit-set must be one of {FIT_SETS}")

    if i is None or j is None:
        di, dj = default_layers(pair)
        i = di if i is None else i
        j = dj if j is None else j
    validate_layers(pair, i, j)

    X, Y, prompt_index, set_label, is_last = load_layer_pair(pair, i, j)
    mask = np.ones(len(X), bool) if fit_set == "all" else (set_label == SET_LABEL[fit_set])
    if mask.sum() == 0:
        raise SystemExit(f"fit set '{fit_set}' has no rows.")
    Xf, Yf = X[mask], Y[mask]
    pid_f = prompt_index[mask]
    print(f"[fit] {pair.name}  small L{i} -> large L{j}  set={fit_set}  "
          f"rows={len(Xf)}  ({pair.dim_small} -> {pair.dim_large})")

    # ── the shipped map: every row of the chosen set ──────────────────────────
    W, b, mu_x, sd_x, mu_y = solve_ridge(Xf, Yf, alpha)
    in_sample = quality(apply_adapter(Xf, W, b, mu_x, sd_x), Yf)

    # ── honest quality estimate: prompt-level holdout, refit from scratch ─────
    pids = np.unique(pid_f)
    rng = np.random.default_rng(SEED)
    rng.shuffle(pids)
    n_test = max(1, int(round(len(pids) * ADAPTER_TEST_FRAC)))
    test_pids = set(pids[:n_test].tolist())
    te = np.isin(pid_f, list(test_pids))
    tr = ~te
    held_out = held_out_last = None
    if tr.sum() > 0 and te.sum() > 0:
        Wh, bh, mxh, sxh, _ = solve_ridge(Xf[tr], Yf[tr], alpha)
        Yh = apply_adapter(Xf[te], Wh, bh, mxh, sxh)
        held_out = quality(Yh, Yf[te])
        last_te = (is_last[mask][te] == 1)
        if last_te.sum() > 0:
            held_out_last = quality(Yh[last_te], Yf[te][last_te])

    out_dir = adapters_dir(pair)
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = adapter_path(pair, i, j)
    np.savez(npz_path, W=W.astype(np.float32), b=b.astype(np.float32),
             mu_x=mu_x.astype(np.float32), sd_x=sd_x.astype(np.float32),
             mu_y=mu_y.astype(np.float32))

    meta = {
        "pair": pair.name, "small_id": pair.small_id, "large_id": pair.large_id,
        "align": pair.align,
        "small_layer_i": i, "large_layer_j": j,
        "dim_small": pair.dim_small, "dim_large": pair.dim_large,
        "n_layers_small": pair.n_layers_small, "n_layers_large": pair.n_layers_large,
        "form": "Y_hat = ((X - mu_x) / sd_x) @ W + b",
        "injection_point": (f"input to large decoder block {j} "
                            f"(== hidden_states[{j}] with output_hidden_states=True)"),
        "fit_set": fit_set, "n_fit_rows": int(len(Xf)),
        "n_fit_prompts": int(len(pids)),
        "ridge_alpha": alpha, "seed": SEED,
        "holdout_frac_for_reporting": ADAPTER_TEST_FRAC,
        "in_sample": in_sample,
        "held_out": held_out,
        "held_out_last_token": held_out_last,
    }
    json_path = npz_path.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  in-sample : R2={in_sample['r2']:+.4f}  cos={in_sample['cosine_mean']:.4f}  "
          f"relL2={in_sample['rel_l2_mean']:.4f}")
    if held_out:
        print(f"  held-out  : R2={held_out['r2']:+.4f}  cos={held_out['cosine_mean']:.4f}  "
              f"relL2={held_out['rel_l2_mean']:.4f}  (n={held_out['n_rows']})")
    if held_out_last:
        print(f"  held-out* : R2={held_out_last['r2']:+.4f}  cos={held_out_last['cosine_mean']:.4f}  "
              f"relL2={held_out_last['rel_l2_mean']:.4f}  (n={held_out_last['n_rows']} answer tokens)")
    print(f"Wrote {npz_path}\n      {json_path}")
    return meta


def load_adapter(pair: ModelPair, i: int, j: int):
    """Load a saved adapter as (arrays dict, metadata dict)."""
    npz_path = adapter_path(pair, i, j)
    if not npz_path.exists():
        raise SystemExit(f"{npz_path} not found — run the fit_adapter step first.")
    z = np.load(npz_path)
    arrays = {k: z[k] for k in ("W", "b", "mu_x", "sd_x", "mu_y")}
    with open(npz_path.with_suffix(".json")) as f:
        meta = json.load(f)
    return arrays, meta


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair", choices=sorted(PAIRS), default=DEFAULT_PAIR)
    ap.add_argument("--i", type=int, default=None,
                    help="small-model layer (default: divergence_layer_small from dm_summary.json)")
    ap.add_argument("--j", type=int, default=None,
                    help="large-model layer (default: its depth-matched best match)")
    ap.add_argument("--fit-set", choices=FIT_SETS, default=ADAPTER_FIT_SET)
    ap.add_argument("--alpha", type=float, default=RIDGE_ALPHA)
    a = ap.parse_args()
    run(PAIRS[a.pair], a.i, a.j, a.fit_set, a.alpha)


if __name__ == "__main__":
    main()
