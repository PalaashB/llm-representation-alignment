"""Step 3 — train layer-wise Direct Matching (DM) adapters.

A DM adapter is an affine map  Y_hat = X_std @ W + b  that translates a small-
model hidden layer (X) into a large-model hidden layer (Y). It is fit in closed
form by ridge regression (the canonical linear representational-alignment /
"stitching" tool). The *held-out* residual of this map measures how translatable
the small model's representation is into the large model's:

    R^2 -> 1   the small layer is (linearly) a rotated/scaled copy of the large layer
    R^2 -> 0   the small layer carries information the large layer's geometry does
               not share — the representations have diverged.

We fit a DM adapter for every (small layer i, large layer j) pair, so each small
layer's *best* attainable R^2 over all large layers tells us whether that layer
still lives in a large-translatable subspace. We do this separately for the
divergent (small-wrong/large-right) set and the both-right control set; the
layer where the divergent curve drops while the control curve does not is the
candidate "permanent divergence" / root-cause layer.

To avoid leakage the train/test split is by *prompt*: tokens from a prompt are
entirely in train or entirely in test. The DM map is fit on train tokens and the
residual is reported on held-out test tokens — both on all test tokens and,
separately, on just the final (answer-generating) token of each test prompt.

Outputs (in results/<pair>/dm/):
    grids_divergent.npz, grids_control.npz  — r2_all/r2_last/nmse_all grids
    dm_summary.json                          — best-match curves + divergence layer
"""

import json

import numpy as np

from q1.config import (
    ModelPair, PAIRS, DEFAULT_PAIR,
    RIDGE_ALPHA, TEST_FRAC, N_SPLITS, TRAIN_TOK_CAP, TEST_TOK_CAP, SEED,
    states_dir, dm_dir,
)


def load_states(pair: ModelPair):
    sdir = states_dir(pair)
    x = np.load(sdir / "x_small.npz")
    y = np.load(sdir / "y_large.npz")
    meta = np.load(sdir / "meta.npz")
    X = [x[f"layer_{L:02d}"] for L in range(pair.n_layers_small + 1)]
    Y = [y[f"layer_{L:02d}"] for L in range(pair.n_layers_large + 1)]
    return X, Y, meta["prompt_index"], meta["set_label"], meta["is_last"]


def split_prompts(prompt_index, row_mask, seed):
    """Split the prompts present in row_mask into train/test prompt-id sets."""
    pids = np.unique(prompt_index[row_mask])
    rng = np.random.default_rng(seed)
    rng.shuffle(pids)
    n_test = max(1, int(round(len(pids) * TEST_FRAC)))
    test_pids = set(pids[:n_test].tolist())
    train_pids = set(pids[n_test:].tolist())
    return train_pids, test_pids, len(pids)


def fit_grid_once(X, Y, prompt_index, is_last, row_mask, seed):
    """
    Fit the full (small layer i) x (large layer j) DM residual grid for the rows
    selected by row_mask, for a single train/test split. Returns dict of
    (nL_small+1 x nL_large+1) arrays.

    For a fixed split + small layer i the same ridge system is solved once and
    reused for every large layer j via the "hat" matrix H = Xte_a (AᵀA+λ)⁻¹ Xtrᵀ,
    so predictions for all target layers are a single batched matmul
    (H @ Y_train). Train/test tokens are capped (last tokens always kept) so the
    control set stays cheap; the per-layer curve is stable well below the full
    token count.
    """
    rng = np.random.default_rng(seed + 7919)
    train_pids, test_pids, n_prompts = split_prompts(prompt_index, row_mask, seed)
    in_train = row_mask & np.isin(prompt_index, list(train_pids))
    in_test = row_mask & np.isin(prompt_index, list(test_pids))
    last_mask = (is_last == 1)

    tr = np.where(in_train)[0]
    if len(tr) > TRAIN_TOK_CAP:
        tr = np.sort(rng.choice(tr, TRAIN_TOK_CAP, replace=False))
    te_last = np.where(in_test & last_mask)[0]
    te_nonlast = np.where(in_test & ~last_mask)[0]
    if len(te_nonlast) > TEST_TOK_CAP:
        te_nonlast = rng.choice(te_nonlast, TEST_TOK_CAP, replace=False)
    te = np.sort(np.concatenate([te_nonlast, te_last]))
    te_is_last = np.isin(te, te_last)            # which test rows are answer positions
    n_tr, n_te = len(tr), len(te)

    nL1, nL3 = len(X), len(Y)
    d3 = Y[0].shape[1]

    # Stack all large-model target layers once: (n, nL3*d3). Train block is mean-centred.
    Ytr = np.concatenate([Y[j][tr].astype(np.float32) for j in range(nL3)], axis=1)
    Yte = np.concatenate([Y[j][te].astype(np.float32) for j in range(nL3)], axis=1)
    mu_y = Ytr.mean(0)
    Ytr -= mu_y                                   # in-place: Ytr now centred
    diff_tot = Yte - mu_y
    ss_tot = (diff_tot ** 2).reshape(n_te, nL3, d3).sum(axis=(0, 2))
    ss_tot_last = (diff_tot[te_is_last] ** 2).reshape(-1, nL3, d3).sum(axis=(0, 2))
    del diff_tot

    r2_all = np.full((nL1, nL3), np.nan)
    r2_last = np.full((nL1, nL3), np.nan)
    nmse_all = np.full((nL1, nL3), np.nan)

    for i in range(nL1):
        Xi = X[i].astype(np.float32)
        Xtr, Xte = Xi[tr], Xi[te]
        mu_x = Xtr.mean(0)
        sd_x = Xtr.std(0) + 1e-6
        Xtr_a = np.concatenate([(Xtr - mu_x) / sd_x, np.ones((n_tr, 1), np.float32)], 1)
        Xte_a = np.concatenate([(Xte - mu_x) / sd_x, np.ones((n_te, 1), np.float32)], 1)

        d = Xtr_a.shape[1]
        # ridge: A = XᵀX + λI, no penalty on the bias column. Standardised
        # columns => diag(XᵀX) ~ n_tr, so scale λ by n_tr.
        reg = np.full(d, RIDGE_ALPHA * n_tr, np.float32)
        reg[-1] = 0.0
        A = Xtr_a.T @ Xtr_a + np.diag(reg)
        L = np.linalg.cholesky(A)
        # M = A⁻¹ Xtrᵀ  (d x n_tr); H = Xte_a M  (n_te x n_tr) maps train -> test
        M = np.linalg.solve(L.T, np.linalg.solve(L, Xtr_a.T))
        H = Xte_a @ M
        pred = H @ Ytr                            # (n_te x nL3*d3), all layers at once
        res = Yte - (pred + mu_y)

        ss_res = (res ** 2).reshape(n_te, nL3, d3).sum(axis=(0, 2))
        r2_all[i] = 1.0 - ss_res / ss_tot
        nmse_all[i] = ss_res / ss_tot
        ss_res_last = (res[te_is_last] ** 2).reshape(-1, nL3, d3).sum(axis=(0, 2))
        r2_last[i] = 1.0 - ss_res_last / ss_tot_last

    return {
        "r2_all": r2_all, "r2_last": r2_last, "nmse_all": nmse_all,
        "n_prompts": n_prompts, "n_train_tokens": n_tr, "n_test_tokens": n_te,
        "n_test_last_tokens": int(te_is_last.sum()),
    }


def fit_grid(X, Y, prompt_index, is_last, row_mask):
    """Average the DM residual grid over N_SPLITS independent prompt-splits so the
    per-layer curves are stable despite a modest number of divergent prompts."""
    accum = None
    counts = {}
    for s in range(N_SPLITS):
        g = fit_grid_once(X, Y, prompt_index, is_last, row_mask, seed=SEED + s)
        if accum is None:
            accum = {k: np.zeros_like(g[k]) for k in ("r2_all", "r2_last", "nmse_all")}
        for k in accum:
            accum[k] += g[k]
        counts = g  # keep the last split's token/prompt counts for reporting
    for k in accum:
        accum[k] /= N_SPLITS
    accum.update({
        "n_prompts": counts["n_prompts"],
        "n_train_tokens": counts["n_train_tokens"],
        "n_test_tokens": counts["n_test_tokens"],
        "n_test_last_tokens": counts["n_test_last_tokens"],
        "n_splits": N_SPLITS,
    })
    return accum


def best_match_curve(grid_r2):
    """For each small layer, the best R^2 over all large layers and which layer."""
    best_r2 = np.nanmax(grid_r2, axis=1)
    best_j = np.nanargmax(grid_r2, axis=1)
    return best_r2, best_j


def find_divergence_layers(best_r2_div, best_r2_ctrl=None, onset_thresh=0.95):
    """
    Localise divergence with two complementary readouts:

      onset_layer   the first small-model layer whose best-match R^2 into the
                    large model's space drops below `onset_thresh` — the depth at
                    which the small model's representation stops being
                    (near-)perfectly linearly translatable at all. This is a
                    generic depth effect.

      hallucination_specific_layer
                    the layer (at or after onset) with the most negative
                    divergent-minus-control gap — i.e. where the small model's
                    representation is *additionally* harder to translate
                    specifically on prompts it hallucinates on, beyond the
                    generic depth effect. This is the candidate root-cause layer.
    """
    div = np.asarray(best_r2_div, dtype=float)
    below = np.where(div < onset_thresh)[0]
    onset = int(below[0]) if len(below) else int(np.nanargmin(div))
    out = {"onset_layer": onset, "onset_thresh": onset_thresh}
    if best_r2_ctrl is not None:
        gap = div - np.asarray(best_r2_ctrl, dtype=float)   # <0 => divergent worse
        masked = gap.copy()
        masked[:onset] = np.inf                              # only consider onset onward
        specific = int(np.nanargmin(masked))
        out["gap"] = gap.tolist()
        out["hallucination_specific_layer"] = specific
        out["max_gap"] = float(gap[specific])
    return out


def run(pair: ModelPair) -> dict:
    X, Y, prompt_index, set_label, is_last = load_states(pair)
    out_dir = dm_dir(pair)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = {"pair": pair.name}
    for name, lbl in [("divergent", 0), ("control", 1)]:
        mask = (set_label == lbl)
        if mask.sum() == 0 or len(np.unique(prompt_index[mask])) < 4:
            print(f"[skip] '{name}' set has too few prompts for a split.")
            continue
        if mask.sum() < pair.dim_small:
            # Happens for prompt-aligned (one row/prompt) pairs: fewer rows than
            # input dims, so the ridge solution is underdetermined. It still
            # runs (the ridge keeps A positive-definite), but treat R² as
            # indicative, not as a well-estimated quantity.
            print(f"[warn] '{name}' set: {int(mask.sum())} rows < {pair.dim_small} "
                  f"input dims — DM fit is underdetermined; R² is indicative only.")
        g = fit_grid(X, Y, prompt_index, is_last, mask)
        np.savez(out_dir / f"grids_{name}.npz",
                 r2_all=g["r2_all"], r2_last=g["r2_last"], nmse_all=g["nmse_all"])
        br_all, bj_all = best_match_curve(g["r2_all"])
        br_last, bj_last = best_match_curve(g["r2_last"])
        out[name] = {
            "n_prompts": g["n_prompts"],
            "n_train_tokens": g["n_train_tokens"],
            "n_test_tokens": g["n_test_tokens"],
            "n_test_last_tokens": g["n_test_last_tokens"],
            "best_r2_all_per_small_layer": br_all.tolist(),
            "best_match_large_layer_all": bj_all.tolist(),
            "best_r2_last_per_small_layer": br_last.tolist(),
            "best_match_large_layer_last": bj_last.tolist(),
        }
        print(f"\n[{name}] prompts={g['n_prompts']} "
              f"train_tok={g['n_train_tokens']} test_tok={g['n_test_tokens']}")
        print(f"  {pair.small_tag} layer :  best-R2(all)  best-R2(last)  ->{pair.large_tag}(all)")
        for i in range(pair.n_layers_small + 1):
            print(f"   {i:2d}      :   {br_all[i]:+.3f}        {br_last[i]:+.3f}        {bj_all[i]:2d}")

    # ── divergence layers (uses last-token best-R^2 curves) ───────────────────
    if "divergent" in out:
        bd = np.array(out["divergent"]["best_r2_last_per_small_layer"])
        bc = np.array(out["control"]["best_r2_last_per_small_layer"]) if "control" in out else None
        div = find_divergence_layers(bd, bc)
        out["divergence"] = div
        out["divergence_layer_small"] = div.get("hallucination_specific_layer", div["onset_layer"])
        print("\n" + "=" * 60)
        print(f"DIVERGENCE ONSET LAYER ({pair.small_tag})       : layer {div['onset_layer']}")
        if "hallucination_specific_layer" in div:
            print(f"HALLUCINATION-SPECIFIC LAYER (max gap) : layer "
                  f"{div['hallucination_specific_layer']} (gap {div['max_gap']:+.3f})")
        print("=" * 60)

    out["config"] = {"ridge_alpha": RIDGE_ALPHA, "test_frac": TEST_FRAC,
                     "seed": SEED, "n_splits": N_SPLITS}
    with open(out_dir / "dm_summary.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_dir}/dm_summary.json and grids_*.npz")
    return out


if __name__ == "__main__":
    run(PAIRS[DEFAULT_PAIR])
