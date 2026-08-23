"""Step 5 — CKA (Centered Kernel Alignment) analogue of the DM grid.

Why CKA next to DM: the DM adapter answers a *predictive* question — "can a
ridge-fit affine map translate small-layer i into large-layer j?" — and that
fit is badly underdetermined whenever there are fewer paired rows than input
dimensions (the cross-family, prompt-aligned pairs: ~130 rows vs 896–2048
dims). CKA instead measures the similarity of the two layers' representational
*geometries* directly from their Gram matrices: no fitted map, no train/test
split, symmetric in X and Y, agnostic to the two widths, and invariant to
rotation and isotropic scaling. It is the standard tool for comparing
representations across different architectures (Kornblith et al. 2019), so it
stays well-defined exactly where the DM fit is weakest.

We use the *debiased* linear-kernel CKA (unbiased HSIC estimator of Song et
al. 2012, as used for minibatch CKA in Kornblith et al.). This matters here
because the divergent and control sets have very different row counts, and the
biased estimator's O(1/n) offset would masquerade as a divergent-vs-control
gap.

The readout mirrors the DM step: a (small layer i) x (large layer j) grid per
set, each small layer's best match over all large layers, and the layer where
the divergent curve falls furthest below the control curve. Two grids are
computed per set:

    cka_all  — over (up to CKA_ROW_CAP) token rows: stable, used for heatmaps
    cka_last — over final answer-token rows only (one per prompt): the
               headline curve, directly comparable across alignment modes
               (for prompt-aligned pairs the two grids coincide).

Outputs (in results/<pair>/cka/):
    grids_divergent.npz / grids_control.npz — cka_all / cka_last grids
    cka_summary.json — best-match curves + divergence readout
plus figures/cka_curve.png, figures/cka_heatmap_*.png and verdict_cka.txt.
"""

import json

import numpy as np

from diagnosis.config import (
    ModelPair, PAIRS, DEFAULT_PAIR, CKA_ROW_CAP, SEED,
    cka_dir, figures_dir, results_dir,
)
from diagnosis.train_dm import load_states


def _gram_stats(Z: np.ndarray):
    """Zero-diagonal linear Gram matrix of Z plus its row sums and total sum
    (the reusable pieces of the unbiased HSIC estimator)."""
    K = Z @ Z.T
    np.fill_diagonal(K, 0.0)
    r = K.sum(axis=1)
    return K, r, float(r.sum())


def _hsic1(KA, rA, sA, KB, rB, sB, n: int) -> float:
    """Unbiased HSIC estimator (Song et al. 2012) from zero-diagonal Grams."""
    return float(
        (np.vdot(KA, KB)
         + sA * sB / ((n - 1) * (n - 2))
         - 2.0 * (rA @ rB) / (n - 2))
        / (n * (n - 3))
    )


def _self_hsic(K, r, s, n: int) -> float:
    """Self-HSIC with a degeneracy guard. If all rows are (near-)identical —
    e.g. layer-0 embeddings of last-token rows, which are all the same chat-
    template token — the estimator cancels to ~0 up to float error and the CKA
    ratio would be garbage; return NaN so those cells are marked undefined."""
    h = _hsic1(K, r, s, K, r, s, n)
    scale = float(np.vdot(K, K)) / (n * (n - 3))
    return h if h > 1e-8 * scale else float("nan")


def cka_grid(X: list, Y: list, rows: np.ndarray) -> np.ndarray:
    """Debiased linear CKA for every (small layer i, large layer j) pair over
    the given row indices. Grams are computed in float64 (row-norm products
    summed over n^2 entries overflow float32 precision)."""
    n = len(rows)
    assert n >= 4, "unbiased HSIC needs at least 4 rows"
    xs = []
    for Xi in X:
        K, r, s = _gram_stats(Xi[rows].astype(np.float64))
        xs.append((K, r, s, _self_hsic(K, r, s, n)))
    grid = np.full((len(X), len(Y)), np.nan)
    for j, Yj in enumerate(Y):
        L, rL, sL = _gram_stats(Yj[rows].astype(np.float64))
        hL = _self_hsic(L, rL, sL, n)
        if not np.isfinite(hL):
            continue
        for i, (K, rK, sK, hK) in enumerate(xs):
            if np.isfinite(hK):
                grid[i, j] = _hsic1(K, rK, sK, L, rL, sL, n) / np.sqrt(hK * hL)
    return grid


def best_match_curve(grid: np.ndarray):
    """Row-wise best CKA and argmax large layer. Unlike the DM version this
    tolerates all-NaN rows: at layer 0 of token-aligned pairs every last-token
    row is the same chat-template token, so the embeddings have zero variance
    and CKA is undefined (NaN curve value, -1 best layer)."""
    ok = ~np.isnan(grid).all(axis=1)
    best = np.full(grid.shape[0], np.nan)
    best_j = np.full(grid.shape[0], -1, dtype=int)
    best[ok] = np.nanmax(grid[ok], axis=1)
    best_j[ok] = np.nanargmax(grid[ok], axis=1)
    return best, best_j


def _jsonable(x):
    """Replace non-finite floats with None so the summary stays strict JSON."""
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_jsonable(v) for v in x]
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def run(pair: ModelPair) -> dict:
    X, Y, prompt_index, set_label, is_last = load_states(pair)
    out_dir = cka_dir(pair)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    out = {"pair": pair.name, "method": "debiased linear CKA"}

    for name, lbl in [("divergent", 0), ("control", 1)]:
        mask = (set_label == lbl)
        rows_last = np.where(mask & (is_last == 1))[0]
        rows_all = np.where(mask)[0]
        if len(rows_all) > CKA_ROW_CAP:
            rows_all = np.sort(rng.choice(rows_all, CKA_ROW_CAP, replace=False))
        if len(rows_last) < 4:
            print(f"[skip] '{name}' set has {len(rows_last)} prompts (< 4 needed for CKA).")
            continue

        g_all = cka_grid(X, Y, rows_all)
        g_last = g_all if len(rows_all) == len(rows_last) and (rows_all == rows_last).all() \
            else cka_grid(X, Y, rows_last)
        np.savez(out_dir / f"grids_{name}.npz", cka_all=g_all, cka_last=g_last)

        bc_all, bj_all = best_match_curve(g_all)
        bc_last, bj_last = best_match_curve(g_last)
        out[name] = {
            "n_prompts": int(len(rows_last)),
            "n_rows_all": int(len(rows_all)),
            "best_cka_all_per_small_layer": bc_all.tolist(),
            "best_match_large_layer_all": bj_all.tolist(),
            "best_cka_last_per_small_layer": bc_last.tolist(),
            "best_match_large_layer_last": bj_last.tolist(),
        }
        print(f"\n[{name}] prompts={len(rows_last)} rows_all={len(rows_all)}")
        print(f"  {pair.small_tag} layer :  best-CKA(all)  best-CKA(last)  ->{pair.large_tag}(all)")
        for i in range(pair.n_layers_small + 1):
            print(f"   {i:2d}      :   {bc_all[i]:+.3f}         {bc_last[i]:+.3f}        {bj_all[i]:2d}")

    if "divergent" in out and "control" in out:
        # Unlike the DM curves, CKA has no "translatability collapse", so there
        # is no onset notion — the readout is simply the layer where the
        # divergent geometry is furthest below the control geometry.
        bd = np.array(out["divergent"]["best_cka_last_per_small_layer"], dtype=float)
        bc = np.array(out["control"]["best_cka_last_per_small_layer"], dtype=float)
        gap = bd - bc
        layer = int(np.nanargmin(gap))
        out["divergence"] = {"gap": gap.tolist(), "max_gap_layer": layer,
                             "max_gap": float(gap[layer])}
        out["divergence_layer_small"] = layer

    out["config"] = {"row_cap": CKA_ROW_CAP, "seed": SEED}
    with open(out_dir / "cka_summary.json", "w") as f:
        json.dump(_jsonable(out), f, indent=2)
    print(f"\nWrote {out_dir}/cka_summary.json and grids_*.npz")

    if "divergent" in out:
        _verdict_and_figures(pair, out)
    return out


def _verdict_and_figures(pair: ModelPair, s: dict) -> None:
    small, large = pair.small_tag, pair.large_tag
    have_ctrl = "control" in s
    bd = np.array(s["divergent"]["best_cka_last_per_small_layer"], dtype=float)
    bc = (np.array(s["control"]["best_cka_last_per_small_layer"], dtype=float)
          if have_ctrl else None)
    layers = np.arange(pair.n_layers_small + 1)
    div = s.get("divergence", {})
    dlayer = div.get("max_gap_layer")

    lines = []
    lines.append("=" * 64)
    lines.append("CKA readout — geometry similarity, no fitted map, no split")
    lines.append(f"                                   pair: {pair.name} ({small} vs {large})")
    lines.append(f"  small: {pair.small_id}")
    lines.append(f"  large: {pair.large_id}")
    lines.append(f"  alignment: {pair.align}; metric: debiased linear CKA on final")
    lines.append(f"  answer-token rows (one per prompt) — well-defined even when the")
    lines.append(f"  DM ridge fit is underdetermined (rows < input dims).")
    lines.append("=" * 64)
    lines.append(f"\nDivergent set: {s['divergent']['n_prompts']} prompts")
    if have_ctrl:
        lines.append(f"Control set  : {s['control']['n_prompts']} prompts")
    lines.append(f"\n {small} layer | best-CKA (divergent) | best-CKA (control) | gap")
    lines.append(" ---------+----------------------+--------------------+-------")
    for i in layers:
        g = (bd[i] - bc[i]) if have_ctrl else float("nan")
        cstr = f"{bc[i]:+.3f}" if have_ctrl else "  n/a "
        mark = "  <== max hallucination-specific gap" if i == dlayer else ""
        lines.append(f"   {i:2d}    |       {bd[i]:+.3f}        |      {cstr}       | {g:+.3f}{mark}")
    if dlayer is not None and have_ctrl:
        lines.append(f"\nLargest divergent-minus-control CKA gap at layer {dlayer} "
                     f"(gap {div.get('max_gap', float('nan')):+.3f}).")

    verdict = "\n".join(lines) + "\n"
    print(verdict, end="")
    verdict_path = results_dir(pair) / "verdict_cka.txt"
    verdict_path.write_text(verdict, encoding="utf-8")
    print(f"\nWrote {verdict_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[plots skipped: matplotlib unavailable: {e}]")
        return
    fig_dir = figures_dir(pair)
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(layers, bd, "o-", color="#c0392b",
            label=f"divergent ({small}-wrong / {large}-right)")
    if have_ctrl:
        ax.plot(layers, bc, "s--", color="#2980b9", label="control (both right)")
    if dlayer is not None:
        ax.axvline(dlayer, color="#8e44ad", ls="--", lw=1.5)
        ax.annotate(f"max hallucination-specific\ngap (L{dlayer})",
                    xy=(dlayer, np.nanmin(bd)),
                    xytext=(max(dlayer - 5.5, 0.2), np.nanmin(bd) + 0.02),
                    color="#8e44ad", fontsize=9)
    ax.set_xlabel(f"{small} hidden layer (0 = embeddings)")
    ax.set_ylabel(f"best-match debiased linear CKA (vs best {large} layer, answer tokens)")
    ax.set_title(f"Layer-wise CKA similarity of {small} vs {large} representations ({pair.name})")
    ax.set_xticks(layers)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "cka_curve.png", dpi=140)
    plt.close(fig)

    for name in (["divergent", "control"] if have_ctrl else ["divergent"]):
        g = np.load(cka_dir(pair) / f"grids_{name}.npz")["cka_all"]
        fig, ax = plt.subplots(figsize=(8, 5))
        im = ax.imshow(g, aspect="auto", origin="lower", cmap="viridis",
                       vmin=max(-1.0, float(np.nanmin(g))), vmax=1.0)
        ax.set_xlabel(f"{large} hidden layer j")
        ax.set_ylabel(f"{small} hidden layer i")
        ax.set_title(f"debiased linear CKA(i, j) — {name} set ({pair.name})")
        fig.colorbar(im, ax=ax, label="CKA")
        fig.tight_layout()
        fig.savefig(fig_dir / f"cka_heatmap_{name}.png", dpi=140)
        plt.close(fig)
    print(f"Wrote CKA figures to {fig_dir}/")


if __name__ == "__main__":
    run(PAIRS[DEFAULT_PAIR])
