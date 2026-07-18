"""Step 4 — turn the DM residual grids into the answer to Question 1.

Produces three figures (in results/<pair>/figures/), prints a verdict, and
saves the same verdict text to results/<pair>/verdict.txt so the results
folder is self-documenting:

  1. divergence_curve.png — best-match R^2 per small-model layer, divergent vs
     control. The layer where the divergent curve drops while the control curve
     holds is the layer whose representation has "permanently diverged" on
     hallucination prompts. That is the candidate root-cause layer.

  2. heatmap_divergent.png / heatmap_control.png — full (small layer x large
     layer) R^2 grids, showing which large layer each small layer best maps into.

The script is plot-optional: if matplotlib is unavailable it still prints the
numeric verdict from dm_summary.json.
"""

import json

import numpy as np

from q1.config import ModelPair, PAIRS, DEFAULT_PAIR, dm_dir, figures_dir, results_dir


def run(pair: ModelPair) -> None:
    summary_path = dm_dir(pair) / "dm_summary.json"
    if not summary_path.exists():
        raise SystemExit(f"{summary_path} not found — run the train step first.")
    with open(summary_path) as f:
        s = json.load(f)
    if "divergent" not in s:
        raise SystemExit("dm_summary.json has no 'divergent' set (too few divergent "
                         "prompts in the train step) — nothing to analyse.")
    fig_dir = figures_dir(pair)
    fig_dir.mkdir(parents=True, exist_ok=True)
    small, large = pair.small_tag, pair.large_tag

    have_ctrl = "control" in s
    bd = np.array(s["divergent"]["best_r2_last_per_small_layer"])
    bc = np.array(s["control"]["best_r2_last_per_small_layer"]) if have_ctrl else None
    layers = np.arange(pair.n_layers_small + 1)
    div = s.get("divergence", {})
    onset = div.get("onset_layer")
    dlayer = div.get("hallucination_specific_layer", onset)   # root-cause candidate

    # ── text verdict (printed AND saved to results/<pair>/verdict.txt) ────────
    lines = []
    lines.append("=" * 64)
    lines.append("QUESTION 1 — can alignment identify the root cause of hallucination?")
    lines.append(f"                                   pair: {pair.name} ({small} vs {large})")
    lines.append(f"  small: {pair.small_id}")
    lines.append(f"  large: {pair.large_id}")
    lines.append(f"  alignment: {pair.align}"
                 + ("  (one final-token row per prompt — underdetermined fit,"
                    " read R² as indicative)" if pair.align == "prompt" else ""))
    lines.append("=" * 64)
    lines.append(f"\nDivergent set: {s['divergent']['n_prompts']} prompts "
                 f"({s['divergent']['n_test_last_tokens']} held-out answer positions)")
    if have_ctrl:
        lines.append(f"Control set  : {s['control']['n_prompts']} prompts")
    lines.append(f"\n {small} layer | best-R^2 (divergent) | best-R^2 (control) | gap")
    lines.append(" ---------+----------------------+--------------------+-------")
    for i in layers:
        g = (bd[i] - bc[i]) if have_ctrl else float("nan")
        cstr = f"{bc[i]:+.3f}" if have_ctrl else "  n/a "
        mark = ""
        if i == onset:
            mark += "  <- onset"
        if i == dlayer:
            mark += "  <== hallucination-specific (root-cause candidate)"
        lines.append(f"   {i:2d}    |       {bd[i]:+.3f}        |      {cstr}       | {g:+.3f}{mark}")

    if onset is not None:
        early = (f"through layers 0-{onset - 1} the {small} representation is near-perfectly\n"
                 f"linearly translatable into the {large} space on BOTH hallucination\n"
                 f"and control prompts — the small model's early representations are not the\n"
                 f"problem. Translatability then falls off"
                 if onset > 0 else
                 f"translatability of the {small} into the {large} space falls off already")
        lines.append(f"\nVERDICT: {early} from layer {onset} (onset), and on\n"
                     f"hallucination prompts it drops further than on controls, with the largest\n"
                     f"hallucination-specific gap at layer {dlayer} "
                     f"(gap {div.get('max_gap', float('nan')):+.3f}).\n"
                     f"Alignment thus localises the divergence to the layers around {dlayer},\n"
                     f"the candidate root-cause region for the hallucination.")

    verdict = "\n".join(lines) + "\n"
    print(verdict, end="")
    verdict_path = results_dir(pair) / "verdict.txt"
    verdict_path.write_text(verdict, encoding="utf-8")
    print(f"\nWrote {verdict_path}")

    # ── plots ─────────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"\n[plots skipped: matplotlib unavailable: {e}]")
        return

    # Figure 1: divergence curve
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(layers, bd, "o-", color="#c0392b",
            label=f"divergent ({small}-wrong / {large}-right)")
    if have_ctrl:
        ax.plot(layers, bc, "s--", color="#2980b9", label="control (both right)")
    if onset is not None:
        ax.axvline(onset, color="gray", ls=":", lw=1.2)
        ax.annotate(f"onset (L{onset})", xy=(onset, 0.95),
                    xytext=(onset + 0.2, 0.97), color="gray", fontsize=9)
    if dlayer is not None:
        ax.axvline(dlayer, color="#8e44ad", ls="--", lw=1.5)
        ax.annotate(f"hallucination-specific\nroot-cause candidate (L{dlayer})",
                    xy=(dlayer, np.nanmin(bd)),
                    xytext=(max(dlayer - 5.5, 0.2), np.nanmin(bd) + 0.02),
                    color="#8e44ad", fontsize=9)
    ax.set_xlabel(f"{small} hidden layer (0 = embeddings)")
    ax.set_ylabel(f"best-match R²  (DM map into best {large} layer, held-out answer token)")
    ax.set_title(f"Layer-wise translatability of {small} → {large} representations ({pair.name})")
    ax.set_xticks(layers)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "divergence_curve.png", dpi=140)
    plt.close(fig)

    # Figures 2/3: heatmaps
    for name in (["divergent", "control"] if have_ctrl else ["divergent"]):
        g = np.load(dm_dir(pair) / f"grids_{name}.npz")["r2_all"]
        fig, ax = plt.subplots(figsize=(8, 5))
        im = ax.imshow(g, aspect="auto", origin="lower", cmap="viridis",
                       vmin=max(-1.0, np.nanmin(g)), vmax=1.0)
        ax.set_xlabel(f"{large} hidden layer j")
        ax.set_ylabel(f"{small} hidden layer i")
        ax.set_title(f"DM held-out R²(i→j) — {name} set ({pair.name})")
        fig.colorbar(im, ax=ax, label="R²")
        fig.tight_layout()
        fig.savefig(fig_dir / f"heatmap_{name}.png", dpi=140)
        plt.close(fig)

    print(f"\nWrote figures to {fig_dir}/")


if __name__ == "__main__":
    run(PAIRS[DEFAULT_PAIR])
