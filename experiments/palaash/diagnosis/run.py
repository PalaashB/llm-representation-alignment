"""Run the Question-1 diagnosis pipeline for a small-vs-large model pair.

From experiments/palaash:

    python -m diagnosis.run                             # default steps, llama pair
    python -m diagnosis.run --pair qwen                 # default steps, qwen pair
    python -m diagnosis.run --pair qwen select extract  # just the model-dependent steps
    python -m diagnosis.run train analyze               # re-fit + re-plot from saved states
    python -m diagnosis.run fit_adapter                 # materialise the selected DM map

Pairs (see diagnosis/config.py):
    llama       Llama-3.2-1B-Instruct  vs Llama-3.2-3B-Instruct   (token-aligned)
    qwen        Qwen2.5-0.5B-Instruct  vs Qwen2.5-3B-Instruct     (token-aligned)
    llama2qwen  Llama-3.2-1B-Instruct  vs Qwen2.5-3B-Instruct     (prompt-aligned)
    qwen2llama  Qwen2.5-0.5B-Instruct  vs Llama-3.2-3B-Instruct   (prompt-aligned)

Steps (outputs under diagnosis/results/<pair>/):
    select       -> generations.csv, selection.json
    extract      -> states/*.npz
    train        -> dm/*.npz, dm/dm_summary.json
    analyze      -> figures/*.png, verdict.txt (+ printed verdict)
    cka          -> cka/*.npz, cka/cka_summary.json, figures/cka_*.png, verdict_cka.txt
    fit_adapter  -> adapters/adapter_i{i}_j{j}.npz + .json   (materialised DM map)

The first five steps *diagnose*. `fit_adapter` materialises the map the analysis
selected — the diagnostic grid never forms W explicitly — for token-aligned pairs
only, so it is not in the default run; ask for it by name.

Stitching is a separate project and lives in the sibling `stitching/` package;
it fits its own adapters and does not read anything produced here.

The model-dependent steps (select, extract) share a single load of both models.
"""

import argparse

from diagnosis.config import PAIRS, DEFAULT_PAIR

STEP_ORDER = ["select", "extract", "train", "analyze", "cka", "fit_adapter"]
DEFAULT_STEPS = ["select", "extract", "train", "analyze", "cka"]
NEEDS_MODELS = {"select", "extract"}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("steps", nargs="*", metavar="step",
                    help=f"steps to run, from {STEP_ORDER} (default: {DEFAULT_STEPS})")
    ap.add_argument("--pair", choices=sorted(PAIRS), default=DEFAULT_PAIR,
                    help=f"model pair to run (default: {DEFAULT_PAIR})")
    ap.add_argument("--i", type=int, default=None,
                    help="small layer to map from (fit_adapter; "
                         "default: divergence_layer_small)")
    ap.add_argument("--j", type=int, default=None,
                    help="large layer to map into (fit_adapter; "
                         "default: its depth-matched best match)")
    args = ap.parse_args()
    unknown = [s for s in args.steps if s not in STEP_ORDER]
    if unknown:
        ap.error(f"unknown step(s) {unknown}; choose from {STEP_ORDER}")
    steps = [s for s in STEP_ORDER if s in (args.steps or DEFAULT_STEPS)]
    pair = PAIRS[args.pair]

    lm_small = lm_large = None
    if NEEDS_MODELS & set(steps):
        from common.model_utils import load_pair
        lm_small, lm_large = load_pair(pair)

    for step in steps:
        print("\n" + "#" * 70 + f"\n# step: {step}  (pair: {pair.name})\n" + "#" * 70)
        if step == "select":
            from diagnosis import select_prompts
            select_prompts.run(pair, lm_small, lm_large)
        elif step == "extract":
            from diagnosis import extract_states
            extract_states.run(pair, lm_small, lm_large)
        elif step == "train":
            from diagnosis import train_dm
            train_dm.run(pair)
        elif step == "analyze":
            from diagnosis import analyze
            analyze.run(pair)
        elif step == "cka":
            from diagnosis import cka
            cka.run(pair)
        elif step == "fit_adapter":
            from diagnosis import fit_adapter
            fit_adapter.run(pair, args.i, args.j)


if __name__ == "__main__":
    main()
