"""Run the Question-1 pipeline for a small-vs-large model pair.

    python run_q1.py                             # diagnosis steps, llama pair
    python run_q1.py --pair qwen                 # diagnosis steps, qwen pair
    python run_q1.py --pair qwen select extract  # just the model-dependent steps
    python run_q1.py train analyze               # re-fit + re-plot from saved states
    python run_q1.py fit_adapter stitch          # save an adapter, then stitch with it
    python run_q1.py fit_adapter stitch_fast     # ... then benchmark the early-exit stitch

Pairs (see q1/config.py):
    llama       Llama-3.2-1B-Instruct  vs Llama-3.2-3B-Instruct   (token-aligned)
    qwen        Qwen2.5-0.5B-Instruct  vs Qwen2.5-3B-Instruct     (token-aligned)
    llama2qwen  Llama-3.2-1B-Instruct  vs Qwen2.5-3B-Instruct     (prompt-aligned)
    qwen2llama  Qwen2.5-0.5B-Instruct  vs Llama-3.2-3B-Instruct   (prompt-aligned)

Steps (outputs under results/<pair>/):
    select       -> generations.csv, selection.json
    extract      -> states/*.npz
    train        -> dm/*.npz, dm/dm_summary.json
    analyze      -> figures/*.png, verdict.txt (+ printed verdict)
    cka          -> cka/*.npz, cka/cka_summary.json, figures/cka_*.png, verdict_cka.txt
    fit_adapter  -> adapters/adapter_i{i}_j{j}.npz + .json   (materialised DM map)
    stitch       -> stitch/stitch_i{i}_j{j}.json/.csv, stitch/checks_i{i}_j{j}.json
    stitch_fast  -> stitch/fast_i{i}_j{j}.json/.csv, stitch/fast_checks_i{i}_j{j}.json

The first five steps *diagnose*; fit_adapter + stitch + stitch_fast *act* on the
diagnosis by injecting the small model's diverging layer into the large model's
residual stream. They default to the (i, j) the analysis selected and are
token-aligned pairs only, so they are not in the default run — ask for them by
name. The two stitch steps are different claims, not versions of one:

    stitch       mechanism check — runs both full models, no cache, no speedup
    stitch_fast  the early-exit path — skips large blocks 0..j-1, KV-cached,
                 and reports latency + accuracy against both baselines

The model-dependent steps (select, extract, stitch, stitch_fast) share a single
load of the small and large models.
"""

import argparse

from q1.config import PAIRS, DEFAULT_PAIR, STITCH_MAX_NEW_TOKENS

STEP_ORDER = ["select", "extract", "train", "analyze", "cka", "fit_adapter",
              "stitch", "stitch_fast"]
DEFAULT_STEPS = ["select", "extract", "train", "analyze", "cka"]
NEEDS_MODELS = {"select", "extract", "stitch", "stitch_fast"}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("steps", nargs="*", metavar="step",
                    help=f"steps to run, from {STEP_ORDER} (default: {DEFAULT_STEPS})")
    ap.add_argument("--pair", choices=sorted(PAIRS), default=DEFAULT_PAIR,
                    help=f"model pair to run (default: {DEFAULT_PAIR})")
    ap.add_argument("--i", type=int, default=None,
                    help="small layer to stitch from (fit_adapter/stitch; "
                         "default: divergence_layer_small)")
    ap.add_argument("--j", type=int, default=None,
                    help="large layer to stitch into (fit_adapter/stitch; "
                         "default: its depth-matched best match)")
    ap.add_argument("--max-new-tokens", type=int, default=STITCH_MAX_NEW_TOKENS,
                    help=f"generation budget for stitch_fast (default: {STITCH_MAX_NEW_TOKENS})")
    ap.add_argument("--skip-checks", action="store_true",
                    help="stitch_fast: benchmark without re-verifying against the v1 path")
    args = ap.parse_args()
    unknown = [s for s in args.steps if s not in STEP_ORDER]
    if unknown:
        ap.error(f"unknown step(s) {unknown}; choose from {STEP_ORDER}")
    steps = [s for s in STEP_ORDER if s in (args.steps or DEFAULT_STEPS)]
    pair = PAIRS[args.pair]

    lm_small = lm_large = None
    if NEEDS_MODELS & set(steps):
        from q1.model_utils import load_pair
        lm_small, lm_large = load_pair(pair)

    for step in steps:
        print("\n" + "#" * 70 + f"\n# step: {step}  (pair: {pair.name})\n" + "#" * 70)
        if step == "select":
            from q1 import select_prompts
            select_prompts.run(pair, lm_small, lm_large)
        elif step == "extract":
            from q1 import extract_states
            extract_states.run(pair, lm_small, lm_large)
        elif step == "train":
            from q1 import train_dm
            train_dm.run(pair)
        elif step == "analyze":
            from q1 import analyze
            analyze.run(pair)
        elif step == "cka":
            from q1 import cka
            cka.run(pair)
        elif step == "fit_adapter":
            from q1 import fit_adapter
            fit_adapter.run(pair, args.i, args.j)
        elif step == "stitch":
            from q1 import stitch
            stitch.run(pair, lm_small, lm_large, args.i, args.j)
        elif step == "stitch_fast":
            from q1 import stitch_fast
            stitch_fast.run(pair, lm_small, lm_large, args.i, args.j,
                            max_new_tokens=args.max_new_tokens,
                            skip_checks=args.skip_checks)


if __name__ == "__main__":
    main()
