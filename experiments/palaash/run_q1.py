"""Run the Question-1 pipeline for a small-vs-large model pair.

    python run_q1.py                             # all four steps, llama pair
    python run_q1.py --pair qwen                 # all four steps, qwen pair
    python run_q1.py --pair qwen select extract  # just the model-dependent steps
    python run_q1.py train analyze               # re-fit + re-plot from saved states

Pairs (see q1/config.py):
    llama       Llama-3.2-1B-Instruct  vs Llama-3.2-3B-Instruct   (token-aligned)
    qwen        Qwen2.5-0.5B-Instruct  vs Qwen2.5-3B-Instruct     (token-aligned)
    llama2qwen  Llama-3.2-1B-Instruct  vs Qwen2.5-3B-Instruct     (prompt-aligned)
    qwen2llama  Qwen2.5-0.5B-Instruct  vs Llama-3.2-3B-Instruct   (prompt-aligned)

Steps (outputs under results/<pair>/):
    select   -> generations.csv, selection.json
    extract  -> states/*.npz
    train    -> dm/*.npz, dm/dm_summary.json
    analyze  -> figures/*.png, verdict.txt (+ printed verdict)
    cka      -> cka/*.npz, cka/cka_summary.json, figures/cka_*.png, verdict_cka.txt

The two model-dependent steps (select, extract) share a single load of the
small and large models.
"""

import argparse

from q1.config import PAIRS, DEFAULT_PAIR

STEP_ORDER = ["select", "extract", "train", "analyze", "cka"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("steps", nargs="*", metavar="step",
                    help=f"steps to run, from {STEP_ORDER} (default: all, in order)")
    ap.add_argument("--pair", choices=sorted(PAIRS), default=DEFAULT_PAIR,
                    help=f"model pair to run (default: {DEFAULT_PAIR})")
    args = ap.parse_args()
    unknown = [s for s in args.steps if s not in STEP_ORDER]
    if unknown:
        ap.error(f"unknown step(s) {unknown}; choose from {STEP_ORDER}")
    steps = [s for s in STEP_ORDER if s in (args.steps or STEP_ORDER)]
    pair = PAIRS[args.pair]

    lm_small = lm_large = None
    if "select" in steps or "extract" in steps:
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


if __name__ == "__main__":
    main()
