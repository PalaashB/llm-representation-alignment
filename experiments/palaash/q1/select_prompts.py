"""Step 1 — find prompts where the small model hallucinates but the large model
answers correctly (the cases the experiment studies), plus a "both-right"
control set.

For every prompt in the bank we greedily generate a short answer from each model
and score it. Prompts are then bucketed:

    divergent  : small WRONG and large RIGHT   <- hallucination cases (the focus)
    both_right : small RIGHT and large RIGHT   <- easy control
    (other buckets are recorded but not used downstream)

Outputs (in results/<pair>/):
    generations.csv   — every prompt with both answers + correctness
    selection.json    — the divergent / both_right id lists + bucket counts
"""

import csv
import json

from q1.config import ModelPair, PAIRS, DEFAULT_PAIR, results_dir
from q1.model_utils import LM, load_pair, generate_answer
from q1.prompts import PROMPTS
from q1.scoring import is_correct


def run(pair: ModelPair, lm_small: LM | None = None, lm_large: LM | None = None) -> dict:
    if lm_small is None or lm_large is None:
        lm_small, lm_large = load_pair(pair)
    out_dir = results_dir(pair)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, p in enumerate(PROMPTS):
        g_s = generate_answer(lm_small, p["question"])
        g_l = generate_answer(lm_large, p["question"])
        c_s = is_correct(g_s, p["answers"])
        c_l = is_correct(g_l, p["answers"])
        rows.append({
            "id": p["id"], "category": p["category"], "question": p["question"],
            "gold": " | ".join(p["answers"]),
            "gen_small": g_s, "correct_small": int(c_s),
            "gen_large": g_l, "correct_large": int(c_l),
        })
        flag = "DIVERGENT" if (not c_s and c_l) else ("both_ok" if (c_s and c_l) else "")
        print(f"[{i+1:>2}/{len(PROMPTS)}] {p['id']:14s} "
              f"{lm_small.tag}={int(c_s)} {lm_large.tag}={int(c_l)} {flag}")
        print(f"      {lm_small.tag}: {g_s[:70]!r}")
        print(f"      {lm_large.tag}: {g_l[:70]!r}")

    # ── bucket ────────────────────────────────────────────────────────────────
    divergent = [r["id"] for r in rows if not r["correct_small"] and r["correct_large"]]
    both_right = [r["id"] for r in rows if r["correct_small"] and r["correct_large"]]
    both_wrong = [r["id"] for r in rows if not r["correct_small"] and not r["correct_large"]]
    only_small = [r["id"] for r in rows if r["correct_small"] and not r["correct_large"]]

    # ── write csv ─────────────────────────────────────────────────────────────
    csv_path = out_dir / "generations.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    selection = {
        "pair": pair.name,
        "divergent_small_wrong_large_right": divergent,
        "control_both_right": both_right,
        "both_wrong": both_wrong,
        "only_small_right": only_small,
        "counts": {
            "total": len(rows),
            "divergent": len(divergent),
            "both_right": len(both_right),
            "both_wrong": len(both_wrong),
            "only_small_right": len(only_small),
            "acc_small": sum(r["correct_small"] for r in rows) / len(rows),
            "acc_large": sum(r["correct_large"] for r in rows) / len(rows),
        },
    }
    sel_path = out_dir / "selection.json"
    with open(sel_path, "w") as f:
        json.dump(selection, f, indent=2)

    print("\n" + "=" * 60)
    print(f"SELECTION SUMMARY ({pair.name}: {lm_small.tag} vs {lm_large.tag})")
    print("=" * 60)
    for k, v in selection["counts"].items():
        print(f"  {k:16s}: {v:.3f}" if isinstance(v, float) else f"  {k:16s}: {v}")
    print(f"\n  divergent ids : {divergent}")
    print(f"  control  ids  : {both_right}")
    print(f"\nWrote {csv_path}")
    print(f"Wrote {sel_path}")
    return selection


if __name__ == "__main__":
    run(PAIRS[DEFAULT_PAIR])
