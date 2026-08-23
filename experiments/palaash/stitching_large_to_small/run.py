"""CLI for the accuracy-oriented stitching experiment (large -> small).

    python -m stitching_large_to_small.run headroom --pair llama   # is the bank usable?
    python -m stitching_large_to_small.run capture  --pair llama
    python -m stitching_large_to_small.run fit      --pair llama --i 10 --j 18
    python -m stitching_large_to_small.run check    --pair llama --i 10 --j 18
    python -m stitching_large_to_small.run bench    --pair llama --i 10 --j 18
    python -m stitching_large_to_small.run sweep    --pair llama
    python -m stitching_large_to_small.run final    --pair llama --i 10 --j 18

Everything on disk is scoped by (pair, bank), so runs never overwrite each other.

`check` gates the rest: `bench` and `sweep` refuse to report accuracy unless the
plumbing checks pass, because a mis-plumbed injection still emits fluent text and
would otherwise be written up as a result.
"""

from __future__ import annotations

import argparse
import csv
import json

from stitching_large_to_small import adapter as adapter_mod
from stitching_large_to_small import data
from stitching_large_to_small.config import (
    ANSWER_WEIGHT, BANKS, DEFAULT_BANK, DEFAULT_PAIR, MAX_CI_WIDTH_PTS,
    MIN_DIVERGENT_PROMPTS, MIN_HEADROOM_PTS, NORM_MATCH, PAIRS, RIDGE_ALPHA,
    adapter_path, checks_path, results_dir, sweep_path, table_path, validate_layers,
)
from stitching_large_to_small.evaluate import Harness, format_table, verdict
from stitching_large_to_small.stitch import run_checks
from common.stats import ci_width_pts, fmt_pct_ci, separated, wilson


def chosen(a):
    return PAIRS[a.pair], BANKS[a.bank]


def cmd_capture(a):
    pair, bank = chosen(a)
    data.run(pair, bank, split=a.split, max_prompts=a.max_prompts)


def cmd_fit(a):
    pair, bank = chosen(a)
    adapter_mod.fit(pair, a.i, a.j, bank, alpha=a.alpha,
                    answer_weight=a.answer_weight, norm_match=not a.no_norm_match)


def _run_and_store_checks(h, pair, bank, i, j) -> list[dict]:
    checks = run_checks(h.small, h.large, i, j)
    checks.append(adapter_mod.reload_quality(pair, i, j, bank))
    out = checks_path(pair, bank, i, j)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(checks, f, indent=2)
    return checks


def _print_checks(checks: list[dict]) -> bool:
    for c in checks:
        print(f"  [{'PASS' if c['passed'] else 'FAIL'}] {c['check']}")
        for k, v in c.items():
            if k not in ("check", "passed"):
                print(f"          {k}: {v}")
    ok = all(c["passed"] for c in checks)
    print(f"\n{sum(c['passed'] for c in checks)}/{len(checks)} passed")
    return ok


def cmd_check(a):
    pair, bank = chosen(a)
    validate_layers(pair, a.i, a.j)
    h = Harness(pair, bank)
    if not _print_checks(_run_and_store_checks(h, pair, bank, a.i, a.j)):
        raise SystemExit(1)


def cmd_headroom(a):
    """Small and large alone on a bank: the accuracy a stitch could recover.

    Run this before anything else on a new bank. A large->small stitch buys back
    prompts the small model got wrong and the large model got right; a bank with
    few such prompts cannot show the effect however good the adapter is, and the
    result would be a null finding about the bank rather than about the method.
    """
    pair, bank = chosen(a)
    h = Harness(pair, bank)
    prompts = data.split_prompts(bank, a.split)
    small = h.baseline("small", a.split, prompts)
    large = h.baseline("large", a.split, prompts)

    S = {r["prompt_id"]: r for r in small["accuracy"]["rows"]}
    L = {r["prompt_id"]: r for r in large["accuracy"]["rows"]}
    div = sorted(p for p in S if not S[p]["correct"] and L[p]["correct"])
    both_wrong = sorted(p for p in S if not S[p]["correct"] and not L[p]["correct"])
    a_s, a_l = small["accuracy"]["accuracy"], large["accuracy"]["accuracy"]
    gap = (a_l - a_s) * 100

    n = len(prompts)
    print(f"\nheadroom {pair.name} bank={bank.name} split={a.split} n={n}")
    print(f"{'path':<10}{'acc (95% CI)':>22}{'ms/tok':>10}{'out tok':>9}")
    for d in (small, large):
        acc, lat = d["accuracy"], d["latency"]
        ci = acc.get("accuracy_ci95") or list(wilson(acc["n_correct"], acc["n"]))
        print(f"{d['label']:<10}{fmt_pct_ci(acc['accuracy'], ci):>22}"
              f"{lat['decode_ms_per_token']:>10.1f}{acc['mean_generated_tokens']:>9.1f}")
    sep = separated(large["accuracy"]["n_correct"], n, small["accuracy"]["n_correct"], n)
    ci_w = ci_width_pts(*wilson(small["accuracy"]["n_correct"], n))
    print(f"\naccuracy headroom      : {gap:+.1f} pts "
          f"(intervals separated: {'YES' if sep else 'NO'})")
    print(f"divergent prompts      : {len(div)} (small wrong, large right)")
    print(f"small-model CI width   : {ci_w:.1f} pts at n={n} — the smallest "
          f"improvement this split could demonstrate")
    # A bank must offer a real gap AND be big enough to resolve one. The second
    # condition is new: `factual` passes the first on paper (8.6 pts) while its
    # 35-prompt split cannot resolve anything smaller than ~23 points.
    ok = (gap >= MIN_HEADROOM_PTS and len(div) >= MIN_DIVERGENT_PROMPTS
          and sep and ci_w <= MAX_CI_WIDTH_PTS)
    print(f"usable for this study  : {'YES' if ok else 'NO'} "
          f"(need >= {MIN_HEADROOM_PTS:.0f} pts, >= {MIN_DIVERGENT_PROMPTS} divergent, "
          f"separated intervals, CI width <= {MAX_CI_WIDTH_PTS:.0f} pts)")

    # Both-wrong prompts are the audit queue: when two models of different sizes
    # agree on an answer the bank calls wrong, the gold is the likely error.
    if both_wrong:
        print(f"\n{len(both_wrong)} prompts BOTH models got wrong — audit these gold "
              f"answers before trusting the bank:")
        for pid in both_wrong[:25]:
            print(f"    {pid:24s} gold={S[pid]['gold']}  "
                  f"small={S[pid]['generation'][:28]!r}  large={L[pid]['generation'][:28]!r}")
        if len(both_wrong) > 25:
            print(f"    ... and {len(both_wrong) - 25} more (see the JSON)")

    out = results_dir(pair, bank) / f"headroom_{a.split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"pair": pair.name, "bank": bank.name, "split": a.split,
                   "n_prompts": len(prompts), "accuracy_small": a_s,
                   "accuracy_large": a_l, "headroom_pts": gap,
                   "divergent_ids": div, "n_divergent": len(div),
                   "both_wrong_ids": both_wrong, "usable": ok,
                   "mean_generated_tokens_small":
                       small["accuracy"]["mean_generated_tokens"],
                   "mean_generated_tokens_large":
                       large["accuracy"]["mean_generated_tokens"],
                   "paths": {"small": small, "large": large}}, f, indent=2)
    print(f"\nWrote {out}")


def cmd_bench(a):
    pair, bank = chosen(a)
    validate_layers(pair, a.i, a.j)
    h = Harness(pair, bank)
    if not a.skip_checks:
        print("[checks] plumbing must pass before accuracy is reported")
        if not _print_checks(_run_and_store_checks(h, pair, bank, a.i, a.j)):
            raise SystemExit("checks FAILED — refusing to report accuracy. "
                             "Fix the injection, or pass --skip-checks to see the "
                             "numbers anyway (they are not trustworthy).")
        print()
    for mode in a.modes:
        h.bench(a.i, a.j, split=a.split, mode=mode)


def cmd_final(a):
    a.split = "test"
    cmd_bench(a)


def cmd_sweep(a):
    pair, bank = chosen(a)
    grid_i = a.grid_i or list(pair.grid_i)
    grid_j = a.grid_j or list(pair.grid_j)
    combos = [(i, j) for i in grid_i for j in grid_j]
    print(f"[sweep] {pair.name}/{bank.name}: {len(combos)} points  "
          f"i={grid_i}  j={grid_j}  modes={a.modes}\n")

    for i, j in combos:
        if a.refit or not adapter_path(pair, i, j, bank).exists():
            adapter_mod.fit(pair, i, j, bank, alpha=a.alpha,
                            answer_weight=a.answer_weight,
                            norm_match=not a.no_norm_match)

    h = Harness(pair, bank)
    if not a.skip_checks:
        i0, j0 = combos[0]
        print("[checks] plumbing must pass before accuracy is reported")
        if not _print_checks(_run_and_store_checks(h, pair, bank, i0, j0)):
            raise SystemExit("checks FAILED — refusing to sweep.")
        print()

    rows = []
    todo = [(i, j, m) for i, j in combos for m in a.modes]
    for n, (i, j, mode) in enumerate(todo, 1):
        print(f"[{n}/{len(todo)}] large L{j} -> small L{i}  mode={mode}")
        rep = h.bench(i, j, split=a.split, mode=mode, quiet=True)
        status, _ = verdict(rep)
        rows.append(sweep_row(rep, status))
        print(f"      acc={rows[-1]['accuracy_stitch']:.1%} "
              f"({rows[-1]['acc_gain_vs_small_pts']:+.1f} vs small)  "
              f"divergent={rows[-1]['divergent_acc_stitch']:.1%}  [{status}]")

    for mode in a.modes:
        sub = [r for r in rows if r["mode"] == mode]
        if sub:
            _report(pair, bank, a.split, mode, sub)


def sweep_row(rep: dict, status: str) -> dict:
    """One sweep row from a bench report. Cost columns are end-to-end as well as
    per-token, because this path pays a large-model prefix prefill that a
    ms/token ratio does not show."""
    return {
        "i": rep["small_layer_i"], "j": rep["large_layer_j"], "mode": rep["mode"],
        "status": status,
        "n_prompts": rep["n_prompts"],
        "accuracy_small": rep["accuracy_small"],
        "accuracy_large": rep["accuracy_large"],
        "accuracy_stitch": rep["accuracy_stitch"],
        "acc_ci_lo_stitch": rep["accuracy_ci95_stitch"][0],
        "acc_ci_hi_stitch": rep["accuracy_ci95_stitch"][1],
        "acc_ci_lo_small": rep["accuracy_ci95_small"][0],
        "acc_ci_hi_small": rep["accuracy_ci95_small"][1],
        "beats_small_ci": rep["beats_small_ci"],
        "acc_gain_vs_small_pts": rep["acc_gain_vs_small_pts"],
        "acc_gap_vs_large_pts": rep["acc_gap_vs_large_pts"],
        "divergent_n": rep["divergent"]["n"],
        "divergent_acc_stitch": rep["divergent"]["accuracy_stitch"],
        "decode_slowdown_vs_small": rep["decode_slowdown_vs_small"],
        "end_to_end_slowdown_vs_small": rep["end_to_end_slowdown_vs_small"],
        "end_to_end_speedup_vs_large": rep["end_to_end_speedup_vs_large"],
        "answer_tokens_priced": rep["answer_tokens_priced"],
        "answer_tokens_priced_source": rep["answer_tokens_priced_source"],
        "stitch_prefill_ms": rep["latency"]["stitch"]["prefill_ms"],
        "small_prefill_ms": rep["latency"]["small"]["prefill_ms"],
        "large_prefill_ms": rep["latency"]["large"]["prefill_ms"],
        "held_out_r2_answer": rep["adapter"]["held_out_r2_answer"],
        "held_out_r2_all": rep["adapter"]["held_out_r2_all"],
        "answer_weight_frac": rep["adapter"].get("answer_weight_frac"),
        "adapter_train_method": rep.get("adapter_train_method", "ridge"),
    }


def select(rows: list[dict]) -> dict:
    """The small model competes here too, and the bar is interval-separated.

    Same rule as the sibling package, adapted to this direction's goal. There is
    no speed axis to trade against — this path is unambiguously slower than the
    small model — so a configuration is only worth recommending if it beats the
    small model's accuracy outright, at 95%. Everything else is either a
    failure or an unresolved measurement, and both get said plainly.
    """
    small_acc = rows[0]["accuracy_small"]
    for r in rows:
        r["beats_small_point"] = bool(r["accuracy_stitch"] > small_acc)
        # No speed dimension to trade on: slower AND not better is dominated.
        r["dominated_by_small"] = not r["beats_small_ci"]
    viable = [r for r in rows if r["beats_small_ci"]]
    best = (max(viable, key=lambda r: r["accuracy_stitch"]) if viable else None)
    top = max(rows, key=lambda r: r["accuracy_stitch"])
    n_point = sum(r["beats_small_point"] for r in rows)
    if best is None:
        if n_point:
            verdict_txt = (
                f"NO VIABLE POINT. {n_point}/{len(rows)} cells beat the small model's "
                f"{small_acc:.1%} on the point estimate, but none does so with "
                f"non-overlapping 95% intervals at n={rows[0]['n_prompts']}, so none "
                f"is a demonstrated win. Best was {top['accuracy_stitch']:.1%} "
                f"[{top['acc_ci_lo_stitch']:.1%}-{top['acc_ci_hi_stitch']:.1%}] at "
                f"large L{top['j']} -> small L{top['i']} ({top['mode']}), costing "
                f"{top['end_to_end_slowdown_vs_small']:.2f}x the small model's "
                f"end-to-end time.")
        else:
            verdict_txt = (
                f"NO VIABLE POINT. No configuration beat the small model alone "
                f"({small_acc:.1%}) even on the point estimate. Best was "
                f"{top['accuracy_stitch']:.1%} "
                f"[{top['acc_ci_lo_stitch']:.1%}-{top['acc_ci_hi_stitch']:.1%}] at "
                f"large L{top['j']} -> small L{top['i']} ({top['mode']}), costing "
                f"{top['end_to_end_slowdown_vs_small']:.2f}x the small model's "
                f"end-to-end time. On this bank and grid, large->small stitching "
                f"does not buy accuracy.")
    else:
        verdict_txt = (
            f"large L{best['j']} -> small L{best['i']} ({best['mode']}): "
            f"{best['accuracy_stitch']:.1%} "
            f"[{best['acc_ci_lo_stitch']:.1%}-{best['acc_ci_hi_stitch']:.1%}], "
            f"{best['acc_gain_vs_small_pts']:+.1f} pts over the small model with "
            f"non-overlapping intervals, at "
            f"{best['end_to_end_slowdown_vs_small']:.2f}x its end-to-end cost.")
    return {"best": best, "verdict": verdict_txt, "small_accuracy": small_acc,
            "n_beating_small_point": n_point, "n_viable": len(viable),
            "n_dominated_by_small": sum(r["dominated_by_small"] for r in rows)}


def _report(pair, bank, split, mode, rows):
    jp, tp = sweep_path(pair, bank, split, mode), table_path(pair, bank, split, mode)
    for p in (jp, tp):
        p.parent.mkdir(parents=True, exist_ok=True)
    pick = select(rows)
    with open(jp, "w") as f:
        json.dump({"pair": pair.name, "bank": bank.name, "split": split,
                   "mode": mode, "rows": rows, "selection": pick}, f, indent=2)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(tp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    r0 = rows[0]
    print("\n" + "=" * 96)
    print(f"sweep {pair.name}/{bank.name} split={split} mode={mode} "
          f"n={r0['n_prompts']}   small={r0['accuracy_small']:.1%}  "
          f"large={r0['accuracy_large']:.1%}   answers priced at "
          f"{r0['answer_tokens_priced']:.2f} tok "
          f"(source: {r0['answer_tokens_priced_source']})")
    print(f"{'j(lg)':>6}{'i(sm)':>6}{'acc (95% CI)':>22}{'vs sm':>7}"
          f"{'div':>7}{'e2e/sm':>8}{'pre(st/sm)':>12}{'R2a':>7}  status")
    for r in sorted(rows, key=lambda r: -r["accuracy_stitch"]):
        mark = " *" if r is pick["best"] else ""
        print(f"{r['j']:>6}{r['i']:>6}"
              f"{fmt_pct_ci(r['accuracy_stitch'], (r['acc_ci_lo_stitch'], r['acc_ci_hi_stitch'])):>22}"
              f"{r['acc_gain_vs_small_pts']:>+7.1f}"
              f"{r['divergent_acc_stitch'] * 100:>6.0f}%"
              f"{r['end_to_end_slowdown_vs_small']:>7.2f}x"
              f"{r['stitch_prefill_ms']:>8.0f}/{r['small_prefill_ms']:>3.0f}"
              f"{(r['held_out_r2_answer'] if r['held_out_r2_answer'] is not None else float('nan')):>7.2f}"
              f"  {r['status']}{mark}")
    print(f"\n{pick['verdict']}")
    print(f"  ({pick['n_beating_small_point']}/{len(rows)} beat the small model on the "
          f"point estimate, {pick['n_viable']} with non-overlapping 95% intervals)")
    print(f"Wrote {jp}\n      {tp}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--pair", choices=sorted(PAIRS), default=DEFAULT_PAIR)
        p.add_argument("--bank", choices=sorted(BANKS), default=DEFAULT_BANK)

    def layers(p):
        p.add_argument("--i", type=int, required=True, help="small block to resume at")
        p.add_argument("--j", type=int, required=True, help="large block to exit before")

    def fit_opts(p):
        p.add_argument("--alpha", type=float, default=RIDGE_ALPHA)
        p.add_argument("--answer-weight", type=float, default=ANSWER_WEIGHT)
        p.add_argument("--no-norm-match", action="store_true",
                       help=f"disable radial rescaling (default norm_match={NORM_MATCH})")

    def modes(p):
        p.add_argument("--modes", nargs="+", choices=("exit", "warm"), default=["exit"],
                       help="exit: the prompt is stitched too (the large model's reading "
                            "of the prompt transfers). warm: the small model prefills the "
                            "prompt itself (default: %(default)s)")
        p.add_argument("--skip-checks", action="store_true",
                       help="report accuracy without passing the plumbing checks first "
                            "— the numbers are not trustworthy")

    p = sub.add_parser("headroom", help="small vs large alone; is this bank usable?")
    common(p)
    p.add_argument("--split", default="dev", choices=("fit", "dev", "test", "all"))
    p.set_defaults(func=cmd_headroom)

    p = sub.add_parser("capture", help="paired states over prompt + large-model answer")
    common(p)
    p.add_argument("--split", default="fit", choices=("fit", "dev", "test", "all"))
    p.add_argument("--max-prompts", type=int, default=None)
    p.set_defaults(func=cmd_capture)

    p = sub.add_parser("fit", help="fit one large->small adapter")
    common(p); layers(p); fit_opts(p)
    p.set_defaults(func=cmd_fit)

    p = sub.add_parser("check", help="plumbing checks for one (i, j)")
    common(p); layers(p)
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("bench", help="small vs large vs stitched")
    common(p); layers(p); modes(p)
    p.add_argument("--split", default="dev", choices=("fit", "dev", "test", "all"))
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser("final", help="bench on the untouched test split")
    common(p); layers(p); modes(p)
    p.set_defaults(func=cmd_final)

    p = sub.add_parser("sweep", help="grid over (i, j)")
    common(p); fit_opts(p); modes(p)
    p.add_argument("--split", default="dev", choices=("fit", "dev", "test", "all"))
    p.add_argument("--grid-i", type=int, nargs="+", default=None)
    p.add_argument("--grid-j", type=int, nargs="+", default=None)
    p.add_argument("--refit", action="store_true")
    p.set_defaults(func=cmd_sweep)

    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
