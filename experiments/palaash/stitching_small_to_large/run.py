"""CLI for the latency-oriented stitching experiment.

    python -m stitching_small_to_large.run headroom --pair llama                      # is this bank worth a sweep?
    python -m stitching_small_to_large.run capture --pair llama                       # paired states (prompt + answer)
    python -m stitching_small_to_large.run fit     --pair llama --i 8 --j 12
    python -m stitching_small_to_large.run bench   --pair llama --i 8 --j 12 --modes exit warm
    python -m stitching_small_to_large.run sweep   --pair llama --modes exit warm     # the (i, j) grid
    python -m stitching_small_to_large.run report  --pair llama --split dev           # rebuild the table, no GPU work
    python -m stitching_small_to_large.run final   --pair llama --i 8 --j 12 --modes warm   # held-out test split
    python -m stitching_small_to_large.run check   --pair llama --j 12                # verify the fast path
    python -m stitching_small_to_large.run compare --pair llama --split dev           # variants side by side

Four axes are chosen per invocation and everything on disk is scoped by them, so
runs never overwrite each other and a table never mixes them:

    --bank          which prompts and decode budgets. `list_hard` is the default
                    and the only bank whose eval splits (173) can resolve the
                    differences being measured; `list` and `factual` are kept.
    --adapter       the map's SHAPE: `linear` or `mlp` (+ nonlinear correction)
    --train-method  how it is FIT: `ridge` (least squares on hidden states) or
                    `distill` (KL to the teacher's next-token distribution,
                    through the frozen suffix, warm-started from ridge)
    --n-taps        how many small-model layers feed the adapter

A capture is per (pair, bank); adapters and bench reports are per variant on top
of that. Changing --adapter, --train-method or --n-taps needs a refit but not a
re-capture (--train-method distill needs a capture that stored teacher logits,
which is the default).

See README.md for what the numbers mean.
"""

from __future__ import annotations

import argparse
import csv
import json

from stitching_small_to_large import adapter as adapter_mod
from stitching_small_to_large import data
from stitching_small_to_large.config import (
    ADAPTER_KIND, ADAPTER_KINDS, ANSWER_WEIGHT, BANKS, DEFAULT_BANK, DEFAULT_PAIR,
    MAX_CI_WIDTH_PTS, MIN_DECODE_SHARE, MIN_DECODE_SPEEDUP, MIN_E2E_SPEEDUP,
    MIN_HEADROOM_PTS, N_TAPS, NORM_MATCH, PAIRS, RIDGE_ALPHA, TRAIN_METHOD,
    TRAIN_METHODS, adapter_path, results_dir, sweep_stem, validate_layers,
)
from stitching_small_to_large.evaluate import (
    Harness, flag_prefill_anomalies, format_table, load_bench_rows, pareto, sweep_row,
)
from stitching_small_to_large.stitch import run_checks
from common.fit_corpus import FIT_CORPORA
from common.stats import ci_width_pts, fmt_pct_ci, separated, wilson


def chosen(a) -> tuple:
    """(pair, bank, kind, n_taps, train_method) — everything that scopes an
    artefact, so no two variants can land on the same filename."""
    return (PAIRS[a.pair], BANKS[a.bank], getattr(a, "adapter", ADAPTER_KIND),
            getattr(a, "n_taps", N_TAPS),
            getattr(a, "train_method", TRAIN_METHOD))


def cmd_capture(a):
    pair, bank, _, _, _ = chosen(a)
    data.run(pair, bank, split=a.split, max_prompts=a.max_prompts,
             fit_corpus=getattr(a, "fit_corpus", None),
             max_prompts_corpus=getattr(a, "corpus_prompts", None),
             vary_templates=not getattr(a, "no_template_variety", False),
             store_teacher_logits=getattr(a, "teacher_logits", True),
             small_layers=getattr(a, "small_layers", None),
             large_layers=getattr(a, "large_layers", None))


def cmd_fit(a):
    pair, bank, kind, n_taps, train_method = chosen(a)
    adapter_mod.fit(pair, a.i, a.j, bank, kind=kind, n_taps=n_taps,
                    train_method=train_method, alpha=a.alpha,
                    answer_weight=a.answer_weight, norm_match=not a.no_norm_match,
                    device=a.fit_device)


def cmd_bench(a):
    pair, bank, kind, n_taps, train_method = chosen(a)
    validate_layers(pair, a.i, a.j)
    h = Harness(pair, bank, kind, n_taps, train_method=train_method)
    for mode in a.modes:
        h.bench(a.i, a.j, split=a.split, mode=mode)


def cmd_final(a):
    """Same as bench, but on the split nothing has selected on."""
    a.split = "test"
    cmd_bench(a)


def cmd_check(a):
    pair, bank, kind, n_taps, train_method = chosen(a)
    h = Harness(pair, bank, kind, n_taps, train_method=train_method)
    checks = run_checks(h.small, h.large, a.j)
    for c in checks:
        print(f"  [{'PASS' if c['passed'] else 'FAIL'}] {c['check']}")
        for k, v in c.items():
            if k not in ("check", "passed"):
                print(f"          {k}: {v}")
    out = results_dir(pair, bank) / "checks.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(checks, f, indent=2)
    print(f"\n{sum(c['passed'] for c in checks)}/{len(checks)} passed -> {out}")
    if not all(c["passed"] for c in checks):
        raise SystemExit(1)


def cmd_headroom(a):
    """Small and large alone on a bank: the accuracy gap a stitch could recover,
    and the speed gap it has to beat. No adapter needed.

    Worth running before any sweep on a new bank. A stitch spends large-model
    layers to buy accuracy back from the small model, so a bank where the two
    models score the same offers nothing to buy and every cell will land
    dominated by the small model no matter how good the map is. This is the
    cheapest way to find that out — two baseline passes instead of a grid.
    """
    pair, bank, kind, n_taps, train_method = chosen(a)
    h = Harness(pair, bank, kind, n_taps, train_method=train_method)
    prompts = data.split_prompts(bank, a.split)
    small = h.baseline("small", a.split, prompts)
    large = h.baseline("large", a.split, prompts)

    a_s, a_l = small["accuracy"]["accuracy"], large["accuracy"]["accuracy"]
    ms = lambda d: d["latency"]["decode_ms_per_token"]
    n = len(prompts)
    # End-to-end, not just per-token: decode share is the thing that decides
    # whether a per-token win can survive into wall-clock at all.
    n_ref = large["accuracy"]["mean_generated_tokens"]
    e2e = {k: d["latency"]["prefill_ms"] + ms(d) * n_ref
           for k, d in (("small", small), ("large", large))}
    decode_share = (ms(large) * n_ref) / e2e["large"]
    print(f"\nheadroom {pair.name} bank={bank.name} split={a.split} n={n}")
    print(f"{'path':<10}{'acc (95% CI)':>22}{'ms/tok':>9}{'prefill':>10}"
          f"{'out tok':>9}{'ms/answer':>11}")
    for d in (small, large):
        acc, lat = d["accuracy"], d["latency"]
        ci = acc.get("accuracy_ci95") or list(wilson(acc["n_correct"], acc["n"]))
        print(f"{d['label']:<10}{fmt_pct_ci(acc['accuracy'], ci):>22}"
              f"{lat['decode_ms_per_token']:>9.1f}"
              f"{lat['prefill_ms']:>9.0f}ms{acc['mean_generated_tokens']:>9.1f}"
              f"{e2e['small' if d is small else 'large']:>10.0f}ms")
    sep = separated(large["accuracy"]["n_correct"], n, small["accuracy"]["n_correct"], n)
    print(f"\naccuracy headroom: {(a_l - a_s) * 100:+.1f} pts for a stitch to recover"
          f"  (intervals separated: {'YES' if sep else 'NO'})")
    print(f"speed to beat:     the small model decodes "
          f"{ms(large) / ms(small):.2f}x faster than large and runs "
          f"{e2e['large'] / e2e['small']:.2f}x faster end-to-end")
    print(f"decode share:      {decode_share:.0%} of the large model's end-to-end "
          f"time is decode (the only phase a stitch speeds up)")
    # The gate. A bank failing any of these cannot support a latency claim, and
    # finding that out costs two baseline passes instead of a whole grid.
    ci_w = ci_width_pts(*wilson(int(round(a_s * n)), n))
    ok = (sep and decode_share >= MIN_DECODE_SHARE
          and (a_l - a_s) * 100 >= MIN_HEADROOM_PTS and ci_w <= MAX_CI_WIDTH_PTS)
    print(f"usable for a latency claim: {'YES' if ok else 'NO'}  "
          f"(need separated intervals, headroom >= {MIN_HEADROOM_PTS:.0f} pts, "
          f"decode share >= {MIN_DECODE_SHARE:.0%}, small-model CI width <= "
          f"{MAX_CI_WIDTH_PTS:.0f} pts; got {ci_w:.1f} pts at n={n})")
    rows = {r["prompt_id"]: r for r in large["accuracy"]["rows"]}
    only_large = [r["prompt_id"] for r in small["accuracy"]["rows"]
                  if not r["correct"] and rows[r["prompt_id"]]["correct"]]
    print(f"large right / small wrong on {len(only_large)}/{len(prompts)} prompts"
          + (f": {', '.join(only_large[:8])}{' ...' if len(only_large) > 8 else ''}"
             if only_large else ""))
    out = results_dir(pair, bank) / f"headroom_{a.split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"pair": pair.name, "bank": bank.name, "split": a.split,
                   "n_prompts": len(prompts), "accuracy_small": a_s,
                   "accuracy_large": a_l, "headroom_pts": (a_l - a_s) * 100,
                   "decode_speedup_small_vs_large": ms(large) / ms(small),
                   "large_right_small_wrong": only_large,
                   "paths": {"small": small, "large": large}}, f, indent=2)
    print(f"Wrote {out}")


def cmd_sweep(a):
    """Fit + bench every (i, j) in the grid, then report the frontier.

    Selection happens on `dev`; `final` re-scores the winner on `test`, which
    nothing in the sweep has touched.
    """
    pair, bank, kind, n_taps, train_method = chosen(a)
    grid_i = a.grid_i or list(pair.grid_i)
    grid_j = a.grid_j or list(pair.grid_j)
    combos = [(i, j) for i in grid_i for j in grid_j]
    print(f"[sweep] {pair.name} bank={bank.name} adapter={kind} n_taps={n_taps} "
          f"train={train_method}: {len(combos)} points  i={grid_i}  j={grid_j}\n")

    for i, j in combos:
        if a.refit or not adapter_path(pair, i, j, bank, kind, n_taps,
                                       train_method).exists():
            adapter_mod.fit(pair, i, j, bank, kind=kind, n_taps=n_taps,
                            train_method=train_method, alpha=a.alpha,
                            answer_weight=a.answer_weight,
                            norm_match=not a.no_norm_match, device=a.fit_device)

    h = Harness(pair, bank, kind, n_taps, train_method=train_method)
    rows = []
    todo = [(i, j, m) for i, j in combos for m in a.modes]
    for n, (i, j, mode) in enumerate(todo, 1):
        print(f"\n[{n}/{len(todo)}] L{i} -> L{j}  mode={mode}")
        rep = h.bench(i, j, split=a.split, mode=mode, quiet=True)
        # Same builder `report` uses, so the two can never disagree.
        rows.append(sweep_row(rep, pair, bank))
        print(f"      acc={rows[-1]['accuracy_stitch']:.1%} "
              f"({rows[-1]['acc_gap_vs_large_pts']:+.1f} vs large, "
              f"{rows[-1]['acc_gain_vs_small_pts']:+.1f} vs small)  "
              f"speedup={rows[-1]['decode_speedup_vs_large']:.2f}x decode / "
              f"{rows[-1]['end_to_end_speedup_vs_large']:.2f}x e2e")

    report_sweep(pair, bank, kind, n_taps, a.split, rows, grid_i, grid_j,
                 train_method)


def cmd_report(a):
    """Rebuild the sweep table from bench reports already on disk.

    Re-analysis costs nothing, so a change to the latency statistic or the
    selection rule never has to be paid for with another sweep. This is also the
    backfill path: benches written before end-to-end pricing existed get their
    e2e and R2-answer columns computed from the paths and adapter sidecars they
    already stored, with no GPU work.
    """
    pair, bank, kind, n_taps, train_method = chosen(a)
    rows = load_bench_rows(pair, a.split, bank, kind, n_taps, train_method)
    report_sweep(pair, bank, kind, n_taps, a.split, rows,
                 sorted({r["i"] for r in rows}), sorted({r["j"] for r in rows}),
                 train_method)


def _variants_on_disk(pair, bank, split, max_taps):
    found = {}
    for kind in ADAPTER_KINDS:
        for n_taps in range(1, max_taps + 1):
            for tm in TRAIN_METHODS:
                try:
                    found[(kind, n_taps, tm)] = load_bench_rows(
                        pair, split, bank, kind, n_taps, tm)
                except SystemExit:
                    continue
    return found


def cmd_compare(a):
    """Adapter variants at *fixed geometry*, from reports on disk.

    The questions the variant flags exist to answer — does a wider map help,
    does a nonlinear one, does distillation beat ridge — are only legible when
    the variants are lined up at the same (i, j, mode). Summarising each variant
    by its own best cell confounds the map with the layer pair, and with a
    ~20-point confidence interval per cell that confound is larger than any
    effect being looked for.

    The head-to-head table is therefore the primary output, and the
    best-cell-per-variant table is secondary.
    """
    pair, bank, _, _, _ = chosen(a)
    found = _variants_on_disk(pair, bank, a.split, a.max_taps)
    if not found:
        raise SystemExit(f"no bench reports at all for {pair.name}/{bank.name} "
                         f"split={a.split!r}.")

    # ── head-to-head: every geometry measured under more than one variant ─────
    by_cell: dict[tuple, dict] = {}
    for variant, rows in found.items():
        for r in rows:
            by_cell.setdefault((r["i"], r["j"], r["mode"]), {})[variant] = r
    shared = {k: v for k, v in by_cell.items() if len(v) > 1}

    print(f"\ncompare {pair.name} bank={bank.name} split={a.split}")
    if shared:
        print(f"\nhead-to-head at fixed geometry ({len(shared)} cells measured "
              f"under >1 variant):")
        print(f"{'cell':<18}{'variant':<22}{'acc (95% CI)':>22}{'vs small':>10}"
              f"{'e2e':>8}{'R2 ans':>9}")
        for (i, j, mode), variants in sorted(shared.items()):
            base = None
            for (kind, n_taps, tm), r in sorted(variants.items()):
                name = f"{kind}/t{n_taps}/{tm}"
                delta = ""
                if base is None:
                    base = r["accuracy_stitch"]
                else:
                    delta = f"  ({(r['accuracy_stitch'] - base) * 100:+.1f} vs first)"
                print(f"{f'L{i}->L{j} {mode}':<18}{name:<22}"
                      f"{fmt_pct_ci(r['accuracy_stitch'], (r['acc_ci_lo_stitch'], r['acc_ci_hi_stitch'])):>22}"
                      f"{r['acc_gain_vs_small_pts']:>+10.1f}"
                      f"{r['end_to_end_speedup_vs_large']:>7.2f}x"
                      f"{(r['held_out_r2_answer'] if r['held_out_r2_answer'] is not None else float('nan')):>9.3f}"
                      f"{delta}")
    else:
        print("\n(no geometry has been measured under more than one variant yet — "
              "the head-to-head table needs the same --i/--j benched twice)")

    print(f"\nbest cell per variant (confounds map with geometry — read the "
          f"head-to-head table above instead):")
    print(f"{'adapter':>10}{'taps':>6}{'train':>9}{'cells':>7}{'best acc':>10}"
          f"{'vs small':>10}{'at':>16}{'decode':>9}{'e2e':>7}{'R2 ans':>9}")
    ref = None
    for (kind, n_taps, tm), rows in sorted(found.items()):
        top = max(rows, key=lambda r: (r["accuracy_stitch"],
                                       r["end_to_end_speedup_vs_large"]))
        ref = ref or rows[0]
        where = "L{}->L{} {}".format(top["i"], top["j"], top["mode"])
        print(f"{kind:>10}{n_taps:>6}{tm:>9}{len(rows):>7}"
              f"{top['accuracy_stitch'] * 100:>9.1f}%"
              f"{top['acc_gain_vs_small_pts']:>+10.1f}{where:>16}"
              f"{top['decode_speedup_vs_large']:>8.2f}x"
              f"{top['end_to_end_speedup_vs_large']:>6.2f}x"
              f"{(top['held_out_r2_answer'] if top['held_out_r2_answer'] is not None else float('nan')):>9.3f}")
    print(f"\nbaselines: small={ref['accuracy_small']:.1%} @ "
          f"{ref['end_to_end_speedup_small_vs_large']:.2f}x e2e, "
          f"large={ref['accuracy_large']:.1%} @ 1.00x")


def report_sweep(pair, bank, kind, n_taps, split, rows, grid_i, grid_j,
                 train_method=TRAIN_METHOD):
    rows = flag_prefill_anomalies(rows)
    pick = pareto(rows, MIN_E2E_SPEEDUP)
    out_dir = results_dir(pair, bank)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = sweep_stem(split, kind, n_taps, train_method)
    csv_path = out_dir / f"{stem}.csv"
    # Union of keys, so a row that gained a field mid-grid cannot truncate the
    # header and silently blank a column for every other row.
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    json_path = out_dir / f"{stem}.json"
    with open(json_path, "w") as f:
        json.dump({"pair": pair.name, "bank": bank.name, "adapter_kind": kind,
                   "n_taps": n_taps, "train_method": train_method, "split": split,
                   "grid_i": grid_i, "grid_j": grid_j, "rows": rows,
                   "selection": pick}, f, indent=2)

    r0 = rows[0]
    print("\n" + "=" * 108)
    print(f"sweep {pair.name} bank={bank.name} adapter={kind} taps={n_taps} "
          f"train={train_method} (split={split}, n={r0.get('n_prompts', '?')})")
    print(f"  small={r0['accuracy_small']:.1%} @ "
          f"{r0['end_to_end_speedup_small_vs_large']:.2f}x e2e   "
          f"large={r0['accuracy_large']:.1%} @ 1.00x   "
          f"answers priced at {r0['answer_tokens_priced']:.2f} tokens "
          f"(source: {r0.get('answer_tokens_priced_source', 'bench')})")
    print(f"{'i':>3}{'j':>4}{'mode':>6}{'acc':>21}{'vs sm':>7}"
          f"{'decode':>8}{'e2e':>7}{'vs small (paired)':>21}{'R2a':>7}  ")
    for r in sorted(rows, key=lambda r: -r["accuracy_stitch"]):
        mark = (" *" if r is pick["best"]
                else " +" if r["on_frontier"] else "  ")
        # A cell the small model beats on BOTH axes has no reason to be run.
        mark += " (dominated by small)" if r["dominated_by_small"] else ""
        if not r["faster_than_large_e2e"]:
            mark += " [slower e2e than large]"
        # Decode cost is prompt-independent, so a wide timing spread means the
        # machine interfered during this cell — its speedup is not trustworthy.
        if r.get("latency_suspect"):
            worst, who = max((r["decode_spread_stitch"], "stitch"),
                             (r["decode_spread_large"], "baseline"))
            mark += f" [latency suspect: {who} spread {worst:.1f}x]"
        if r.get("prefill_suspect"):
            mark += (f" [prefill suspect: {r['prefill_ratio_vs_model']:.1f}x model "
                     f"vs {r['prefill_median_ratio_in_mode']:.1f}x median]")
        print(f"{r['i']:>3}{r['j']:>4}{r['mode']:>6}"
              f"{fmt_pct_ci(r['accuracy_stitch'], (r['acc_ci_lo_stitch'], r['acc_ci_hi_stitch'])):>21}"
              f"{r['acc_gain_vs_small_pts']:>+7.1f}"
              f"{r['decode_speedup_vs_large']:>7.2f}x"
              f"{r['end_to_end_speedup_vs_large']:>6.2f}x"
              f"{r['boot_diff_vs_small'] * 100:>+9.1f} "
              f"[{r['boot_lo_vs_small'] * 100:>+5.1f},{r['boot_hi_vs_small'] * 100:>+5.1f}]"
              f"{(r['held_out_r2_answer'] if r['held_out_r2_answer'] is not None else float('nan')):>7.2f}"
              f"{mark}")

    print(f"\n{pick['verdict']}")
    # Reported separately from the recommendation: "fastest thing that is
    # genuinely faster than the large model" and "thing worth recommending" are
    # different questions, and conflating them is how a 0.93x point got written
    # up as a win.
    bf = pick["best_e2e_faster"]
    if bf is None:
        print(f"  Genuinely faster than the large model end-to-end: NONE of "
              f"{len(rows)} cells.")
    else:
        print(f"  Best cell that is genuinely faster end-to-end than the large model: "
              f"L{bf['i']}->L{bf['j']} ({bf['mode']}) "
              f"{fmt_pct_ci(bf['accuracy_stitch'], (bf['acc_ci_lo_stitch'], bf['acc_ci_hi_stitch']))} "
              f"at {bf['end_to_end_speedup_vs_large']:.2f}x e2e "
              f"({pick['n_faster_than_large_e2e']}/{len(rows)} cells clear 1.00x).")
    print(f"  (+ = Pareto frontier on END-TO-END speed incl. the small model; "
          f"{pick['n_meeting_speedup']}/{len(rows)} clear {MIN_E2E_SPEEDUP:.2f}x e2e, "
          f"{pick['n_viable']} of those beat the small model's accuracy at 95%, "
          f"{pick['n_dominated_by_small']} dominated by it, "
          f"{pick['n_prefill_suspect']} prefill-suspect)")
    print(f"Wrote {csv_path}\n      {json_path}")
    b = pick["best"]
    if b:
        print(f"\nNext: python -m stitching_small_to_large.run final --pair {pair.name} "
              f"--bank {bank.name} --adapter {kind} --n-taps {n_taps} "
              f"--train-method {train_method} "
              f"--i {b['i']} --j {b['j']} --modes {b['mode']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--pair", choices=sorted(PAIRS), default=DEFAULT_PAIR)
        p.add_argument("--bank", choices=sorted(BANKS), default=DEFAULT_BANK,
                       help="prompt set and decode budgets (default: %(default)s)")

    def variant(p):
        p.add_argument("--adapter", choices=ADAPTER_KINDS, default=ADAPTER_KIND,
                       help="the map's SHAPE. linear: affine. mlp: affine plus a "
                            "GELU-MLP correction (default: %(default)s)")
        p.add_argument("--n-taps", type=int, default=N_TAPS,
                       help="small-model layers feeding the adapter; >1 reads "
                            "evenly spaced depths below i at no inference cost "
                            "(default: %(default)s)")
        p.add_argument("--train-method", choices=TRAIN_METHODS, default=TRAIN_METHOD,
                       help="how the map is FIT, orthogonal to its shape. ridge: "
                            "closed-form least squares on hidden states. distill: "
                            "gradient descent on the teacher's next-token "
                            "distribution through the frozen suffix, warm-started "
                            "from ridge (default: %(default)s)")

    def mode_opt(p):
        p.add_argument("--modes", nargs="+", choices=("exit", "warm"), default=["exit"],
                       help="exit: the prompt is stitched too (saves prefill as well). "
                            "warm: the large model prefills the prompt itself, so only "
                            "decode is saved (default: %(default)s)")

    def fit_opts(p):
        p.add_argument("--alpha", type=float, default=RIDGE_ALPHA)
        p.add_argument("--answer-weight", type=float, default=ANSWER_WEIGHT,
                       help="row weight for answer-token positions (default %(default)s)")
        p.add_argument("--no-norm-match", action="store_true",
                       help=f"disable radial rescaling (default is norm_match={NORM_MATCH})")
        p.add_argument("--fit-device", default="cpu",
                       help="device for the mlp correction's training loop; the ridge "
                            "solve is numpy either way (default: %(default)s)")

    p = sub.add_parser("capture", help="capture paired states over prompt + large-model answer")
    common(p)
    p.add_argument("--split", default="fit", choices=("fit", "dev", "test", "all"))
    p.add_argument("--max-prompts", type=int, default=None)
    p.add_argument("--fit-corpus", default=None, choices=sorted(FIT_CORPORA),
                   help="extra generic continuations mixed into the fit set, so "
                        "answer positions are not all short bank answers. Never "
                        "drawn from an eval bank (default: none)")
    p.add_argument("--corpus-prompts", type=int, default=None,
                   help="how many corpus items to capture (default: the corpus's own)")
    p.add_argument("--no-template-variety", action="store_true",
                   help="use one fixed system prompt and question framing, the "
                        "old behaviour that made 95%% of rows identical boilerplate")
    p.add_argument("--no-teacher-logits", dest="teacher_logits", action="store_false",
                   help="skip storing top-K teacher logits (they are what the "
                        "distill training method fits against)")
    p.add_argument("--small-layers", type=int, nargs="+", default=None,
                   help="override which small-model layers to store. The default "
                        "covers the whole grid plus tap depths, which is the right "
                        "trade for a bank-sized capture but writes several GB once "
                        "--fit-corpus multiplies the row count")
    p.add_argument("--large-layers", type=int, nargs="+", default=None,
                   help="override which large-model layers to store (see --small-layers)")
    p.set_defaults(func=cmd_capture, teacher_logits=True)

    p = sub.add_parser("fit", help="fit one adapter")
    common(p)
    variant(p)
    fit_opts(p)
    p.add_argument("--i", type=int, required=True)
    p.add_argument("--j", type=int, required=True)
    p.set_defaults(func=cmd_fit)

    p = sub.add_parser("bench", help="small vs large vs stitched")
    common(p)
    variant(p)
    mode_opt(p)
    p.add_argument("--i", type=int, required=True)
    p.add_argument("--j", type=int, required=True)
    p.add_argument("--split", default="dev", choices=("fit", "dev", "test", "all"))
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser("final", help="bench on the untouched test split")
    common(p)
    variant(p)
    mode_opt(p)
    p.add_argument("--i", type=int, required=True)
    p.add_argument("--j", type=int, required=True)
    p.set_defaults(func=cmd_final)

    p = sub.add_parser("sweep", help="grid over (i, j), pick the Pareto point")
    common(p)
    variant(p)
    fit_opts(p)
    mode_opt(p)
    p.add_argument("--split", default="dev", choices=("fit", "dev", "test", "all"))
    p.add_argument("--grid-i", type=int, nargs="+", default=None)
    p.add_argument("--grid-j", type=int, nargs="+", default=None)
    p.add_argument("--refit", action="store_true", help="refit adapters that already exist")
    p.set_defaults(func=cmd_sweep)

    p = sub.add_parser("report", help="rebuild the sweep table from saved bench reports")
    common(p)
    variant(p)
    p.add_argument("--split", default="dev", choices=("fit", "dev", "test", "all"))
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("headroom", help="small vs large alone on a bank; no adapter needed")
    common(p)
    p.add_argument("--split", default="dev", choices=("fit", "dev", "test", "all"))
    p.set_defaults(func=cmd_headroom, adapter=ADAPTER_KIND, n_taps=N_TAPS,
                   train_method=TRAIN_METHOD)

    p = sub.add_parser("compare", help="best point per adapter variant, side by side")
    common(p)
    p.add_argument("--split", default="dev", choices=("fit", "dev", "test", "all"))
    p.add_argument("--max-taps", type=int, default=3)
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("check", help="verify the early-exit path against HF's own forward")
    common(p)
    variant(p)
    p.add_argument("--j", type=int, required=True)
    p.set_defaults(func=cmd_check)

    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
