"""Side-by-side benchmark: small vs large vs stitched, on the same prompt set.

Accuracy and latency are measured in *separate* passes, on purpose:

  accuracy — greedy, stops at EOS, budget `bank.max_new_tokens`, scored by
             common.scoring.score (ANY-alias for the factual bank, ALL-items for
             the list bank).
  latency  — greedy, EOS ignored, exactly `bank.latency_steps` decode steps on
             every path, so each path is timed over identical work rather than
             over however many tokens it happened to emit.

Both are then combined into one end-to-end figure. Per-token speedup is the
honest measure of what stitching does, but it is not what a user waits for: that
is prefill plus decode over a whole answer, and the two modes differ sharply
there (`exit` saves prefill, `warm` spends extra). `end_to_end_ms` prices every
path over the *same* answer length — the large model's mean on that split — so
the comparison is a fixed workload and a path cannot look fast by stopping early.

Baseline *accuracy* is deterministic, so it is computed once per split and reused
across a sweep. Baseline *latency* is re-timed alongside every grid point, so a
speedup ratio cannot be corrupted by drift between the start of a sweep and its
end.
"""

from __future__ import annotations

import json

import torch

from stitching_small_to_large.adapter import TorchAdapter
from stitching_small_to_large.adapter import load as load_adapter
from stitching_small_to_large.config import (
    ADAPTER_KIND, Bank, LATENCY_PROMPTS, LATENCY_REPEATS, LATENCY_SPREAD_FLAG,
    MIN_DECODE_SPEEDUP, MIN_E2E_SPEEDUP, N_TAPS, PREFILL_ANOMALY_FACTOR, Pair,
    TRAIN_METHOD, bench_path, results_dir, variant_tag,
)
from stitching_small_to_large.data import split_prompts
from common.decoding import FullRunner, eos_ids, greedy, warmup
from stitching_small_to_large.stitch import StitchRunner
from common.model_utils import LM, build_prompt_ids, load_lm, pick_device
from common.scoring import n_required_present, score
from common.stats import (
    bootstrap_diff, fmt_pct_ci, separated, wilson, wilson_from_rate,
)


class Harness:
    """Both models, loaded once, plus memoised baselines.

    A harness is tied to one (pair, bank, adapter variant): the bank fixes the
    prompts and the decode budgets, and the variant fixes which adapter files are
    read, so two variants cannot be mixed into one report by accident.
    """

    def __init__(self, pair: Pair, bank: Bank, kind: str = ADAPTER_KIND,
                 n_taps: int = N_TAPS, device: str | None = None,
                 train_method: str = TRAIN_METHOD):
        self.pair = pair
        self.bank = bank
        self.kind = kind
        self.n_taps = n_taps
        self.train_method = train_method
        self.device = device or pick_device()
        self.small = load_lm(pair.small_id, pair.small_tag, self.device)
        self.large = load_lm(pair.large_id, pair.large_tag, self.device)
        assert self.small.n_layers == pair.n_layers_small and \
               self.large.n_layers == pair.n_layers_large, \
            "loaded models do not match the geometry in config.PAIRS"
        self.stop = eos_ids(self.large)
        self._baselines: dict[str, dict] = {}
        self._adapters: dict[tuple[int, int], TorchAdapter] = {}

    # ── ids are shared: same family, same tokenizer ───────────────────────────
    def ids_for(self, question: str) -> torch.Tensor:
        return build_prompt_ids(self.large.tokenizer, question, self.device,
                                self.bank.system)

    def decode(self, token_ids: list[int]) -> str:
        return self.large.tokenizer.decode(token_ids, skip_special_tokens=True).strip()

    def adapter(self, i: int, j: int) -> TorchAdapter:
        if (i, j) not in self._adapters:
            arrays, meta = load_adapter(self.pair, i, j, self.bank, self.kind,
                                        self.n_taps, self.train_method)
            self._adapters[(i, j)] = TorchAdapter(arrays, meta, self.device)
        return self._adapters[(i, j)]

    def stitch_runner(self, i: int, j: int, mode: str = "exit") -> StitchRunner:
        return StitchRunner(self.small, self.large, self.adapter(i, j), mode=mode)

    # ── the two measurements ──────────────────────────────────────────────────
    @torch.no_grad()
    def accuracy(self, runner, prompts: list[dict]) -> dict:
        """Greedy, EOS-stopping, scored per the bank's shape.

        `n_items_found` is recorded for conjunctive items even though accuracy is
        all-or-nothing: "4 of 5 required items" and "0 of 5" are both failures but
        only the second means the stitch has stopped producing usable text, and
        that distinction is the first thing to look at when a cell scores badly.
        """
        rows, n_ok, gen_tokens = [], 0, []
        for p in prompts:
            out = greedy(runner, self.ids_for(p["question"]),
                         self.bank.max_new_tokens, self.stop)
            text = self.decode(out["token_ids"])
            ok = score(text, p)
            n_ok += ok
            gen_tokens.append(out["n_generated"])
            row = {"prompt_id": p["id"], "generation": text, "correct": bool(ok),
                   "n_generated": out["n_generated"]}
            if p.get("requires"):
                row["n_items_found"] = n_required_present(text, p["requires"])
                row["n_items_required"] = len(p["requires"])
            rows.append(row)
        n = len(prompts)
        lo, hi = wilson(n_ok, n)
        return {"n": n, "n_correct": n_ok,
                "accuracy": n_ok / n if prompts else float("nan"),
                # Carried on every accuracy result so no table can print a rate
                # without the interval that says whether it is resolvable.
                "accuracy_ci95": [lo, hi],
                "mean_generated_tokens": (sum(gen_tokens) / len(gen_tokens)
                                          if gen_tokens else float("nan")),
                "rows": rows}

    @torch.no_grad()
    def latency(self, runner, prompts: list[dict], steps: int | None = None) -> dict:
        """Headline figures are the *minimum* over probe prompts, not the mean or
        median. Every source of contamination here — GC, scheduler preemption,
        another process touching the GPU, thermal throttling — only ever adds
        time, so the minimum is the least-biased estimate of the steady-state
        cost, and the decode cost genuinely is prompt-independent (same layers,
        same shapes, one token at a time). The full per-prompt spread is kept in
        `decode_ms_per_token_all` so contamination stays visible rather than
        being averaged into the answer.
        """
        steps = steps or self.bank.latency_steps
        probe = prompts[:LATENCY_PROMPTS]
        warmup(runner, self.ids_for(probe[0]["question"]))
        pre, per, tot = [], [], []
        for _ in range(LATENCY_REPEATS):
            for p in probe:
                out = greedy(runner, self.ids_for(p["question"]), steps, stop_ids=None)
                pre.append(out["prefill_ms"])
                per.append(out["decode_ms_per_token"])
                tot.append(out["total_ms"])
        mean = lambda v: sum(v) / len(v)
        med = lambda v: sorted(v)[len(v) // 2]
        spread = max(per) / min(per)
        return {"prefill_ms": min(pre), "decode_ms_per_token": min(per),
                "tokens_per_second": 1e3 / min(per),
                "decode_ms_per_token_all": sorted(per),
                "decode_ms_per_token_median": med(per),
                "decode_spread": spread,
                "latency_suspect": spread > LATENCY_SPREAD_FLAG,
                "total_ms_at_{}_tokens".format(steps): mean(tot),
                "n_timed_prompts": len(probe), "n_repeats": LATENCY_REPEATS,
                "decode_steps": steps,
                "active_params_per_token": runner.active_params()}

    # ── composed reports ──────────────────────────────────────────────────────
    def baseline(self, which: str, split: str, prompts: list[dict]) -> dict:
        """Baseline accuracy is deterministic, so it is computed once per split
        and reused. Baseline *latency* is re-measured on every call: a sweep runs
        for twenty-odd minutes and this machine throttles, so a speedup computed
        against a baseline timed once at the start silently degrades down the
        grid. Timing the reference next to each candidate makes the ratio
        immune to drift, at the cost of ~8s per grid point.
        """
        lm = self.small if which == "small" else self.large
        runner = FullRunner(lm, lm.tag)
        key = f"{which}:{split}"
        if key not in self._baselines:
            self._baselines[key] = self.accuracy(runner, prompts)
        return {"label": lm.tag, "latency": self.latency(runner, prompts),
                "accuracy": self._baselines[key]}

    def bench(self, i: int, j: int, split: str = "dev", mode: str = "exit",
              quiet: bool = False) -> dict:
        prompts = split_prompts(self.bank, split)
        small = self.baseline("small", split, prompts)
        large = self.baseline("large", split, prompts)
        runner = self.stitch_runner(i, j, mode)
        stitch = {"label": runner.label,
                  "latency": self.latency(runner, prompts),
                  "accuracy": self.accuracy(runner, prompts)}

        rep = summarise(self.pair, self.bank, i, j, split, mode, self.device,
                        small, large, stitch, self.adapter(i, j).meta)
        path = bench_path(self.pair, i, j, split, mode, self.bank, self.kind,
                          self.n_taps, self.train_method)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(rep, f, indent=2)
        if not quiet:
            print(format_table(rep))
            print(f"\nWrote {path}")
        return rep


# ── reporting ─────────────────────────────────────────────────────────────────
def priced_tokens(large_report: dict, bank: Bank | None = None,
                  pair: Pair | None = None, split: str | None = None) -> tuple[float, str]:
    """Output length every path is priced over, and where the number came from.

    Never returns NaN: an unpriced row drops `end_to_end_speedup` out of the
    table, which is how a 0.93x result stayed invisible while the decode column
    advertised 1.11x.

    But the fallback must not *flatter*. Pricing short answers at a long budget
    shifts weight onto decode, which is the phase stitching helps, so a
    too-large default turns a loss into a win: the `factual` benches predate
    this field, and pricing their 3.46-token answers at the bank's 24-token cap
    reports 1.06x end-to-end for a cell that actually runs 0.93x. So the order
    of preference is measured-here, measured-elsewhere-on-the-same-split, and
    only then the cap — and the source is returned so the table can say which.
    """
    n = large_report.get("accuracy", {}).get("mean_generated_tokens")
    if n is not None and n == n and n > 0:
        return float(n), "bench"

    rows = large_report.get("accuracy", {}).get("rows") or []
    gen = [r.get("n_generated") for r in rows if r.get("n_generated") is not None]
    if gen:
        return sum(gen) / len(gen), "bench_rows"

    # A headroom run on the same (pair, bank, split) measured exactly this
    # quantity on exactly these prompts. Preferring it over the bank cap is what
    # keeps a backfilled table honest.
    if pair is not None and split is not None:
        hp = results_dir(pair, bank) / f"headroom_{split}.json"
        if hp.exists():
            try:
                with open(hp) as f:
                    h = json.load(f)
                m = h.get("paths", {}).get("large", {}).get("accuracy", {}).get(
                    "mean_generated_tokens")
                if m and m == m and m > 0:
                    return float(m), f"headroom_{split}"
            except (OSError, ValueError, KeyError):
                pass
    if bank is not None:
        return float(bank.max_new_tokens), "bank_cap_UPPER_BOUND"
    return 1.0, "unpriced"


def expected_prefill_ms(mode: str, i: int, j: int, pair: Pair,
                        small_prefill: float, large_prefill: float) -> float:
    """What the stitch's prefill *should* cost, from the layer fractions alone.

    `warm` runs the whole large model over the prompt plus small blocks 0..i-1;
    `exit` runs small blocks 0..i-1 plus large blocks j..end. Both ignore the
    fixed costs (the one-token sink forward, the adapter matmul, per-call
    overhead), so the model is a lower bound and the measured/modelled ratio
    sits above 1 — around 1.13 for warm and 1.58 for exit on this machine.

    It is not used as an absolute threshold for that reason. Its job is to
    normalise cells against each other so `flag_prefill_anomalies` can compare
    like with like across a grid where i and j vary.
    """
    fs = i / pair.n_layers_small
    if mode == "warm":
        return large_prefill + fs * small_prefill
    return fs * small_prefill + ((pair.n_layers_large - j) / pair.n_layers_large) * large_prefill


def flag_prefill_anomalies(rows: list[dict],
                           factor: float = PREFILL_ANOMALY_FACTOR) -> list[dict]:
    """Mark cells whose prefill is out of line with their peers in the same mode.

    This is a cross-cell test because it has to be. `latency_suspect` looks at
    the spread of decode samples *within* one cell, so it catches a machine that
    interfered intermittently but is blind to one that was uniformly slow for
    the whole cell — which is what happened at i=12, j=10 warm: prefill 403.9 ms
    against ~160 ms for its neighbours, decode spread a tight 1.11, flag silent.

    Normalising by `expected_prefill_ms` removes the real variation with i and
    j; taking the median ratio within a mode gives a peer group; anything
    `factor` times worse than that median is contamination, not cost. There is
    no mechanism by which two extra small blocks cost 240 ms of prefill while
    that same cell's decode ratio stays normal.
    """
    for mode in {r["mode"] for r in rows}:
        peers = [r for r in rows if r["mode"] == mode
                 and (r.get("prefill_ratio_vs_model") or 0) > 0]
        if len(peers) < 3:              # too few to have a peer group
            for r in peers:
                r["prefill_suspect"] = False
            continue
        ratios = sorted(r["prefill_ratio_vs_model"] for r in peers)
        med = ratios[len(ratios) // 2]
        for r in peers:
            r["prefill_median_ratio_in_mode"] = med
            r["prefill_suspect"] = bool(r["prefill_ratio_vs_model"] > factor * med)
    return rows


def end_to_end_ms(path_report: dict, n_tokens: float) -> float:
    """Wall-clock time for one request of `n_tokens` output: prefill + decode.

    The token count is the same for every path (see the module docstring), so this
    is a fixed-workload comparison. It is the figure that decides whether `exit`
    or `warm` is the right mode: `exit` starts ahead by the prefill it skips,
    `warm` starts behind by the extra small-model prefill it pays, and they decode
    at the same rate, so `warm` only ever wins on accuracy.
    """
    L = path_report["latency"]
    return L["prefill_ms"] + robust_ms(path_report) * n_tokens


def _paired(a: dict, b: dict) -> dict:
    """Paired bootstrap of accuracy(a) - accuracy(b) over the shared prompts.

    Matching by prompt id rather than trusting row order: the two paths are run
    in separate passes, and pairing the wrong rows would silently turn a paired
    test back into an unpaired one.
    """
    A = {r["prompt_id"]: r["correct"] for r in a["accuracy"]["rows"]}
    B = {r["prompt_id"]: r["correct"] for r in b["accuracy"]["rows"]}
    ids = sorted(set(A) & set(B))
    if not ids:
        return {"diff": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "excludes_zero": False, "n": 0, "n_boot": 0}
    return bootstrap_diff([A[i] for i in ids], [B[i] for i in ids])


def summarise(pair: Pair, bank: Bank, i: int, j: int, split: str, mode: str,
              device: str, small: dict, large: dict, stitch: dict,
              adapter_meta: dict) -> dict:
    acc = lambda d: d["accuracy"]["accuracy"]
    ms = lambda d: d["latency"]["decode_ms_per_token"]
    a_s, a_l, a_x = acc(small), acc(large), acc(stitch)
    held = adapter_meta.get("held_out", {})
    # For a distilled map the shipped weights are not the ridge ones, so the
    # R2 to report is the post-distillation figure. `held_out.answer` is the
    # ridge map's, kept alongside so the trade stays visible.
    shipped = held.get("after_distill") or held
    # Every path is priced over the large model's answer length, so a path that
    # stops early is not credited with being fast.
    n_ref, n_ref_src = priced_tokens(large, bank, pair, split)
    e2e = {k: end_to_end_ms(d, n_ref) for k, d in
           (("small", small), ("large", large), ("stitch", stitch))}
    n = small["accuracy"]["n"]
    ci = {k: d["accuracy"].get("accuracy_ci95")
             or list(wilson(d["accuracy"]["n_correct"], d["accuracy"]["n"]))
          for k, d in (("small", small), ("large", large), ("stitch", stitch))}
    model_prefill = expected_prefill_ms(
        mode, i, j, pair, small["latency"]["prefill_ms"], large["latency"]["prefill_ms"])
    return {
        "pair": pair.name, "bank": bank.name,
        "small_layer_i": i, "large_layer_j": j,
        "depth_matched_j": pair.depth_matched_j(i),
        "split": split, "mode": mode, "device": device,
        "adapter_kind": adapter_meta.get("kind", "linear"),
        "adapter_train_method": adapter_meta.get("train_method", "ridge"),
        "adapter_taps": adapter_meta.get("taps") or [i],
        "n_prompts": n,
        "skipped_large_blocks": j, "skipped_small_blocks": pair.n_layers_small - i,
        "paths": {"small": small, "large": large, "stitch": stitch},
        "tradeoff": {
            "accuracy_small": a_s, "accuracy_large": a_l, "accuracy_stitch": a_x,
            "accuracy_ci95_small": ci["small"], "accuracy_ci95_large": ci["large"],
            "accuracy_ci95_stitch": ci["stitch"],
            # Two tests against the incumbent, because they answer slightly
            # different questions and the cheap one is the wrong one.
            #
            # `beats_small_ci` asks whether two *independent* Wilson intervals
            # separate. They are not independent — both paths are scored on the
            # same prompts and get the same easy ones right — so this is
            # conservative to the point of hiding real effects.
            #
            # `bootstrap_vs_small` resamples prompts and recomputes both
            # accuracies together, preserving that pairing. It is the correct
            # test here and it is materially more powerful: on llama/list_hard
            # the distill-vs-ridge difference at L8->L14 is +10.4 pts
            # [+5.8, +15.6] paired, while the two Wilson intervals overlap.
            "beats_small_ci": bool(separated(stitch["accuracy"]["n_correct"], n,
                                             small["accuracy"]["n_correct"], n)),
            "bootstrap_vs_small": _paired(stitch, small),
            "acc_gap_vs_large_pts": (a_x - a_l) * 100,
            "acc_gain_vs_small_pts": (a_x - a_s) * 100,
            # Where the stitch lands between the two models, 0 = small, 1 = large.
            "gap_recovered": ((a_x - a_s) / (a_l - a_s)) if a_l != a_s else float("nan"),
            "decode_speedup_vs_large": ms(large) / ms(stitch),
            "decode_speedup_small_vs_large": ms(large) / ms(small),
            "answer_tokens_priced": n_ref,
            "answer_tokens_priced_source": n_ref_src,
            "end_to_end_ms": e2e,
            "end_to_end_speedup_vs_large": e2e["large"] / e2e["stitch"],
            "end_to_end_speedup_small_vs_large": e2e["large"] / e2e["small"],
            "stitch_prefill_ms": stitch["latency"]["prefill_ms"],
            "large_prefill_ms": large["latency"]["prefill_ms"],
            "small_prefill_ms": small["latency"]["prefill_ms"],
            "model_prefill_ms": model_prefill,
            "prefill_ratio_vs_model": (stitch["latency"]["prefill_ms"] / model_prefill
                                       if model_prefill > 0 else float("nan")),
            "params_frac_of_large": (stitch["latency"]["active_params_per_token"]
                                     / large["latency"]["active_params_per_token"]),
        },
        "adapter": {k: adapter_meta.get(k) for k in
                    ("kind", "train_method", "n_taps", "taps", "dim_in", "mlp_hidden",
                     "n_map_params", "ridge_alpha", "answer_weight", "norm_match",
                     "norm_gain", "n_rows", "n_answer_rows", "answer_weight_frac")}
        | {"held_out_r2_all": (shipped.get("all") or {}).get("r2"),
           "held_out_r2_answer": (shipped.get("answer") or {}).get("r2"),
           "held_out_r2_answer_ridge": (held.get("answer") or {}).get("r2"),
           "r2_is_post_distill": "after_distill" in held},
    }


def format_table(rep: dict) -> str:
    t, p = rep["tradeoff"], rep["paths"]
    a = rep["adapter"]
    head = (f"{rep['pair']}  bank={rep.get('bank', 'factual')}  "
            f"L{rep['small_layer_i']} -> L{rep['large_layer_j']}  "
            f"adapter={rep.get('adapter_kind', 'linear')}/taps{rep.get('adapter_taps') or ''}  "
            f"mode={rep['mode']}  split={rep['split']}  n={rep['n_prompts']}  "
            f"device={rep['device']}")
    n_ref = t["answer_tokens_priced"]
    lines = [head, "-" * len(head),
             f"{'path':<30}{'acc (95% CI)':>22}{'ms/tok':>9}"
             f"{'prefill':>10}{'ms/answer':>11}{'decode':>9}{'e2e':>8}{'params':>9}"]
    ref = p["large"]["latency"]["decode_ms_per_token"]
    for key in ("small", "large", "stitch"):
        d, L = p[key], p[key]["latency"]
        ci = d["accuracy"].get("accuracy_ci95") or list(
            wilson_from_rate(d["accuracy"]["accuracy"], d["accuracy"]["n"]))
        lines.append(
            f"{d['label']:<30}"
            f"{fmt_pct_ci(d['accuracy']['accuracy'], ci):>22}"
            f"{L['decode_ms_per_token']:>9.1f}"
            f"{L['prefill_ms']:>9.0f}ms"
            f"{t['end_to_end_ms'][key]:>10.0f}ms"
            f"{ref / L['decode_ms_per_token']:>8.2f}x"
            f"{t['end_to_end_ms']['large'] / t['end_to_end_ms'][key]:>7.2f}x"
            f"{L['active_params_per_token'] / p['large']['latency']['active_params_per_token'] * 100:>8.0f}%")
    lines.append("")
    lines.append(f"ms/answer prices every path over {n_ref:.1f} output tokens "
                 f"(the large model's mean on this split), prefill included.")
    # Printed explicitly so a prefill regression cannot hide behind a decode win.
    lines.append(f"prefill    stitch {t['stitch_prefill_ms']:.0f}ms vs large "
                 f"{t['large_prefill_ms']:.0f}ms "
                 f"({t['stitch_prefill_ms'] / t['large_prefill_ms']:.2f}x), "
                 f"layer-fraction model says {t['model_prefill_ms']:.0f}ms "
                 f"(ratio {t['prefill_ratio_vs_model']:.2f})")
    lines.append(f"stitch vs large: {t['acc_gap_vs_large_pts']:+.1f} accuracy pts at "
                 f"{t['decode_speedup_vs_large']:.2f}x decode / "
                 f"{t['end_to_end_speedup_vs_large']:.2f}x end-to-end"
                 + ("" if t["end_to_end_speedup_vs_large"] > 1
                    else "  <- SLOWER end-to-end than the large model"))
    lines.append(f"stitch vs small: {t['acc_gain_vs_small_pts']:+.1f} accuracy pts "
                 f"(gap recovered: {t['gap_recovered']:.0%})")
    b = t.get("bootstrap_vs_small")
    if b:
        lines.append(f"           paired bootstrap vs small: {b['diff'] * 100:+.1f} pts "
                     f"[{b['lo'] * 100:+.1f}, {b['hi'] * 100:+.1f}]  "
                     f"beats small at 95%: {'YES' if b['excludes_zero'] and b['diff'] > 0 else 'NO'}")
    r2a = a.get("held_out_r2_answer")
    if r2a is not None:
        lines.append(f"adapter {a.get('kind', 'linear')} taps={a.get('taps')} "
                     f"{(a.get('n_map_params') or 0) / 1e6:.1f}M params  held-out R2: "
                     f"all={a['held_out_r2_all']:+.3f}  answer={r2a:+.3f}")
    return "\n".join(lines)


def robust_ms(path_report: dict) -> float:
    """Steady-state decode ms/token from a saved bench report (see
    `Harness.latency` for why it is the minimum)."""
    L = path_report["latency"]
    return min(L.get("decode_ms_per_token_all") or [L["decode_ms_per_token"]])


def _spread(path_report: dict) -> float:
    """max/min of a path's timing samples. Decode cost is prompt-independent
    here, so anything much above 1 is interference rather than signal."""
    all_ = path_report["latency"].get("decode_ms_per_token_all")
    return (max(all_) / min(all_)) if all_ else 1.0


def sweep_row(rep: dict, pair: Pair | None = None, bank: Bank | None = None) -> dict:
    """One sweep row from a bench report, with latency recomputed on the
    contamination-resistant statistic.

    Everything derived here is recomputed from the saved paths rather than read
    from `tradeoff`, so a bench written before a column existed still yields a
    complete row. That is what makes `report` a real backfill: the older
    `factual` benches predate end-to-end pricing entirely, and their tables came
    out with the column missing — which is precisely why a best-accuracy point
    running at 0.93x end-to-end was tabulated as a 1.11x win.
    """
    P, t = rep["paths"], rep["tradeoff"]
    stitch_ms, large_ms, small_ms = (robust_ms(P["stitch"]), robust_ms(P["large"]),
                                     robust_ms(P["small"]))
    n_ref = t.get("answer_tokens_priced")
    n_ref_src = t.get("answer_tokens_priced_source") or "bench"
    if not n_ref or n_ref != n_ref:
        n_ref, n_ref_src = priced_tokens(P["large"], bank, pair, rep.get("split"))
    e2e = {k: end_to_end_ms(P[k], n_ref) for k in ("small", "large", "stitch")}

    i, j, mode = rep["small_layer_i"], rep["large_layer_j"], rep["mode"]
    n = P["small"]["accuracy"]["n"]
    ns = {k: P[k]["accuracy"].get("n_correct",
                                  int(round(P[k]["accuracy"]["accuracy"] * n)))
          for k in ("small", "large", "stitch")}
    ci = {k: (P[k]["accuracy"].get("accuracy_ci95")
              or list(wilson_from_rate(P[k]["accuracy"]["accuracy"], n)))
          for k in ("small", "large", "stitch")}

    paired = t.get("bootstrap_vs_small") or _paired(P["stitch"], P["small"])
    prefill_stitch = P["stitch"]["latency"]["prefill_ms"]
    prefill_large = P["large"]["latency"]["prefill_ms"]
    prefill_small = P["small"]["latency"]["prefill_ms"]
    model_prefill = (expected_prefill_ms(mode, i, j, pair, prefill_small, prefill_large)
                     if pair is not None else t.get("model_prefill_ms") or float("nan"))

    return {
        "i": i, "j": j, "mode": mode,
        "kind": rep.get("adapter_kind", "linear"),
        "train_method": rep.get("adapter_train_method",
                                (rep.get("adapter") or {}).get("train_method") or "ridge"),
        "n_taps": len(rep.get("adapter_taps") or [i]),
        "depth_matched_j": rep["depth_matched_j"],
        "n_prompts": n,
        "accuracy_small": t["accuracy_small"], "accuracy_large": t["accuracy_large"],
        "accuracy_stitch": t["accuracy_stitch"],
        "acc_ci_lo_stitch": ci["stitch"][0], "acc_ci_hi_stitch": ci["stitch"][1],
        "acc_ci_lo_small": ci["small"][0], "acc_ci_hi_small": ci["small"][1],
        # The only accuracy comparison the selection rule is allowed to use.
        "beats_small_ci": bool(separated(ns["stitch"], n, ns["small"], n)),
        # Paired bootstrap, recomputed from the saved per-prompt rows so
        # backfilled benches get it too. This is the gate `pareto` uses.
        "boot_diff_vs_small": paired["diff"],
        "boot_lo_vs_small": paired["lo"], "boot_hi_vs_small": paired["hi"],
        "beats_small_boot": bool(paired["excludes_zero"] and paired["diff"] > 0),
        "acc_gap_vs_large_pts": t["acc_gap_vs_large_pts"],
        "acc_gain_vs_small_pts": t["acc_gain_vs_small_pts"],
        "gap_recovered": t["gap_recovered"],
        "decode_speedup_vs_large": large_ms / stitch_ms,
        "decode_speedup_small_vs_large": large_ms / small_ms,
        "answer_tokens_priced": n_ref,
        "answer_tokens_priced_source": n_ref_src,
        "end_to_end_speedup_vs_large": e2e["large"] / e2e["stitch"],
        "end_to_end_speedup_small_vs_large": e2e["large"] / e2e["small"],
        "end_to_end_ms_stitch": e2e["stitch"], "end_to_end_ms_large": e2e["large"],
        "end_to_end_ms_small": e2e["small"],
        "params_frac_of_large": t["params_frac_of_large"],
        "stitch_ms_per_token": stitch_ms, "large_ms_per_token": large_ms,
        "latency_suspect": bool(_spread(P["stitch"]) > LATENCY_SPREAD_FLAG
                                or _spread(P["large"]) > LATENCY_SPREAD_FLAG),
        "decode_spread_stitch": _spread(P["stitch"]),
        "decode_spread_large": _spread(P["large"]),
        "stitch_prefill_ms": prefill_stitch,
        "large_prefill_ms": prefill_large,
        "small_prefill_ms": prefill_small,
        "model_prefill_ms": model_prefill,
        "prefill_ratio_vs_model": (prefill_stitch / model_prefill
                                   if model_prefill and model_prefill == model_prefill
                                   and model_prefill > 0 else float("nan")),
        # Filled by flag_prefill_anomalies once the whole grid is visible.
        "prefill_suspect": False,
        "held_out_r2_answer": rep["adapter"].get("held_out_r2_answer"),
        "held_out_r2_all": rep["adapter"].get("held_out_r2_all"),
        "answer_weight_frac": rep["adapter"].get("answer_weight_frac"),
        "n_answer_rows": rep["adapter"].get("n_answer_rows"),
    }


def load_bench_rows(pair: Pair, split: str, bank: Bank | None = None,
                    kind: str = ADAPTER_KIND, n_taps: int = N_TAPS,
                    train_method: str = TRAIN_METHOD) -> list[dict]:
    """Rebuild sweep rows from every bench report on disk for one split and
    adapter variant. Variants are kept apart so a table never mixes a linear
    result with an mlp one, or a ridge fit with a distilled one, and calls the
    difference an (i, j) effect."""
    from stitching_small_to_large.config import adapter_path

    d = results_dir(pair, bank) / "bench"
    tag = variant_tag(kind, n_taps, train_method)
    files = sorted(f for f in d.glob(f"bench_i*_j*_*_{split}*.json")
                   if _variant_suffix(f.stem, split) == tag)
    if not files:
        raise SystemExit(f"no {kind}/{train_method} bench reports for split={split!r} "
                         f"in {d} — run sweep first.")
    rows = []
    for f in files:
        with open(f) as fh:
            rep = json.load(fh)
        row = sweep_row(rep, pair, bank)
        # The sidecar is the source of truth for the map's quality, and the
        # bench only ever held a copy. Always refresh from it: older benches
        # predate these fields entirely, and benches written before the
        # post-distillation R2 was distinguished from the ridge one cached the
        # wrong figure for distilled maps.
        side = adapter_path(pair, row["i"], row["j"], bank, kind, n_taps,
                            train_method).with_suffix(".json")
        if side.exists():
            with open(side) as fh:
                meta = json.load(fh)
            held = meta.get("held_out", {})
            # `after_distill` when present: for a distilled map the ridge R2 in
            # `held_out` describes weights that did not ship.
            sh = held.get("after_distill") or held
            row["held_out_r2_answer"] = (sh.get("answer") or {}).get("r2")
            row["held_out_r2_all"] = (sh.get("all") or {}).get("r2")
            row["answer_weight_frac"] = meta.get("answer_weight_frac")
            row["n_answer_rows"] = meta.get("n_answer_rows")
        rows.append(row)
    return flag_prefill_anomalies(rows)


def _variant_suffix(stem: str, split: str) -> str | None:
    """The variant tag in a bench filename, or None if the split does not match.

    Exact rather than `endswith`, which is not good enough once tags compose:
    `bench_..._dev_lineart2_distill` ends with `_distill`, so an endswith test
    would pull a two-tap adapter into the one-tap distill table and report the
    difference as an (i, j) effect. Everything after `_<split>` is the tag, and
    it has to match in full.
    """
    marker = f"_{split}"
    k = stem.rfind(marker)
    if k < 0:
        return None
    return stem[k + len(marker):]


def pareto(rows: list[dict], min_speedup: float = MIN_E2E_SPEEDUP) -> dict:
    """Pick the (i, j) to recommend, and mark the frontier.

    Two rules, and the experiment's previous conclusions turned on both:

    **The speed axis is end-to-end, not per-token.** Decode speedup is the
    mechanism; end-to-end is what a user waits for, and `warm` mode pays a full
    large prefill *plus* a small one to get its decode win. At the `factual`
    bank's 3.46-token answers that made the best-accuracy cell 0.93x — slower
    than the large model — while its decode column advertised 1.11x. Selecting
    on decode recommends cells that lose wall-clock time.

    **The small model competes, and accuracy is compared with intervals.**
    Ranking stitch cells only against each other will happily recommend one that
    is both slower and less accurate than simply running the small model, which
    is always available at its own speedup. So the small model enters the
    frontier as a real point, and a cell it dominates on both axes is marked
    `dominated_by_small` and never recommended. Beating it on the accuracy axis
    means beating it with *non-overlapping* Wilson intervals: on a 35-prompt
    split one prompt is 2.9 points and the interval is ~23 points wide, so a
    point-estimate comparison between neighbouring cells is reading noise.

    A recommendation must therefore (a) clear `min_speedup` end-to-end and (b)
    beat the small model's accuracy at 95%. If nothing qualifies, `best` is None
    and the verdict says so — reporting no viable point is a valid outcome, and
    a better one than dressing up a dominated cell.
    """
    small_acc = rows[0]["accuracy_small"]
    small_speedup = rows[0]["end_to_end_speedup_small_vs_large"]
    field = [(r["end_to_end_speedup_vs_large"], r["accuracy_stitch"]) for r in rows]
    field.append((small_speedup, small_acc))

    for r in rows:
        s, a = r["end_to_end_speedup_vs_large"], r["accuracy_stitch"]
        r["on_frontier"] = not any(
            os_ >= s and oa >= a and (os_ > s or oa > a) for os_, oa in field)
        r["dominated_by_small"] = bool(small_speedup >= s and small_acc >= a)
        # Point estimate (kept for the table) vs the interval test (used to pick).
        r["beats_small_accuracy"] = bool(a >= small_acc)
        r["faster_than_large_e2e"] = bool(r["end_to_end_speedup_vs_large"] > 1.0)

    fast = [r for r in rows if r["end_to_end_speedup_vs_large"] >= min_speedup]
    # The paired bootstrap is the gate. Both paths are scored on the same
    # prompts, so comparing two independent Wilson intervals throws away the
    # pairing and is conservative enough to hide real effects — it is still
    # computed and reported (`beats_small_ci`), but it does not decide.
    viable = [r for r in fast if r.get("beats_small_boot")]
    best = (max(viable, key=lambda r: (r["accuracy_stitch"],
                                       r["end_to_end_speedup_vs_large"]))
            if viable else None)

    # Reported whether or not anything is viable: the honest answer to "is any
    # of this actually faster than just running the large model?"
    e2e_faster = [r for r in rows if r["faster_than_large_e2e"]]
    best_e2e_faster = (max(e2e_faster, key=lambda r: (r["accuracy_stitch"],
                                                      r["end_to_end_speedup_vs_large"]))
                       if e2e_faster else None)

    n_dom = sum(r["dominated_by_small"] for r in rows)
    if best is None:
        top = max(rows, key=lambda r: r["accuracy_stitch"])
        tail = (f"The small model dominates every point on both axes; running it "
                f"directly is the better trade."
                if n_dom == len(rows) else
                f"{n_dom}/{len(rows)} points are dominated by the small model outright; "
                f"the rest buy accuracy only by giving up the speedup that motivates "
                f"stitching at all.")
        verdict = (
            f"NO VIABLE POINT. No stitch cell is both >= {min_speedup:.2f}x faster "
            f"end-to-end than the large model and more accurate than the small model "
            f"at 95% confidence. The small model scores {small_acc:.1%} and already "
            f"runs {small_speedup:.2f}x faster end-to-end than large. Best stitch "
            f"accuracy was {top['accuracy_stitch']:.1%} "
            f"[{top['acc_ci_lo_stitch']:.1%}-{top['acc_ci_hi_stitch']:.1%}] at "
            f"L{top['i']}->L{top['j']} ({top['mode']}, "
            f"{top['end_to_end_speedup_vs_large']:.2f}x end-to-end), "
            f"{top['boot_diff_vs_small'] * 100:+.1f} pts vs small "
            f"[{top['boot_lo_vs_small'] * 100:+.1f}, {top['boot_hi_vs_small'] * 100:+.1f}] "
            f"paired. {tail}")
    else:
        verdict = (f"L{best['i']}->L{best['j']} ({best['mode']}, {best['kind']}/"
                   f"{best['train_method']}): {best['accuracy_stitch']:.1%} "
                   f"[{best['acc_ci_lo_stitch']:.1%}-{best['acc_ci_hi_stitch']:.1%}], "
                   f"{best['boot_diff_vs_small'] * 100:+.1f} pts vs the small model "
                   f"[{best['boot_lo_vs_small'] * 100:+.1f}, "
                   f"{best['boot_hi_vs_small'] * 100:+.1f}] paired, at "
                   f"{best['end_to_end_speedup_vs_large']:.2f}x end-to-end "
                   f"({best['decode_speedup_vs_large']:.2f}x decode).")
    return {"best": best, "verdict": verdict, "min_speedup": min_speedup,
            "speed_axis": "end_to_end_speedup_vs_large",
            "n_meeting_speedup": len(fast), "n_viable": len(viable),
            "small_accuracy": small_acc,
            "small_speedup_vs_large_e2e": small_speedup,
            "n_dominated_by_small": n_dom,
            "n_faster_than_large_e2e": len(e2e_faster),
            "best_e2e_faster": best_e2e_faster,
            "n_prefill_suspect": sum(bool(r.get("prefill_suspect")) for r in rows)}
