"""Side-by-side evaluation: small vs large vs stitched, accuracy first.

Accuracy is the primary number and is measured on the whole eval split *and*
restricted to the divergent subset — the prompts where the small model was wrong
and the large model was right. That subset is the only place a large->small
stitch can demonstrably add anything: everywhere else the small model was
already correct, so a stitch can at best avoid breaking it.

Latency is reported but is not the goal. The stitched path runs j large blocks
plus (n_small - i) small blocks, so it is slower than the small model and
usually slower than the large one; printing that keeps the cost visible.
"""

from __future__ import annotations

import json

import torch

from common.decoding import FullRunner, eos_ids, greedy, warmup
from common.model_utils import build_prompt_ids, load_lm, pick_device
from common.scoring import score
from common.stats import fmt_pct_ci, separated, wilson, wilson_from_rate
from stitching_large_to_small.adapter import TorchAdapter
from stitching_large_to_small.adapter import load as load_adapter
from stitching_large_to_small.config import (
    LATENCY_PROMPTS, LATENCY_REPEATS, LATENCY_SPREAD_FLAG, Bank, Pair, bench_path,
    results_dir,
)
from stitching_large_to_small.data import split_prompts
from stitching_large_to_small.stitch import LargeToSmallRunner


class Harness:
    """Both models, loaded once, plus memoised baseline accuracy."""

    def __init__(self, pair: Pair, bank: Bank, device: str | None = None):
        self.pair, self.bank = pair, bank
        self.device = device or pick_device()
        self.small = load_lm(pair.small_id, pair.small_tag, self.device)
        self.large = load_lm(pair.large_id, pair.large_tag, self.device)
        assert self.small.n_layers == pair.n_layers_small and \
            self.large.n_layers == pair.n_layers_large, \
            "loaded models do not match the geometry in config.PAIRS"
        self.stop = eos_ids(self.small)
        self._acc: dict[str, dict] = {}
        self._adapters: dict[tuple[int, int], TorchAdapter] = {}

    def ids_for(self, question: str) -> torch.Tensor:
        return build_prompt_ids(self.small.tokenizer, question, self.device,
                                self.bank.system)

    def decode(self, token_ids: list[int]) -> str:
        return self.small.tokenizer.decode(token_ids, skip_special_tokens=True).strip()

    def adapter(self, i: int, j: int) -> TorchAdapter:
        if (i, j) not in self._adapters:
            arrays, meta = load_adapter(self.pair, i, j, self.bank)
            self._adapters[(i, j)] = TorchAdapter(arrays, meta, self.device)
        return self._adapters[(i, j)]

    def stitch_runner(self, i: int, j: int, mode: str = "exit") -> LargeToSmallRunner:
        return LargeToSmallRunner(self.small, self.large, self.adapter(i, j), mode=mode)

    # ── the two measurements ──────────────────────────────────────────────────
    @torch.no_grad()
    def accuracy(self, runner, prompts: list[dict]) -> dict:
        rows, n_ok, n_tok = [], 0, 0
        for p in prompts:
            out = greedy(runner, self.ids_for(p["question"]),
                         self.bank.max_new_tokens, self.stop)
            text = self.decode(out["token_ids"])
            ok = score(text, p)
            n_ok += ok
            n_tok += out["n_generated"]
            rows.append({"prompt_id": p["id"], "gold": p.get("answers") or p.get("requires"),
                         "generation": text, "correct": bool(ok),
                         "n_generated": out["n_generated"]})
        n = len(prompts)
        lo, hi = wilson(n_ok, n)
        return {"n": n, "n_correct": n_ok, "accuracy": n_ok / n if n else float("nan"),
                "accuracy_ci95": [lo, hi],
                "mean_generated_tokens": n_tok / n if n else float("nan"), "rows": rows}

    @torch.no_grad()
    def latency(self, runner, prompts: list[dict]) -> dict:
        probe = prompts[:LATENCY_PROMPTS]
        warmup(runner, self.ids_for(probe[0]["question"]))
        pre, per = [], []
        for _ in range(LATENCY_REPEATS):
            for p in probe:
                out = greedy(runner, self.ids_for(p["question"]),
                             self.bank.latency_steps, stop_ids=None)
                pre.append(out["prefill_ms"])
                per.append(out["decode_ms_per_token"])
        spread = max(per) / min(per)
        return {"prefill_ms": min(pre), "decode_ms_per_token": min(per),
                "tokens_per_second": 1e3 / min(per),
                "decode_ms_per_token_all": sorted(per),
                "decode_spread": spread,
                "latency_suspect": spread > LATENCY_SPREAD_FLAG,
                "n_timed_prompts": len(probe), "n_repeats": LATENCY_REPEATS,
                "decode_steps": self.bank.latency_steps,
                "active_params_per_token": runner.active_params()}

    def baseline(self, which: str, split: str, prompts: list[dict]) -> dict:
        """Accuracy is deterministic and memoised per split; latency is re-timed
        alongside every candidate so a ratio cannot drift across a long sweep."""
        lm = self.small if which == "small" else self.large
        runner = FullRunner(lm, lm.tag)
        key = f"{which}:{split}"
        if key not in self._acc:
            self._acc[key] = self.accuracy(runner, prompts)
        return {"label": lm.tag, "latency": self.latency(runner, prompts),
                "accuracy": self._acc[key]}

    def bench(self, i: int, j: int, split: str = "dev", mode: str = "exit",
              quiet: bool = False) -> dict:
        prompts = split_prompts(self.bank, split)
        small = self.baseline("small", split, prompts)
        large = self.baseline("large", split, prompts)
        runner = self.stitch_runner(i, j, mode)
        stitch = {"label": runner.label, "latency": self.latency(runner, prompts),
                  "accuracy": self.accuracy(runner, prompts)}

        rep = summarise(self.pair, self.bank, i, j, split, mode, self.device,
                        small, large, stitch, self.adapter(i, j).meta)
        path = bench_path(self.pair, i, j, split, mode, self.bank)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(rep, f, indent=2)
        if not quiet:
            print(format_table(rep))
            print(f"\nWrote {path}")
        return rep


# ── reporting ─────────────────────────────────────────────────────────────────
def priced_tokens(large: dict, bank: Bank, pair: Pair | None = None,
                  split: str | None = None) -> tuple[float, str]:
    """Output length every path is priced over, and its provenance.

    Mirrors the sibling package deliberately. Even though this direction is not
    a latency play, its cost has to be quotable end-to-end rather than only
    per-token: "1.7x the small model's ms/token" understates a path that also
    pays a full large-model prefix prefill, and a cost that is understated is a
    cost that gets dropped from the story.

    The fallback order avoids flattering the stitch: pricing short answers at a
    long budget would shift weight onto decode and shrink the prefill penalty
    this path actually carries.
    """
    n = large.get("accuracy", {}).get("mean_generated_tokens")
    if n is not None and n == n and n > 0:
        return float(n), "bench"
    rows = large.get("accuracy", {}).get("rows") or []
    gen = [r.get("n_generated") for r in rows if r.get("n_generated") is not None]
    if gen:
        return sum(gen) / len(gen), "bench_rows"
    if pair is not None and split is not None:
        hp = results_dir(pair, bank) / f"headroom_{split}.json"
        if hp.exists():
            try:
                with open(hp) as f:
                    h = json.load(f)
                m = h.get("mean_generated_tokens_large")
                if m and m == m and m > 0:
                    return float(m), f"headroom_{split}"
            except (OSError, ValueError, KeyError):
                pass
    return float(bank.max_new_tokens), "bank_cap_UPPER_BOUND"


def end_to_end_ms(path: dict, n_tokens: float) -> float:
    L = path["latency"]
    return L["prefill_ms"] + L["decode_ms_per_token"] * n_tokens


def divergent_ids(small: dict, large: dict) -> list[str]:
    """Prompts the small model got wrong and the large model got right — the
    only subset where a large->small stitch has anything to prove."""
    s = {r["prompt_id"]: r["correct"] for r in small["accuracy"]["rows"]}
    l = {r["prompt_id"]: r["correct"] for r in large["accuracy"]["rows"]}
    return sorted(pid for pid in s if not s[pid] and l.get(pid))


def subset_accuracy(path: dict, ids: set[str]) -> float:
    rows = [r for r in path["accuracy"]["rows"] if r["prompt_id"] in ids]
    return (sum(r["correct"] for r in rows) / len(rows)) if rows else float("nan")


def summarise(pair: Pair, bank: Bank, i: int, j: int, split: str, mode: str,
              device: str, small: dict, large: dict, stitch: dict,
              adapter_meta: dict) -> dict:
    acc = lambda d: d["accuracy"]["accuracy"]
    ms = lambda d: d["latency"]["decode_ms_per_token"]
    a_s, a_l, a_x = acc(small), acc(large), acc(stitch)
    div = divergent_ids(small, large)
    dset = set(div)
    held = adapter_meta.get("held_out", {})

    # Per-prompt side-by-side, which is what makes a claim auditable.
    by = lambda d: {r["prompt_id"]: r for r in d["accuracy"]["rows"]}
    S, L, X = by(small), by(large), by(stitch)
    rows = [{"prompt_id": pid, "gold": X[pid]["gold"], "divergent": pid in dset,
             "small": S[pid]["generation"], "small_correct": S[pid]["correct"],
             "large": L[pid]["generation"], "large_correct": L[pid]["correct"],
             "stitch": X[pid]["generation"], "stitch_correct": X[pid]["correct"]}
            for pid in X]

    n = small["accuracy"]["n"]
    ci = {k: (d["accuracy"].get("accuracy_ci95")
              or list(wilson_from_rate(d["accuracy"]["accuracy"], n)))
          for k, d in (("small", small), ("large", large), ("stitch", stitch))}
    nc = {k: d["accuracy"].get("n_correct",
                               int(round(d["accuracy"]["accuracy"] * n)))
          for k, d in (("small", small), ("large", large), ("stitch", stitch))}
    n_ref, n_ref_src = priced_tokens(large, bank, pair, split)
    e2e = {k: end_to_end_ms(d, n_ref) for k, d in
           (("small", small), ("large", large), ("stitch", stitch))}
    return {
        "pair": pair.name, "bank": bank.name, "direction": "large->small",
        "small_layer_i": i, "large_layer_j": j,
        "depth_matched_i": pair.depth_matched_i(j),
        "split": split, "mode": mode, "device": device,
        "n_prompts": n,
        "adapter_kind": adapter_meta.get("kind", "linear"),
        "adapter_train_method": adapter_meta.get("train_method", "ridge"),
        "seed_provenance": {"seed": adapter_meta.get("seed"),
                            "ridge_alpha": adapter_meta.get("ridge_alpha"),
                            "answer_weight": adapter_meta.get("answer_weight"),
                            "norm_match": adapter_meta.get("norm_match")},
        "accuracy_small": a_s, "accuracy_large": a_l, "accuracy_stitch": a_x,
        "accuracy_ci95_small": ci["small"], "accuracy_ci95_large": ci["large"],
        "accuracy_ci95_stitch": ci["stitch"],
        # The bar this direction has to clear, and the only form of it that
        # means anything at n in the low hundreds: strictly above the small
        # model with non-overlapping 95% intervals.
        "beats_small_ci": bool(separated(nc["stitch"], n, nc["small"], n)),
        "acc_gain_vs_small_pts": (a_x - a_s) * 100,
        "acc_gap_vs_large_pts": (a_x - a_l) * 100,
        "answer_tokens_priced": n_ref,
        "answer_tokens_priced_source": n_ref_src,
        "end_to_end_ms": e2e,
        # Reported even though this direction never claims a speed win: the
        # path runs j large blocks AND (n_small - i) small ones, and the
        # per-token ratio alone hides the extra prefix prefill.
        "end_to_end_slowdown_vs_small": e2e["stitch"] / e2e["small"],
        "end_to_end_speedup_vs_large": e2e["large"] / e2e["stitch"],
        "divergent": {
            "n": len(div), "prompt_ids": div,
            "accuracy_small": 0.0 if div else float("nan"),
            "accuracy_large": 1.0 if div else float("nan"),
            "accuracy_stitch": subset_accuracy(stitch, dset),
        },
        "latency": {k: {"decode_ms_per_token": ms(v), "prefill_ms": v["latency"]["prefill_ms"],
                        "active_params_per_token": v["latency"]["active_params_per_token"],
                        "latency_suspect": v["latency"]["latency_suspect"]}
                    for k, v in (("small", small), ("large", large), ("stitch", stitch))},
        "decode_slowdown_vs_small": ms(stitch) / ms(small),
        "decode_speedup_vs_large": ms(large) / ms(stitch),
        "adapter": {"held_out_r2_all": held.get("all", {}).get("r2"),
                    "held_out_r2_answer": (held.get("answer") or {}).get("r2"),
                    "norm_gain": adapter_meta.get("norm_gain"),
                    "n_rows": adapter_meta.get("n_rows"),
                    "n_answer_rows": adapter_meta.get("n_answer_rows")},
        "paths": {"small": small, "large": large, "stitch": stitch},
        "rows": rows,
    }


def verdict(rep: dict) -> tuple[str, str]:
    """(status, sentence). Deliberately blunt: the failure case is the likely
    one and must not be reported as a partial success.

    Three outcomes, not two. The bar for this direction is
    `accuracy_stitch > accuracy_small with non-overlapping CIs`, and a point
    estimate that clears the small model without clearing its interval is
    neither a success nor a failure — it is an unresolved measurement, and
    calling it either would be the same error the sibling package made when it
    read a 65.7%/85.7% dev/test swing on one adapter as a 20-point effect.
    """
    a_s, a_x, a_l = rep["accuracy_small"], rep["accuracy_stitch"], rep["accuracy_large"]
    gain = rep["acc_gain_vs_small_pts"]
    dstitch = rep["divergent"]["accuracy_stitch"]
    n_div = rep["divergent"]["n"]
    cs, cx = rep.get("accuracy_ci95_small"), rep.get("accuracy_ci95_stitch")
    n = rep["n_prompts"]
    cost = (f" It costs {rep['end_to_end_slowdown_vs_small']:.2f}x the small model's "
            f"end-to-end time." if "end_to_end_slowdown_vs_small" in rep else "")
    garbage = sum(1 for r in rep["rows"] if not r["stitch_correct"]
                  and not r["stitch"].strip()) / max(1, len(rep["rows"]))
    if a_x <= a_s:
        return "FAILURE", (
            f"stitched accuracy {a_x:.1%} does not beat the small model alone "
            f"({a_s:.1%}) — and the stitch costs a partial large-model forward on top. "
            f"Running the small model directly is strictly better here.")
    if garbage > 0.1:
        return "FAILURE", (f"{garbage:.0%} of stitched generations are empty; the "
                           f"injection is producing degenerate output.")
    if not rep.get("beats_small_ci"):
        return "INCONCLUSIVE", (
            f"stitched accuracy {a_x:.1%} "
            + (f"[{cx[0]:.1%}-{cx[1]:.1%}] " if cx else "")
            + f"is above the small model's {a_s:.1%} "
            + (f"[{cs[0]:.1%}-{cs[1]:.1%}] " if cs else "")
            + f"by {gain:+.1f} pts, but the 95% intervals overlap at n={n}, so the "
              f"difference is not resolved. This is not a win; it needs a larger "
              f"eval split before it can be called one.")
    recovered = "" if n_div == 0 else (
        f" On the {n_div} divergent prompts (small wrong, large right) it recovers "
        f"{dstitch:.1%}.")
    return "USEFUL", (
        f"stitched accuracy {a_x:.1%} beats the small model alone ({a_s:.1%}) by "
        f"{gain:+.1f} pts with non-overlapping 95% intervals, against a ceiling of "
        f"{a_l:.1%} (the large model).{recovered}{cost}")


def format_table(rep: dict) -> str:
    head = (f"{rep['pair']}/{rep['bank']}  large L{rep['large_layer_j']} -> "
            f"small L{rep['small_layer_i']}  mode={rep['mode']}  split={rep['split']}  "
            f"n={rep['n_prompts']}  device={rep['device']}")
    lines = [head, "-" * len(head),
             f"{'path':<26}{'acc (95% CI)':>22}{'divergent':>11}{'ms/tok':>9}"
             f"{'prefill':>10}{'ms/answer':>11}{'params':>9}"]
    lp = rep["latency"]
    dset = set(rep["divergent"]["prompt_ids"])
    e2e = rep.get("end_to_end_ms", {})
    for key in ("small", "large", "stitch"):
        d, L = rep["paths"][key], lp[key]
        sub = subset_accuracy(d, dset) if dset else float("nan")
        ci = rep.get(f"accuracy_ci95_{key}") or list(
            wilson_from_rate(d["accuracy"]["accuracy"], rep["n_prompts"]))
        lines.append(
            f"{d['label']:<26}"
            f"{fmt_pct_ci(d['accuracy']['accuracy'], ci):>22}"
            f"{sub * 100:>10.1f}%"
            f"{L['decode_ms_per_token']:>9.1f}"
            f"{L['prefill_ms']:>9.0f}ms"
            f"{e2e.get(key, float('nan')):>10.0f}ms"
            f"{L['active_params_per_token'] / 1e9:>8.2f}B")
    status, sentence = verdict(rep)
    lines += ["",
              f"ms/answer prices every path over "
              f"{rep.get('answer_tokens_priced', float('nan')):.2f} output tokens "
              f"(source: {rep.get('answer_tokens_priced_source', 'bench')}).",
              f"accuracy   stitch {rep['accuracy_stitch']:.1%}   "
              f"small {rep['accuracy_small']:.1%}   large {rep['accuracy_large']:.1%}",
              f"           {rep['acc_gain_vs_small_pts']:+.1f} pts vs small, "
              f"{rep['acc_gap_vs_large_pts']:+.1f} pts vs large; beats small at 95%: "
              f"{'YES' if rep.get('beats_small_ci') else 'NO'}",
              f"cost       {rep['decode_slowdown_vs_small']:.2f}x the small model's "
              f"ms/token, {rep.get('end_to_end_slowdown_vs_small', float('nan')):.2f}x "
              f"its end-to-end time ({rep['decode_speedup_vs_large']:.2f}x decode "
              f"vs large)",
              f"adapter    held-out R2 answer="
              f"{rep['adapter']['held_out_r2_answer']:+.3f} "
              f"all={rep['adapter']['held_out_r2_all']:+.3f}",
              "", f"{status}: {sentence}"]
    return "\n".join(lines)
