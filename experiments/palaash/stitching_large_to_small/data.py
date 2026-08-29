"""Prompt splits and paired hidden-state capture (large -> small).

Capture is direction-agnostic in mechanism and directional in meaning: each fit
item is run through the *large* model to get its greedy answer, then
`prompt + answer` is teacher-forced through both models and the hidden states of
both are recorded at every position, with answer positions flagged.

The direction shows up at fit time, not here: the large model's layer-j states
are the adapter's input `X` and the small model's layer-i states are the ridge
target `Y`, which is the reverse of the sibling package. `Y` keeps that name
under distillation too, but its role changes — there it is the *base stream*
the frozen small suffix attends back to at prompt positions, not a regression
target.

Why the teacher answer is the *large* model's: it is the trajectory we want the
stitched path to reproduce. Fitting on the small model's own answer would teach
the adapter to reconstruct the very output we are trying to improve on.

Three things shape the fit distribution, and all three were wrong in the capture
that produced the published all-ridge failure:

`vary_templates`      paraphrases the system prompt and the framing around each
    question, so prompt positions stop being one boilerplate prefix repeated
    once per prompt. See common/templates.py.
`fit_corpus`          mixes in open-ended generic instructions whose
    continuations run 40-80 tokens. `hard_factual` answers are 3-8 tokens, so
    the bank alone cannot produce answer rows in the tens of thousands at any
    reasonable prompt count: the old capture had 753 of them against 15555
    prompt rows. The corpus is never an eval bank, and the overlap is checked.
`store_teacher_logits` records the *large* model's own next-token distribution
    (top-K) at every position. That is the target the distill training method
    fits, and it is captured here because recomputing it inside the training
    loop would mean a full large-model forward per step.

Outputs (results/<pair>/<bank>/states/):
    x_large.npz       — one key per captured large layer, (N, dim_large) float16
    y_small.npz       — one key per captured small layer, (N, dim_small) float16
    meta.npz          — prompt_index, is_answer, position, source
    teacher_topk.npz  — values, indices: the large model's top-K next-token logits
    meta.json         — prompt ids, layers, token counts, answer-weight fraction
"""

from __future__ import annotations

import json

import numpy as np
import torch

from common.fit_corpus import load_corpus, overlap_with
from common.model_utils import LM, build_prompt_ids, load_lm, pick_device
from common.templates import apply_framing, variant_for
from stitching_large_to_small.config import (
    ANSWER_WEIGHT, BANKS, CORPUS_ANSWER_TOKENS, DEFAULT_BANK, DEFAULT_PAIR,
    DISTILL_TOPK, MIN_ANSWER_WEIGHT_FRAC, PAIRS, PROMPT_ROW_KEEP, SEED,
    SPLIT_FRACS, Bank, Pair, capture_layers, states_dir,
)


# ── splits ────────────────────────────────────────────────────────────────────
def by_id(bank: Bank) -> dict[str, dict]:
    return {p["id"]: p for p in bank.prompts}


def splits(bank: Bank) -> dict[str, list[str]]:
    """Deterministic three-way split of a bank by prompt id.

    Three ways rather than two so the sweep does not select (i, j) on the same
    prompts it then reports. Seeded on the bank name as well as SEED, so two
    banks of the same size do not receive correlated splits.

    A bank whose items carry their own `split` is taken at its word: `list_hard`
    partitions the underlying *facts* before composing, which is the only way to
    keep a fact out of both the fit and eval sets.
    """
    items = by_id(bank)
    if all(p.get("split") for p in items.values()):
        return {s: sorted(pid for pid, p in items.items() if p["split"] == s)
                for s in ("fit", "dev", "test")}
    ids = sorted(items)
    rng = np.random.default_rng([SEED, len(ids)])
    rng.shuffle(ids)
    n = len(ids)
    n_fit = int(round(n * SPLIT_FRACS["fit"]))
    n_dev = int(round(n * SPLIT_FRACS["dev"]))
    return {"fit": ids[:n_fit], "dev": ids[n_fit:n_fit + n_dev],
            "test": ids[n_fit + n_dev:]}


def split_prompts(bank: Bank, split: str) -> list[dict]:
    items = by_id(bank)
    if split == "all":
        return [items[i] for i in sorted(items)]
    s = splits(bank)
    if split not in s:
        raise SystemExit(f"unknown split {split!r}; use one of {sorted(s)} or 'all'")
    return [items[i] for i in s[split]]


# ── capture ───────────────────────────────────────────────────────────────────
@torch.no_grad()
def teacher_sequence(lm_large: LM, question: str, answer_tokens: int,
                     system: str | None) -> tuple[torch.Tensor, int]:
    """(prompt + the large model's own greedy answer) as one id tensor, plus the
    prompt length."""
    ids = build_prompt_ids(lm_large.tokenizer, question, lm_large.device, system)
    out = lm_large.model.generate(
        ids,
        attention_mask=torch.ones_like(ids),
        max_new_tokens=answer_tokens,
        do_sample=False,
        pad_token_id=lm_large.tokenizer.eos_token_id,
    )
    return out, ids.shape[1]


@torch.no_grad()
def layer_states(lm: LM, ids: torch.Tensor, layers: list[int],
                 want_logits: bool = False):
    """Hidden states at `layers`, and optionally the model's own logits.

    The logits are the teacher signal for the distill training method: the
    distribution the stitched path is asked to reproduce. Here they come from
    the *large* model — the one being distilled *from* — which is the same
    forward pass that produces X, so storing them costs one top-K sort per item
    rather than a second pass over the model.
    """
    out = lm.model(ids, output_hidden_states=True)
    states = {L: out.hidden_states[L][0].float().cpu().numpy().astype(np.float16)
              for L in layers}
    return (states, out.logits[0].float()) if want_logits else (states, None)


def topk_teacher(logits: torch.Tensor, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Top-k teacher logits and their token ids, as (rows, k) arrays.

    Full-vocab targets are not storable at this scale — 128256 floats per row
    against ~90k rows is 20 GB at fp16 — and are not needed: the tail of a
    next-token distribution the large model would greedily decode from carries
    almost no mass, and KL over the renormalised top-k support is within noise
    of the full-vocab quantity for k in the hundreds. k is recorded in the
    sidecar so the approximation is never invisible.

    The two models share a tokenizer (token-aligned pairs only), so these ids
    index the small model's vocabulary too — which is what makes the stored
    teacher usable as a target for the stitched path's logits.
    """
    vals, idx = torch.topk(logits, k, dim=-1)
    return (vals.cpu().numpy().astype(np.float16),
            idx.cpu().numpy().astype(np.int32))


def _prompt_row_budget(n_answer: int, n_prompt: int, answer_weight: float,
                       policy=PROMPT_ROW_KEEP,
                       min_frac: float = MIN_ANSWER_WEIGHT_FRAC) -> int:
    """How many prompt-position rows to keep so answer rows carry the objective.

    Answer rows contribute `n_answer * answer_weight` to the total weight and
    prompt rows contribute one each, so clearing `min_frac` needs

        n_answer*aw / (n_answer*aw + n_keep) > min_frac

    The published capture fails this badly at 16%: 753 answer rows at weight 4
    against 15555 prompt rows. Solving for n_keep and leaving a 5% margin is
    what this returns.

    Prompt rows are thinned rather than dropped. They still matter — `exit` mode
    hands the adapter prompt positions at inference, and they anchor the
    standardiser's mean and scale — they just must not *be* the objective.
    """
    if policy == "all":
        return n_prompt
    if isinstance(policy, int):
        return min(policy, n_prompt)
    aw = n_answer * answer_weight
    target = min_frac + 0.05
    budget = int(aw * (1 - target) / target)
    return max(0, min(n_prompt, budget))


def run(pair: Pair, bank: Bank, split: str = "fit", max_prompts: int | None = None,
        answer_tokens: int | None = None,
        lm_small: LM | None = None, lm_large: LM | None = None,
        fit_corpus: str | None = None, max_prompts_corpus: int | None = None,
        corpus_answer_tokens: int = CORPUS_ANSWER_TOKENS,
        vary_templates: bool = True, store_teacher_logits: bool = True,
        topk: int = DISTILL_TOPK,
        small_layers: list[int] | None = None,
        large_layers: list[int] | None = None) -> dict:
    """Capture paired states (and teacher distributions) for the adapter fit."""
    answer_tokens = answer_tokens or bank.teacher_answer_tokens
    prompts = split_prompts(bank, split)
    if max_prompts:
        prompts = prompts[:max_prompts]
    if split != "fit":
        # Hard error rather than the old warning. Fitting on dev/test and then
        # scoring on it does not produce a slightly optimistic number, it
        # produces a meaningless one, and a printed warning scrolls off the top
        # of a capture log that runs for half an hour.
        raise SystemExit(
            f"refusing to capture on split={split!r}. The adapter is fit on whatever "
            f"is captured, so any bench or sweep on 'dev'/'test' afterwards would be "
            f"scoring on prompts the map was trained on. Capture uses split='fit'. "
            f"If you are deliberately measuring the leak, call data.run(..., "
            f"split={split!r}) from Python where the intent is explicit.")

    items = [dict(p, _source="bank") for p in prompts]
    if fit_corpus:
        corpus = load_corpus(fit_corpus, max_prompts_corpus)
        clash = overlap_with(corpus, bank.prompts)
        if clash:
            raise SystemExit(
                f"fit corpus {fit_corpus!r} overlaps the {bank.name} bank on "
                f"{len(clash)} items ({clash[:5]}). The adapter would be fit on "
                f"prompts it is later scored on.")
        items += [dict(c, _source="corpus") for c in corpus]

    cap_small, cap_large = capture_layers(pair)
    small_layers = sorted(small_layers or cap_small)
    large_layers = sorted(large_layers or cap_large)

    if lm_small is None or lm_large is None:
        device = pick_device()
        lm_small = lm_small or load_lm(pair.small_id, pair.small_tag, device)
        lm_large = lm_large or load_lm(pair.large_id, pair.large_tag, device)
    assert lm_small.n_layers == pair.n_layers_small and \
        lm_large.n_layers == pair.n_layers_large, \
        "loaded models do not match the geometry in config.PAIRS"

    xs = {L: [] for L in large_layers}     # adapter input  (large)
    ys = {L: [] for L in small_layers}     # small-model base stream / ridge target
    prompt_index, is_answer, position, source = [], [], [], []
    tlog, tidx = [], []

    print(f"[capture] {pair.name} bank={bank.name} split={split} "
          f"items={len(items)} (bank {len(prompts)}"
          + (f" + corpus {len(items) - len(prompts)}" if fit_corpus else "")
          + f")\n           answer_tokens={answer_tokens} "
            f"(corpus {corpus_answer_tokens})  vary_templates={vary_templates}  "
            f"teacher_logits={store_teacher_logits} (top-{topk})"
            f"\n           large_layers (X)={large_layers}"
            f"\n           small_layers (Y)={small_layers}")
    for pi, p in enumerate(items):
        if vary_templates:
            system, framing = variant_for(pi, bank.system)
            question = apply_framing(framing, p["question"])
        else:
            system, question = bank.system, p["question"]
        budget = answer_tokens if p["_source"] == "bank" else corpus_answer_tokens
        ids, n_prompt = teacher_sequence(lm_large, question, budget, system)
        # Same family => same tokenizer, so row k of one model lines up with row
        # k of the other.
        sx, logits = layer_states(lm_large, ids, large_layers,
                                  want_logits=store_teacher_logits)
        sy, _ = layer_states(lm_small, ids, small_layers)
        n = ids.shape[1]
        for L in large_layers:
            xs[L].append(sx[L])
        for L in small_layers:
            ys[L].append(sy[L])
        if store_teacher_logits:
            v, k = topk_teacher(logits, topk)
            tlog.append(v)
            tidx.append(k)
        prompt_index.extend([pi] * n)
        is_answer.extend([0] * n_prompt + [1] * (n - n_prompt))
        position.extend(range(n))
        source.extend([0 if p["_source"] == "bank" else 1] * n)
        if pi % 25 == 0 or pi == len(items) - 1:
            ans = lm_large.tokenizer.decode(ids[0, n_prompt:],
                                            skip_special_tokens=True).strip()
            # flush: capture runs for half an hour, and a progress line that
            # sits in an 8 KB stdout buffer until the process exits is not
            # progress. Redirected output is block-buffered by default.
            print(f"  [{pi + 1:>4}/{len(items)}] {p['id']:22s} "
                  f"{n_prompt}p + {n - n_prompt}a tok  -> {ans[:60]!r}", flush=True)

    out_dir = states_dir(pair, bank)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "x_large.npz",
             **{f"layer_{L:02d}": np.concatenate(v) for L, v in xs.items()})
    np.savez(out_dir / "y_small.npz",
             **{f"layer_{L:02d}": np.concatenate(v) for L, v in ys.items()})
    np.savez(out_dir / "meta.npz",
             prompt_index=np.array(prompt_index, np.int32),
             is_answer=np.array(is_answer, np.int8),
             position=np.array(position, np.int32),
             source=np.array(source, np.int8))
    if store_teacher_logits:
        np.savez(out_dir / "teacher_topk.npz",
                 values=np.concatenate(tlog), indices=np.concatenate(tidx))

    n_rows, n_ans = len(prompt_index), int(sum(is_answer))
    n_prompt_rows = n_rows - n_ans
    keep = _prompt_row_budget(n_ans, n_prompt_rows, ANSWER_WEIGHT)
    realised = (n_ans * ANSWER_WEIGHT) / (n_ans * ANSWER_WEIGHT + keep) if n_ans else 0.0
    meta = {
        "pair": pair.name, "bank": bank.name, "split": split,
        "direction": "large->small",
        "prompt_ids": [p["id"] for p in items],
        "n_bank_prompts": len(prompts),
        "n_corpus_prompts": len(items) - len(prompts),
        "fit_corpus": fit_corpus, "vary_templates": vary_templates,
        "large_layers_x": large_layers, "small_layers_y": small_layers,
        "teacher_answer_tokens": answer_tokens,
        "corpus_answer_tokens": corpus_answer_tokens,
        "teacher_logits": store_teacher_logits,
        "teacher_topk": topk if store_teacher_logits else None,
        "n_prompts": len(items), "n_rows": n_rows, "n_answer_rows": n_ans,
        "n_prompt_rows": n_prompt_rows,
        "prompt_rows_kept_at_default_weight": keep,
        "answer_weight_frac_at_default_weight": realised,
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    mb = sum((out_dir / f).stat().st_size for f in
             ("x_large.npz", "y_small.npz", "meta.npz")
             + (("teacher_topk.npz",) if store_teacher_logits else ())) / 1e6
    print(f"\nCaptured {n_rows} rows ({n_ans} answer, {n_prompt_rows} prompt) from "
          f"{len(items)} items -> {out_dir} ({mb:.0f} MB)")
    print(f"  at ANSWER_WEIGHT={ANSWER_WEIGHT} the fit will keep {keep} prompt rows, "
          f"putting {realised:.1%} of the objective's weight on answer positions")
    return meta


def load_layer_pair(pair: Pair, i: int, j: int, bank: Bank):
    """(X from large layer j, Y to small layer i, prompt_index, is_answer)."""
    sdir = states_dir(pair, bank)
    if not (sdir / "meta.npz").exists():
        raise SystemExit(
            f"{sdir} has no captured states — run `python -m stitching_large_to_small.run "
            f"capture --pair {pair.name} --bank {bank.name}` first.")
    with np.load(sdir / "x_large.npz") as z:
        key = f"layer_{j:02d}"
        if key not in z:
            raise SystemExit(f"large layer {j} was not captured (have "
                             f"{sorted(int(k.split('_')[1]) for k in z)}); widen grid_j "
                             f"in config.py and re-capture.")
        X = z[key].astype(np.float32)
    with np.load(sdir / "y_small.npz") as z:
        key = f"layer_{i:02d}"
        if key not in z:
            raise SystemExit(f"small layer {i} was not captured (have "
                             f"{sorted(int(k.split('_')[1]) for k in z)}); widen grid_i "
                             f"in config.py and re-capture.")
        Y = z[key].astype(np.float32)
    m = np.load(sdir / "meta.npz")
    return X, Y, m["prompt_index"], m["is_answer"]


def load_positions(pair: Pair, bank: Bank) -> np.ndarray:
    """Per-row position within its sequence — what the suffix needs for RoPE."""
    with np.load(states_dir(pair, bank) / "meta.npz") as z:
        return z["position"]


def select_fit_rows(is_answer: np.ndarray, pid: np.ndarray, answer_weight: float,
                    policy=PROMPT_ROW_KEEP, seed: int = SEED) -> tuple[np.ndarray, dict]:
    """Row mask putting the majority of the fit objective on answer positions.

    Every answer row is kept. Prompt rows are subsampled to the budget
    `_prompt_row_budget` computes, spread evenly over prompts rather than taken
    from whichever ones come first — a contiguous slice would keep every
    position of the earliest prompts and none of the rest, which is a different
    bias, not less of one.

    Returns (mask, stats) with the realised answer-weight fraction, which the
    adapter sidecar records and `fit` asserts on. Making that number an artefact
    is the point: it was 16% for the whole published sweep and nothing on disk
    said so.
    """
    ans = is_answer == 1
    n_ans, n_pr = int(ans.sum()), int((~ans).sum())
    budget = _prompt_row_budget(n_ans, n_pr, answer_weight, policy)
    mask = ans.copy()
    if budget >= n_pr:
        mask |= ~ans
    elif budget > 0:
        # Stratify by prompt so every prompt contributes some context rows.
        rng = np.random.default_rng(seed)
        idx_prompt = np.flatnonzero(~ans)
        order = rng.permutation(len(idx_prompt))
        keys = pid[idx_prompt][order]
        rank = np.empty(len(order), np.int64)
        for p in np.unique(keys):
            sel = np.flatnonzero(keys == p)
            rank[sel] = np.arange(len(sel))
        chosen = idx_prompt[order[np.argsort(rank, kind="stable")[:budget]]]
        mask[chosen] = True
    kept_prompt = int(mask.sum() - n_ans)
    total_w = n_ans * answer_weight + kept_prompt
    return mask, {
        "n_answer_rows": n_ans, "n_prompt_rows_available": n_pr,
        "n_prompt_rows_kept": kept_prompt, "n_rows_fit": int(mask.sum()),
        "answer_weight": answer_weight,
        "answer_weight_frac": (n_ans * answer_weight / total_w) if total_w else 0.0,
        "prompt_row_policy": str(policy),
    }


def load_teacher_topk(pair: Pair, bank: Bank) -> tuple[np.ndarray, np.ndarray]:
    """(values, indices) of the stored top-K teacher logits, one row per position."""
    path = states_dir(pair, bank) / "teacher_topk.npz"
    if not path.exists():
        raise SystemExit(
            f"{path} not found — the distill training method needs the large "
            f"model's next-token distributions. Re-run capture without "
            f"--no-teacher-logits.")
    z = np.load(path)
    return z["values"], z["indices"]


if __name__ == "__main__":
    run(PAIRS[DEFAULT_PAIR], BANKS[DEFAULT_BANK])
