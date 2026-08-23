"""Prompt splits and paired hidden-state capture.

Capture runs each fit-split prompt through the *large* model to get its greedy
answer, then teacher-forces `prompt + answer` through both models and records
the hidden states at every position of both. The answer positions are flagged.

Why the answer matters (this is the whole point of the folder): at decode time
the adapter is handed the small model's state at positions holding *generated*
answer tokens. A map fit only on prompt positions has never seen those, and the
stitch produces a correct first token followed by noise. Including them, and
up-weighting them at fit time, is what makes stitched generation hold together.

Outputs (results/<pair>[/<bank>]/states/):
    x_small.npz — one key per captured small layer, (N, dim_small) float16
    y_large.npz — one key per captured large layer, (N, dim_large) float16
    meta.npz    — prompt_index, is_answer, position
    meta.json   — prompt ids, layers, token counts

Several small layers are stored per handoff depth i, not just i itself, so a
multi-tap adapter (config.tap_layers) can be fit without re-capturing. Small
states are 2048-d float16; storing the taps speculatively is cheap next to
another capture pass.
"""

from __future__ import annotations

import json

import numpy as np
import torch

from stitching_small_to_large.config import (
    BANKS, Bank, DEFAULT_BANK, DEFAULT_PAIR, DISTILL_TOPK, MIN_ANSWER_WEIGHT_FRAC,
    N_TAPS, PAIRS, PROMPT_ROW_KEEP, Pair, SEED, SPLIT_FRACS, capture_layers,
    states_dir, tap_layers,
)
from common.fit_corpus import load_corpus, overlap_with
from common.model_utils import LM, build_prompt_ids, load_lm, pick_device
from common.templates import apply_framing, variant_for


def by_id(bank: Bank) -> dict[str, dict]:
    return {p["id"]: p for p in bank.prompts}


# ── splits ────────────────────────────────────────────────────────────────────
def splits(bank: Bank) -> dict[str, list[str]]:
    """Deterministic three-way split of the prompt bank by id.

    Three ways rather than two so the sweep does not pick (i, j) on the same
    prompts it then reports. `dev` chooses; `test` is scored once at the end.

    A bank whose items carry their own `split` is taken at its word rather than
    reshuffled. `list_hard` composes each question from several underlying
    facts, so its split has to be made over *facts* — shuffling the composed
    prompts here would put a fact in a fit prompt and in a dev prompt, and the
    adapter would be scored on an answer it was fit on.
    """
    items = by_id(bank)
    if all(p.get("split") for p in items.values()):
        return {s: sorted(pid for pid, p in items.items() if p["split"] == s)
                for s in ("fit", "dev", "test")}
    ids = sorted(items)
    rng = np.random.default_rng(SEED)
    rng.shuffle(ids)
    n = len(ids)
    n_fit = int(round(n * SPLIT_FRACS["fit"]))
    n_dev = int(round(n * SPLIT_FRACS["dev"]))
    return {
        "fit": ids[:n_fit],
        "dev": ids[n_fit:n_fit + n_dev],
        "test": ids[n_fit + n_dev:],
    }


def split_prompts(bank: Bank, split: str) -> list[dict]:
    b = by_id(bank)
    s = splits(bank)
    if split == "all":
        return [b[i] for i in sorted(b)]
    if split not in s:
        raise SystemExit(f"unknown split {split!r}; use one of {sorted(s)} or 'all'")
    return [b[i] for i in s[split]]


# ── capture ───────────────────────────────────────────────────────────────────
@torch.no_grad()
def teacher_sequence(lm_large: LM, question: str, answer_tokens: int,
                     system: str | None = None) -> tuple[torch.Tensor, int]:
    """(prompt + large model's own greedy answer) as one id tensor, plus the
    prompt length. The answer is the trajectory we want the stitch to reproduce,
    so it is also the trajectory the adapter should be fit on."""
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
    distribution the stitched path is asked to reproduce. They come from the
    same forward pass that produces the states, so storing them costs one
    top-K sort per prompt rather than a second pass over the model.
    """
    out = lm.model(ids, output_hidden_states=True)
    states = {L: out.hidden_states[L][0].float().cpu().numpy().astype(np.float16)
              for L in layers}
    return (states, out.logits[0].float()) if want_logits else (states, None)


def topk_teacher(logits: torch.Tensor, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Top-k teacher logits and their token ids, as (rows, k) arrays.

    Full-vocab targets are not storable at this scale — 128256 floats per row
    against tens of thousands of rows is gigabytes — and are not needed: the
    tail of a next-token distribution the large model would greedily decode
    from carries almost no mass, and KL over the top-k support renormalised is
    within noise of the full-vocab quantity for k in the hundreds. k is
    recorded in the sidecar so the approximation is never invisible.
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

    The captured `factual` set fails this badly at 17%: 258 answer rows at
    weight 4 against 5180 prompt rows. Solving for n_keep and leaving a 5%
    margin is what this returns.

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
        vary_templates: bool = True, store_teacher_logits: bool = True,
        topk: int = DISTILL_TOPK,
        small_layers: list[int] | None = None,
        large_layers: list[int] | None = None) -> dict:
    """Capture paired states (and teacher distributions) for the adapter fit.

    Three things here shape the fit distribution, and all three were wrong
    before:

    `vary_templates` paraphrases the system prompt and the framing around each
        question, so prompt positions stop being one boilerplate prefix repeated
        once per prompt. See common/templates.py for why that mattered.
    `fit_corpus` mixes in open-ended generic instructions whose continuations
        run 40-80 tokens. Bank answers are 3-10 tokens, so the bank alone cannot
        produce answer rows in the tens of thousands at any reasonable prompt
        count. The corpus is never an eval bank, and the overlap is checked.
    `store_teacher_logits` records the large model's own next-token distribution
        (top-K) at every position. That is the target the distill training
        method fits, and it has to be captured here because recomputing it
        inside the training loop would mean a full large-model forward per step.
    """
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
    assert lm_small.n_layers == pair.n_layers_small and lm_large.n_layers == pair.n_layers_large, \
        "loaded models do not match the geometry in config.PAIRS"

    xs = {L: [] for L in small_layers}
    ys = {L: [] for L in large_layers}
    prompt_index, is_answer, position, source = [], [], [], []
    tlog, tidx = [], []

    print(f"[capture] {pair.name} bank={bank.name} split={split} "
          f"items={len(items)} (bank {len(prompts)}"
          + (f" + corpus {len(items) - len(prompts)}" if fit_corpus else "")
          + f")\n           answer_tokens={answer_tokens}  "
            f"vary_templates={vary_templates}  teacher_logits={store_teacher_logits}"
            f" (top-{topk})"
            f"\n           small_layers={small_layers}"
            f"\n           large_layers={large_layers}")
    for pi, p in enumerate(items):
        if vary_templates:
            system, framing = variant_for(pi, bank.system)
            question = apply_framing(framing, p["question"])
        else:
            system, question = bank.system, p["question"]
        ids, n_prompt = teacher_sequence(lm_large, question, answer_tokens, system)
        # Same family => same tokenizer, so the small model sees the identical
        # ids and its row k lines up with the large model's row k.
        sx, _ = layer_states(lm_small, ids, small_layers)
        sy, logits = layer_states(lm_large, ids, large_layers,
                                  want_logits=store_teacher_logits)
        n = ids.shape[1]
        for L in small_layers:
            xs[L].append(sx[L])
        for L in large_layers:
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
            print(f"  [{pi + 1:>4}/{len(items)}] {p['id']:18s} "
                  f"{n_prompt}p + {n - n_prompt}a tok  -> {ans[:60]!r}")

    out_dir = states_dir(pair, bank)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "x_small.npz",
             **{f"layer_{L:02d}": np.concatenate(v) for L, v in xs.items()})
    np.savez(out_dir / "y_large.npz",
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
    from stitching_small_to_large.config import ANSWER_WEIGHT
    keep = _prompt_row_budget(n_ans, n_prompt_rows, ANSWER_WEIGHT)
    realised = (n_ans * ANSWER_WEIGHT) / (n_ans * ANSWER_WEIGHT + keep) if n_ans else 0.0
    meta = {
        "pair": pair.name, "bank": bank.name, "split": split,
        "prompt_ids": [p["id"] for p in items],
        "n_bank_prompts": len(prompts),
        "n_corpus_prompts": len(items) - len(prompts),
        "fit_corpus": fit_corpus, "vary_templates": vary_templates,
        "small_layers": small_layers, "large_layers": large_layers,
        "teacher_answer_tokens": answer_tokens,
        "teacher_logits": store_teacher_logits, "teacher_topk": topk if store_teacher_logits else None,
        "n_prompts": len(items), "n_rows": n_rows, "n_answer_rows": n_ans,
        "n_prompt_rows": n_prompt_rows,
        "prompt_rows_kept_at_default_weight": keep,
        "answer_weight_frac_at_default_weight": realised,
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    mb = sum((out_dir / f).stat().st_size for f in
             ("x_small.npz", "y_large.npz", "meta.npz")
             + (("teacher_topk.npz",) if store_teacher_logits else ())) / 1e6
    print(f"\nCaptured {n_rows} rows ({n_ans} answer, {n_prompt_rows} prompt) from "
          f"{len(items)} items -> {out_dir} ({mb:.0f} MB)")
    print(f"  at ANSWER_WEIGHT={ANSWER_WEIGHT} the fit will keep {keep} prompt rows, "
          f"putting {realised:.1%} of the objective's weight on answer positions")
    return meta


def load_layer_pair(pair: Pair, i: int, j: int, bank: Bank, n_taps: int = N_TAPS):
    """Load the layers a given (i, j) needs, plus row metadata.

    X is the small model's residual at every tap layer, concatenated along the
    feature axis (shallowest first, layer i last), so a multi-tap adapter is just
    the same affine/MLP fit on a wider input. Returns
    (X, Y, prompt_index, is_answer, taps).
    """
    taps = tap_layers(i, n_taps)
    sdir = states_dir(pair, bank)
    if not (sdir / "meta.npz").exists():
        raise SystemExit(f"{sdir} has no captured states — run `python -m stitching_small_to_large.run capture "
                         f"--pair {pair.name} --bank {bank.name}` first.")
    with np.load(sdir / "x_small.npz") as z:
        have = sorted(int(k.split("_")[1]) for k in z)
        missing = [L for L in taps if f"layer_{L:02d}" not in z]
        if missing:
            raise SystemExit(
                f"small layers {missing} were not captured (have {have}); "
                f"--n-taps {n_taps} at i={i} needs {list(taps)}. Raise CAPTURE_MAX_TAPS "
                f"or widen grid_i in config.py and re-capture.")
        X = np.concatenate([z[f"layer_{L:02d}"].astype(np.float32) for L in taps], axis=1)
    with np.load(sdir / "y_large.npz") as z:
        key = f"layer_{j:02d}"
        if key not in z:
            raise SystemExit(f"large layer {j} was not captured (have "
                             f"{sorted(int(k.split('_')[1]) for k in z)}); widen grid_j in "
                             f"config.py and re-capture.")
        Y = z[key].astype(np.float32)
    m = np.load(sdir / "meta.npz")
    return X, Y, m["prompt_index"], m["is_answer"], taps


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
    is the point: it was 17% for the whole `factual` campaign and nothing on
    disk said so.
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
        # Stable interleave by prompt id keeps the per-prompt counts even.
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
            f"{path} not found — the distill training method needs the teacher's "
            f"next-token distributions. Re-run capture without --no-teacher-logits.")
    z = np.load(path)
    return z["values"], z["indices"]


if __name__ == "__main__":
    run(PAIRS[DEFAULT_PAIR], BANKS[DEFAULT_BANK])
