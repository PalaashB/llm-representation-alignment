"""Prompt splits and paired hidden-state capture (large -> small).

Capture is direction-agnostic in mechanism and directional in meaning: each
fit-split prompt is run through the *large* model to get its greedy answer, then
`prompt + answer` is teacher-forced through both models and the hidden states of
both are recorded at every position, with answer positions flagged.

The direction shows up at fit time, not here: the large model's layer-j states
are the adapter's input `X` and the small model's layer-i states are its target
`Y`, which is the reverse of the sibling package.

Why the teacher answer is the *large* model's: it is the trajectory we want the
stitched path to reproduce. Fitting on the small model's own answer would teach
the adapter to reconstruct the very output we are trying to improve on.

Outputs (results/<pair>/<bank>/states/):
    x_large.npz — one key per captured large layer, (N, dim_large) float16
    y_small.npz — one key per captured small layer, (N, dim_small) float16
    meta.npz    — prompt_index, is_answer, position
    meta.json   — prompt ids, layers, token counts
"""

from __future__ import annotations

import json

import numpy as np
import torch

from common.model_utils import LM, build_prompt_ids, load_lm, pick_device
from stitching_large_to_small.config import (
    BANKS, DEFAULT_BANK, DEFAULT_PAIR, PAIRS, SEED, SPLIT_FRACS, Bank, Pair,
    capture_layers, states_dir,
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
def layer_states(lm: LM, ids: torch.Tensor, layers: list[int]) -> dict[int, np.ndarray]:
    out = lm.model(ids, output_hidden_states=True)
    return {L: out.hidden_states[L][0].float().cpu().numpy().astype(np.float16)
            for L in layers}


def run(pair: Pair, bank: Bank, split: str = "fit", max_prompts: int | None = None,
        answer_tokens: int | None = None,
        lm_small: LM | None = None, lm_large: LM | None = None) -> dict:
    answer_tokens = answer_tokens or bank.teacher_answer_tokens
    prompts = split_prompts(bank, split)
    if max_prompts:
        prompts = prompts[:max_prompts]
    if split != "fit":
        print(f"WARNING: capturing on split={split!r}. The adapter is fit on whatever "
              f"is captured, so any bench on 'dev'/'test' afterwards scores on prompts "
              f"the map was trained on.")
    small_layers, large_layers = capture_layers(pair)

    if lm_small is None or lm_large is None:
        device = pick_device()
        lm_small = lm_small or load_lm(pair.small_id, pair.small_tag, device)
        lm_large = lm_large or load_lm(pair.large_id, pair.large_tag, device)
    assert lm_small.n_layers == pair.n_layers_small and \
        lm_large.n_layers == pair.n_layers_large, \
        "loaded models do not match the geometry in config.PAIRS"

    xs = {L: [] for L in large_layers}     # adapter input  (large)
    ys = {L: [] for L in small_layers}     # adapter target (small)
    prompt_index, is_answer, position = [], [], []

    print(f"[capture] {pair.name} bank={bank.name} split={split} prompts={len(prompts)} "
          f"answer_tokens={answer_tokens}\n           large_layers (X)={large_layers}"
          f"\n           small_layers (Y)={small_layers}")
    for pi, p in enumerate(prompts):
        ids, n_prompt = teacher_sequence(lm_large, p["question"], answer_tokens,
                                         bank.system)
        # Same family => same tokenizer, so row k of one model lines up with row
        # k of the other.
        sx = layer_states(lm_large, ids, large_layers)
        sy = layer_states(lm_small, ids, small_layers)
        n = ids.shape[1]
        for L in large_layers:
            xs[L].append(sx[L])
        for L in small_layers:
            ys[L].append(sy[L])
        prompt_index.extend([pi] * n)
        is_answer.extend([0] * n_prompt + [1] * (n - n_prompt))
        position.extend(range(n))
        ans = lm_large.tokenizer.decode(ids[0, n_prompt:], skip_special_tokens=True).strip()
        print(f"  [{pi + 1:>3}/{len(prompts)}] {p['id']:22s} "
              f"{n_prompt} prompt + {n - n_prompt} answer tok  -> {ans!r}")

    out_dir = states_dir(pair, bank)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "x_large.npz",
             **{f"layer_{L:02d}": np.concatenate(v) for L, v in xs.items()})
    np.savez(out_dir / "y_small.npz",
             **{f"layer_{L:02d}": np.concatenate(v) for L, v in ys.items()})
    np.savez(out_dir / "meta.npz",
             prompt_index=np.array(prompt_index, np.int32),
             is_answer=np.array(is_answer, np.int8),
             position=np.array(position, np.int32))

    n_rows, n_ans = len(prompt_index), int(sum(is_answer))
    meta = {
        "pair": pair.name, "bank": bank.name, "split": split,
        "direction": "large->small",
        "prompt_ids": [p["id"] for p in prompts],
        "large_layers_x": large_layers, "small_layers_y": small_layers,
        "teacher_answer_tokens": answer_tokens,
        "n_prompts": len(prompts), "n_rows": n_rows, "n_answer_rows": n_ans,
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    mb = sum((out_dir / f).stat().st_size
             for f in ("x_large.npz", "y_small.npz", "meta.npz")) / 1e6
    print(f"\nCaptured {n_rows} rows ({n_ans} answer) from {len(prompts)} prompts "
          f"-> {out_dir} ({mb:.0f} MB)")
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


if __name__ == "__main__":
    run(PAIRS[DEFAULT_PAIR], BANKS[DEFAULT_BANK])
