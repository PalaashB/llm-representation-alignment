"""Step 2 — extract paired hidden states from both models of a pair for the
prompts selected in step 1.

Token-aligned pairs (align="token"): the two models share a tokenizer, so a
given prompt yields the *same* token sequence in both, and hidden-state row k
of the small model's output corresponds to the same token as row k of the
large model's output. For each prompt we run a single forward pass per model
with `output_hidden_states=True` and keep the final `LAST_K` token positions
(bounds storage and concentrates on the answer-relevant context). The very
last position — the token that produces the model's first answer token — is
flagged so the analysis can treat it on its own.

Prompt-aligned pairs (align="prompt", cross-family): the tokenizers differ, so
token positions cannot be paired. Both models answer the same question, so we
keep exactly ONE row per prompt per model — the final answer-generating token
position. Every kept row is a "last" token; the row pairing is by prompt.
NOTE: this yields far fewer rows (one per prompt), so the DM fit on the small
divergent set is statistically underpowered — see README.

The saved layout is identical in either mode:

Outputs (in results/<pair>/states/):
    x_small.npz — keys layer_00 .. layer_NN, each (N_rows, dim_small) float16
    y_large.npz — keys layer_00 .. layer_MM, each (N_rows, dim_large) float16
    meta.npz    — prompt_index, set_label (0=divergent,1=control), is_last
    meta.json   — ordered prompt ids + which set each belongs to
"""

import json

import numpy as np

from q1.config import ModelPair, PAIRS, DEFAULT_PAIR, LAST_K, results_dir, states_dir
from q1.model_utils import LM, load_pair, hidden_states_for_prompt
from q1.prompts import PROMPTS

BY_ID = {p["id"]: p for p in PROMPTS}


def run(pair: ModelPair, lm_small: LM | None = None, lm_large: LM | None = None) -> None:
    sel_path = results_dir(pair) / "selection.json"
    if not sel_path.exists():
        raise SystemExit(f"{sel_path} not found — run the select step first.")
    with open(sel_path) as f:
        sel = json.load(f)

    # Ordered list of (prompt_id, set_label). set_label: 0=divergent, 1=control.
    selected = ([(pid, 0) for pid in sel["divergent_small_wrong_large_right"]] +
                [(pid, 1) for pid in sel["control_both_right"]])
    if not selected:
        raise SystemExit("No prompts selected in step 1 — nothing to extract.")

    if lm_small is None or lm_large is None:
        lm_small, lm_large = load_pair(pair)
    out_dir = states_dir(pair)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_small, n_large = pair.n_layers_small, pair.n_layers_large
    x_layers = [[] for _ in range(n_small + 1)]
    y_layers = [[] for _ in range(n_large + 1)]
    prompt_index, set_label, is_last = [], [], []

    last_k = LAST_K if pair.align == "token" else 1
    for pi, (pid, label) in enumerate(selected):
        q = BY_ID[pid]["question"]
        xs, n_s, last_s = hidden_states_for_prompt(lm_small, q, last_k=last_k)
        ys, n_l, _ = hidden_states_for_prompt(lm_large, q, last_k=last_k)
        if pair.align == "token":
            assert n_s == n_l, f"token-count mismatch on {pid}: {n_s} vs {n_l}"
        else:
            # prompt-aligned: last_k=1 keeps only the final answer-generating
            # position, so both models contribute exactly one row per prompt.
            assert n_s == 1 and n_l == 1

        for L, arr in enumerate(xs):
            x_layers[L].append(arr.astype(np.float16))
        for L, arr in enumerate(ys):
            y_layers[L].append(arr.astype(np.float16))

        prompt_index.extend([pi] * n_s)
        set_label.extend([label] * n_s)
        flags = [0] * n_s
        flags[last_s] = 1
        is_last.extend(flags)
        print(f"[{pi+1:>2}/{len(selected)}] {pid:14s} set={label} tokens={n_s}")

    x_stacked = {f"layer_{L:02d}": np.concatenate(x_layers[L], axis=0) for L in range(n_small + 1)}
    y_stacked = {f"layer_{L:02d}": np.concatenate(y_layers[L], axis=0) for L in range(n_large + 1)}

    np.savez(out_dir / "x_small.npz", **x_stacked)
    np.savez(out_dir / "y_large.npz", **y_stacked)
    np.savez(
        out_dir / "meta.npz",
        prompt_index=np.array(prompt_index, dtype=np.int32),
        set_label=np.array(set_label, dtype=np.int8),
        is_last=np.array(is_last, dtype=np.int8),
    )
    with open(out_dir / "meta.json", "w") as f:
        json.dump({
            "pair": pair.name,
            "align": pair.align,
            "selected": [{"prompt_id": pid, "set_label": lbl} for pid, lbl in selected],
            "last_k": last_k,
            "n_layers_small": n_small, "n_layers_large": n_large,
            "n_tokens_total": len(prompt_index),
        }, f, indent=2)

    n_tok = len(prompt_index)
    print("\n" + "=" * 60)
    print(f"Extracted {n_tok} {'token' if pair.align == 'token' else 'final-token (prompt-aligned)'} "
          f"rows from {len(selected)} prompts "
          f"({sum(1 for _, l in selected if l == 0)} divergent, "
          f"{sum(1 for _, l in selected if l == 1)} control).")
    if pair.align == "prompt":
        print(f"NOTE: prompt-level alignment = 1 row/prompt. With {n_tok} rows vs "
              f"{pair.dim_small}-d inputs the DM fit is underdetermined; treat "
              f"per-set R² as indicative (see README caveat).")
    print(f"  {pair.small_tag}: {n_small + 1} layers x ({n_tok}, {pair.dim_small})")
    print(f"  {pair.large_tag}: {n_large + 1} layers x ({n_tok}, {pair.dim_large})")
    print(f"Wrote {out_dir}/x_small.npz, y_large.npz, meta.npz, meta.json")


if __name__ == "__main__":
    run(PAIRS[DEFAULT_PAIR])
