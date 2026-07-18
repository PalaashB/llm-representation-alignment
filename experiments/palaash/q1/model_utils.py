"""Model loading, chat formatting, greedy generation, hidden-state extraction.

For token-aligned pairs (align="token") the two models come from the same
family and share the *same* tokenizer (asserted in load_pair), so for a given
prompt they see the identical token sequence — that is what lets us align
hidden states position-by-position. Prompt-aligned pairs (align="prompt") have
different tokenizers, so the identity check is skipped and alignment happens
per prompt instead (see extract_states).

This is the only module in the package that imports torch/transformers.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from q1.config import SYSTEM_PROMPT, MAX_NEW_TOKENS, ModelPair


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@dataclass
class LM:
    """A loaded language model plus its tokenizer and a friendly tag."""
    tag: str
    tokenizer: AutoTokenizer
    model: AutoModelForCausalLM
    device: str

    @property
    def n_layers(self) -> int:
        return self.model.config.num_hidden_layers

    @property
    def hidden_size(self) -> int:
        return self.model.config.hidden_size


def load_lm(model_id: str, tag: str, device: str | None = None) -> LM:
    device = device or pick_device()
    print(f"[load] {tag}: {model_id} -> {device}")
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map={"": device},
    )
    model.eval()
    return LM(tag=tag, tokenizer=tok, model=model, device=device)


def load_pair(pair: ModelPair, device: str | None = None) -> tuple[LM, LM]:
    """Load both models of a pair, asserting the expected geometry and — for
    token-aligned pairs only — the shared-tokenizer assumption that
    position-by-position alignment rests on."""
    lm_s = load_lm(pair.small_id, pair.small_tag, device)
    lm_l = load_lm(pair.large_id, pair.large_tag, device)
    assert lm_s.n_layers == pair.n_layers_small and lm_s.hidden_size == pair.dim_small, \
        f"{pair.small_id}: got {lm_s.n_layers} layers / {lm_s.hidden_size}-d, config says otherwise"
    assert lm_l.n_layers == pair.n_layers_large and lm_l.hidden_size == pair.dim_large, \
        f"{pair.large_id}: got {lm_l.n_layers} layers / {lm_l.hidden_size}-d, config says otherwise"
    if pair.align == "token":
        probe = "What is the capital city of Bhutan?"
        ids_s = build_prompt_ids(lm_s.tokenizer, probe, "cpu")
        ids_l = build_prompt_ids(lm_l.tokenizer, probe, "cpu")
        assert ids_s.shape == ids_l.shape and bool((ids_s == ids_l).all()), \
            f"{pair.name}: the two models do not share a tokenizer — hidden states cannot be aligned"
    return lm_s, lm_l


def build_prompt_ids(tok: AutoTokenizer, question: str, device: str) -> torch.Tensor:
    """Return input_ids (1, seq) for the chat-formatted prompt, ready to generate."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    enc = tok.apply_chat_template(
        messages,
        add_generation_prompt=True,   # append the assistant header so the model answers
        return_tensors="pt",
        return_dict=True,             # transformers 5.x returns a BatchEncoding
    )
    return enc["input_ids"].to(device)


@torch.no_grad()
def generate_answer(lm: LM, question: str, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
    ids = build_prompt_ids(lm.tokenizer, question, lm.device)
    out = lm.model.generate(
        ids,
        attention_mask=torch.ones_like(ids),   # batch=1, no padding -> all ones
        max_new_tokens=max_new_tokens,
        do_sample=False,                       # greedy -> deterministic
        pad_token_id=lm.tokenizer.eos_token_id,
    )
    gen = out[0, ids.shape[1]:]                 # only the newly generated tokens
    return lm.tokenizer.decode(gen, skip_special_tokens=True).strip()


@torch.no_grad()
def hidden_states_for_prompt(lm: LM, question: str, last_k: int | None = None):
    """
    Run one forward pass over the chat-formatted prompt and return:
      states : list of (seq_or_k, hidden) float32 numpy arrays, one per hidden
               layer (index 0 = embeddings, ..., index n_layers = final block)
      n_used : number of token positions kept
      last_local_idx : index (within the kept rows) of the final prompt token —
               the position that produces the first answer token.

    If last_k is given, only the final `last_k` token positions are kept (keeps
    storage bounded and focuses on the answer-relevant context); otherwise all
    prompt positions are kept.
    """
    ids = build_prompt_ids(lm.tokenizer, question, lm.device)
    out = lm.model(ids, output_hidden_states=True)
    hs = out.hidden_states                       # tuple len = n_layers + 1
    seq = ids.shape[1]

    if last_k is not None and seq > last_k:
        sl = slice(seq - last_k, seq)
        n_used = last_k
    else:
        sl = slice(0, seq)
        n_used = seq

    states = [h[0, sl].float().cpu().numpy() for h in hs]
    last_local_idx = n_used - 1                   # final prompt token within the slice
    return states, n_used, last_local_idx
