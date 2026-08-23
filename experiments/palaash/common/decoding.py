"""Sliceable decoder plumbing shared by both stitching packages.

HF's `forward` always runs the full stack, so a stitch that skips blocks has to
drive the layer loop itself. `Stack.run` is a transcription of
`LlamaModel.forward` in transformers 5.x (Qwen2's is structurally identical),
restricted to a layer slice and able to tap the residual at chosen depths.

This lives in `common` because it is direction-agnostic: small->large and
large->small both need the same sliced loop, the same KV-cached baselines, and
the same greedy decode. One verified copy is the point — `check_full_stack`
asserts the transcription reproduces `lm.model(ids).logits`, and duplicating it
per package would mean duplicating that risk.

Baselines are timed through this same loop as the stitched paths, so a latency
comparison is never a comparison of two different decoding harnesses.
"""

from __future__ import annotations

from time import perf_counter

import torch
from transformers import DynamicCache

from common.model_utils import LM, build_prompt_ids, generate_answer

WARMUP_TOKENS = 3   # throwaway decode steps before timing, to page in weights


def sync(device: str) -> None:
    """Accelerator queues are asynchronous; timing without this measures nothing."""
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def eos_ids(lm: LM) -> set[int]:
    """Every id that ends a turn. Llama-Instruct stops on <|eot_id|>, not </s>,
    and `generation_config.eos_token_id` may be a list — matching what
    `model.generate` stops on is what keeps the baselines honest."""
    out = set()
    for src in (getattr(lm.model.generation_config, "eos_token_id", None),
                lm.tokenizer.eos_token_id):
        if src is None:
            continue
        out.update(src if isinstance(src, (list, tuple)) else [src])
    return out


def decoder_stack(lm: LM) -> torch.nn.Module:
    """The module owning the decoder blocks, whatever the wrapper nesting is —
    returned rather than just the block list because the early-exit path also
    needs its siblings (embed_tokens, rotary_emb, norm)."""
    for path in (("model",), ("model", "language_model"), ("transformer",), ()):
        node = lm.model
        try:
            for attr in path:
                node = getattr(node, attr)
        except AttributeError:
            continue
        for name in ("layers", "h"):
            if isinstance(getattr(node, name, None), torch.nn.ModuleList):
                return node
    raise RuntimeError(f"could not locate the decoder stack on {type(lm.model).__name__}")


# ── one model's decoder stack, sliceable ──────────────────────────────────────
class Stack:
    def __init__(self, lm: LM):
        self.base = decoder_stack(lm)
        self.layers = self.base.layers
        self.lm_head = lm.model.get_output_embeddings()
        self.config = lm.model.config
        self.device = lm.device
        self.dtype = next(self.layers.parameters()).dtype

    @property
    def n_layers(self) -> int:
        return len(self.layers)

    def new_cache(self) -> DynamicCache:
        return DynamicCache(config=self.config)

    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        return self.base.embed_tokens(ids)

    def _mask(self, hidden: torch.Tensor, past_len: int):
        """Attention mask for an unpadded batch of 1. SDPA and the flash kernels
        derive causality from the query length themselves when handed None,
        which is also the fast kernel path; `eager` needs it spelled out."""
        impl = self.config._attn_implementation
        if impl == "sdpa" or impl.startswith("flash"):
            return None
        if impl != "eager":
            raise RuntimeError(f"attn implementation {impl!r} unsupported by the "
                               f"early-exit stack; load with sdpa or eager.")
        q = hidden.shape[1]
        if q == 1:
            return None
        rows = torch.arange(q, device=hidden.device) + past_len
        cols = torch.arange(past_len + q, device=hidden.device)
        mask = torch.zeros((q, past_len + q), dtype=hidden.dtype, device=hidden.device)
        mask.masked_fill_(cols[None, :] > rows[:, None], torch.finfo(hidden.dtype).min)
        return mask[None, None]

    def run(self, hidden: torch.Tensor, position_ids: torch.Tensor,
            start: int, stop: int, cache: DynamicCache | None,
            taps: dict[int, torch.Tensor] | None = None) -> torch.Tensor:
        """Push `hidden` through blocks [start, stop). The cache is indexed by
        each block's own `layer_idx`, so a suffix run fills slots start..stop-1
        and leaves the skipped ones empty — nothing reads them, since positions
        are passed explicitly rather than inferred from the cache.

        If `taps` is given, the residual at the *input* to each block index it
        holds as a key is stored there. That matches HF's `hidden_states[L]`
        convention, which is what state capture recorded, so a multi-tap adapter
        is handed the same quantities it was fit on. Layer `stop` needs no tap:
        its input is the returned value.
        """
        pos_emb = self.base.rotary_emb(hidden, position_ids)
        mask = self._mask(hidden, int(position_ids[0, 0]))
        for k, layer in enumerate(self.layers[start:stop]):
            if taps is not None and (start + k) in taps:
                taps[start + k] = hidden
            hidden = layer(hidden, attention_mask=mask, position_ids=position_ids,
                           position_embeddings=pos_emb, past_key_values=cache,
                           use_cache=cache is not None)
        return hidden

    def head(self, hidden: torch.Tensor) -> torch.Tensor:
        """Final norm + unembedding of the last position only -> (1, vocab)."""
        return self.lm_head(self.base.norm(hidden[:, -1:]))[:, -1]

    def n_params(self, start: int, stop: int, head: bool = True) -> int:
        """Weights multiplied against every decoded token on this path. The
        embedding table is a gather, not a matmul, so it is excluded; the
        (possibly tied) lm_head is not."""
        mods = list(self.layers[start:stop]) + [self.base.norm]
        if head:
            mods.append(self.lm_head)
        return sum(p.numel() for m in mods for p in m.parameters())


# ── decoding paths ────────────────────────────────────────────────────────────
class FullRunner:
    """Baseline: one model, all layers, KV-cached, same loop as the stitch."""

    def __init__(self, lm: LM, label: str):
        self.stack = Stack(lm)
        self.label = label
        self.device = lm.device
        self.cache = None

    def prefill(self, ids: torch.Tensor) -> torch.Tensor:
        self.cache = self.stack.new_cache()
        pos = torch.arange(ids.shape[1], device=ids.device)[None]
        h = self.stack.run(self.stack.embed(ids), pos, 0, self.stack.n_layers, self.cache)
        return self.stack.head(h)

    def step(self, token_id: int, position: int) -> torch.Tensor:
        ids = torch.tensor([[token_id]], device=self.device)
        pos = torch.tensor([[position]], device=self.device)
        h = self.stack.run(self.stack.embed(ids), pos, 0, self.stack.n_layers, self.cache)
        return self.stack.head(h)

    def active_params(self) -> int:
        return self.stack.n_params(0, self.stack.n_layers)


@torch.no_grad()
def greedy(runner, ids: torch.Tensor, max_new_tokens: int,
           stop_ids: set[int] | None) -> dict:
    """Greedy decode, timing prefill and each decode step separately.

    `stop_ids=None` forces the full budget, which is how latency is collected:
    when answers are short, an EOS-stopping run estimates ms/token from two or
    three samples and times each path over a different amount of work."""
    device = runner.device
    sync(device)
    t0 = perf_counter()
    logits = runner.prefill(ids)
    sync(device)
    prefill_ms = (perf_counter() - t0) * 1e3

    out, step_ms, pos = [], [], ids.shape[1]
    while len(out) < max_new_tokens:
        nxt = int(logits.argmax(-1))
        out.append(nxt)
        if (stop_ids and nxt in stop_ids) or len(out) == max_new_tokens:
            break
        sync(device)
        t0 = perf_counter()
        logits = runner.step(nxt, pos)
        sync(device)
        step_ms.append((perf_counter() - t0) * 1e3)
        pos += 1

    step_ms.sort()
    return {"token_ids": out, "prefill_ms": prefill_ms,
            # Median, not mean: a single GC pause or scheduler preemption in a
            # 32-step run moves the mean by more than the effects being measured.
            "decode_ms_per_token": step_ms[len(step_ms) // 2] if step_ms else float("nan"),
            "decode_ms_per_token_mean":
                (sum(step_ms) / len(step_ms)) if step_ms else float("nan"),
            "total_ms": prefill_ms + sum(step_ms), "n_generated": len(out)}


@torch.no_grad()
def warmup(runner, ids: torch.Tensor) -> None:
    """The first call on an accelerator pays for kernel setup and weight paging;
    without this whichever path is benchmarked first looks slower than it is."""
    logits = runner.prefill(ids)
    pos = ids.shape[1]
    for _ in range(WARMUP_TOKENS):
        logits = runner.step(int(logits.argmax(-1)), pos)
        pos += 1
    sync(runner.device)


# ── correctness checks (cheap; they protect the latency claim) ────────────────
@torch.no_grad()
def check_full_stack(lm: LM, ids: torch.Tensor, tag: str) -> dict:
    """The hand-rolled layer loop, run over *all* layers, must reproduce HF's
    own forward. This is what licenses running it over a slice — and what makes
    the baseline timings a fair comparison rather than a different harness.
    Both stitching packages gate their reports on this."""
    stack = Stack(lm)
    ref = lm.model(ids).logits[0, -1].float()
    pos = torch.arange(ids.shape[1], device=ids.device)[None]
    got = stack.head(stack.run(stack.embed(ids), pos, 0, stack.n_layers,
                               stack.new_cache()))[0].float()
    rel = float((ref - got).norm() / (ref.norm() + 1e-12))
    return {"check": f"full_stack_{tag}", "rel_l2": rel,
            "max_abs_logit_diff": float((ref - got).abs().max()),
            "argmax_match": int(ref.argmax()) == int(got.argmax()),
            "passed": rel < 5e-3 and int(ref.argmax()) == int(got.argmax())}


@torch.no_grad()
def check_baseline_matches_hf(lm: LM, question: str, tag: str, n: int = 8) -> dict:
    """Both baselines are timed through this module's decode loop, which is only
    fair if that loop *is* `model.generate`, stop condition included."""
    ids = build_prompt_ids(lm.tokenizer, question, lm.device)
    got = greedy(FullRunner(lm, tag), ids, n, eos_ids(lm))["token_ids"]
    mine = lm.tokenizer.decode(got, skip_special_tokens=True).strip()
    ref = generate_answer(lm, question, n)
    return {"check": f"baseline_matches_hf_generate_{tag}", "loop": mine,
            "hf_generate": ref, "passed": mine == ref}
