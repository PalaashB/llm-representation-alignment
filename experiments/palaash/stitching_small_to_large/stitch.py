"""Early-exit stitched decoding, and the baselines it is timed against.

    prompt -> small embed + small blocks 0..i-1 -> adapter -> large blocks
    j..end -> large norm -> large lm_head -> token

Large blocks 0..j-1 and small blocks i..end never run. Both models keep a KV
cache over the layers they do run, so decoding is O(n), not O(n^2).

HF's `forward` always runs the full stack, so the layer loop is reimplemented
here (`Stack.run`) as a transcription of `LlamaModel.forward` in transformers
5.x — Qwen2's is structurally identical. `check_full_stack` asserts that
running the whole slice through it reproduces `lm.model(ids).logits`, so the
transcription is verified rather than assumed, and the baselines are timed
through the same loop as the stitch so a latency comparison is not a comparison
of two decoding harnesses.

The attention sink: position 0 carries a residual ~30x the norm of any other
position and the adapter cannot reproduce it; overwriting it destroys attention
globally. The fix is exact and cheap — attention is causal, so running only
tokens 0..N-1 through the skipped large blocks yields exactly the hidden states
a full-sequence run would give those positions. One N=1 token forward per
prompt, spliced over the adapter's output (`check_prefix_exact` verifies it).
"""

from __future__ import annotations

import torch

from stitching_small_to_large.adapter import TorchAdapter
from stitching_small_to_large.config import PRESERVE_PREFIX
from common.decoding import (            # the sliced layer loop and the baselines
    FullRunner, Stack, check_baseline_matches_hf, check_full_stack, greedy,
)
from common.model_utils import LM, build_prompt_ids


class StitchRunner:
    """Small prefix -> adapter -> large suffix, KV-cached on both sides.

    Two prefill strategies; the decode step is identical, so both get the same
    per-token speedup and they differ only in what the large suffix blocks
    attend *back* to.

    `exit`  the prompt goes through the stitch too, so large blocks 0..j-1 run
            over the sink position only. Maximum saving — prefill included —
            but every prompt KV the suffix reads is the adapter's reconstruction.
    `warm`  the large model prefills the prompt itself, so the prompt KVs are
            exactly its own and the adapter only has to supply the residual at
            generated positions. Prefill is *not* saved (it costs a full large
            prefill plus the small one); decode is saved in full. Worth it
            whenever output length dominates, and it needs no sink handling —
            position 0's KV comes from the large model's own forward.
    """

    MODES = ("exit", "warm")

    def __init__(self, lm_small: LM, lm_large: LM, adapter: TorchAdapter,
                 preserve_prefix: int = PRESERVE_PREFIX, mode: str = "exit"):
        if mode not in self.MODES:
            raise SystemExit(f"mode must be one of {self.MODES}")
        self.small, self.large = Stack(lm_small), Stack(lm_large)
        self.adapter = adapter
        self.i, self.j = adapter.i, adapter.j
        self.taps = tuple(adapter.taps)
        if any(L > self.i for L in self.taps):
            raise SystemExit(f"adapter taps {self.taps} include a layer past the handoff "
                             f"depth i={self.i}; those blocks never run on this path.")
        self.preserve_prefix = preserve_prefix
        self.mode = mode
        tag = "" if self.taps == (self.i,) else f"-t{len(self.taps)}"
        self.label = f"stitch-{mode}{tag}-{adapter.kind}-L{self.i}->L{self.j}"
        self.device = lm_large.device
        self.small_cache = self.large_cache = None

    def small_features(self, ids: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """Run small blocks 0..i-1 and return the adapter's input: the residual
        at each tap depth, concatenated shallowest-first. Every tap below i is a
        value this run passes through anyway, so extra taps cost only the memory
        to hold them."""
        inner = {L: None for L in self.taps if L < self.i}
        h = self.small.run(self.small.embed(ids), pos, 0, self.i, self.small_cache,
                           taps=inner or None)
        if not inner:
            return h
        return torch.cat([inner[L] if L < self.i else h for L in self.taps], dim=-1)

    def prefill(self, ids: torch.Tensor) -> torch.Tensor:
        n_prompt = ids.shape[1]
        pos = torch.arange(n_prompt, device=ids.device)[None]

        # The small prefix cache is needed either way: it supplies the tap
        # residuals at every position the stitch decodes.
        self.small_cache = self.small.new_cache()
        xs = self.small_features(ids, pos)
        self.large_cache = self.large.new_cache()

        if self.mode == "warm":
            h = self.large.run(self.large.embed(ids), pos, 0, self.large.n_layers,
                               self.large_cache)
            return self.large.head(h)

        h = self.adapter(xs).to(self.large.dtype)
        n = min(self.preserve_prefix, n_prompt)
        if n > 0:                       # the sink, re-derived exactly (see module docstring)
            h[:, :n] = self.large.run(self.large.embed(ids[:, :n]), pos[:, :n],
                                      0, self.j, cache=None)
        h = self.large.run(h, pos, self.j, self.large.n_layers, self.large_cache)
        return self.large.head(h)

    def step(self, token_id: int, position: int) -> torch.Tensor:
        # Every decoded position is far past preserve_prefix, so the sink never
        # has to be re-derived here.
        ids = torch.tensor([[token_id]], device=self.device)
        pos = torch.tensor([[position]], device=self.device)
        h = self.adapter(self.small_features(ids, pos)).to(self.large.dtype)
        h = self.large.run(h, pos, self.j, self.large.n_layers, self.large_cache)
        return self.large.head(h)

    def active_params(self) -> int:
        """Small prefix (no unembedding — its logits are never formed) + adapter
        + large suffix. The one-off sink run through the skipped large blocks is
        per prompt, not per token, so it lands in prefill_ms, not here."""
        return (self.small.n_params(0, self.i, head=False)
                + self.adapter.n_params
                + self.large.n_params(self.j, self.large.n_layers))


@torch.no_grad()
def check_prefix_exact(lm_large: LM, ids: torch.Tensor, j: int, n: int) -> dict:
    """The sink splice claims a prefix-only run through blocks 0..j-1 equals the
    large model's own layer-j residual at those positions.

    That is an identity in exact arithmetic but not in bf16: an (1, n, d) GEMM
    tiles differently from an (1, seq, d) one. The drift is the library's, not
    ours, so the pass criterion is bit-identity against HF's own prefix-only
    forward, with HF's identical shape drift recorded alongside to show whose it is.
    """
    stack = Stack(lm_large)
    hs = lambda x: lm_large.model(x, output_hidden_states=True).hidden_states[j].float()
    full, hf_pre = hs(ids)[:, :n], hs(ids[:, :n])
    pos = torch.arange(n, device=ids.device)[None]
    ours = stack.run(stack.embed(ids[:, :n]), pos, 0, j, cache=None).float()
    rel = lambda a, b: float((a - b).norm() / (b.norm() + 1e-12))
    return {"check": "prefix_exact", "n_positions": n,
            "rel_l2_vs_hf_prefix": rel(ours, hf_pre),
            "shape_drift_rel_l2": rel(ours, full),
            "hf_own_shape_drift_rel_l2": rel(hf_pre, full),
            "passed": bool((ours == hf_pre).all())}


def run_checks(lm_small: LM, lm_large: LM, j: int,
               probe: str = "What is the capital city of New Zealand?") -> list[dict]:
    ids = build_prompt_ids(lm_large.tokenizer, probe, lm_large.device)
    return [
        check_full_stack(lm_small, ids, "small"),
        check_full_stack(lm_large, ids, "large"),
        check_baseline_matches_hf(lm_small, probe, "small"),
        check_baseline_matches_hf(lm_large, probe, "large"),
        check_prefix_exact(lm_large, ids, j, max(1, PRESERVE_PREFIX)),
    ]
