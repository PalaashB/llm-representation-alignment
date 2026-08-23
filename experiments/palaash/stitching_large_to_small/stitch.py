"""Large->small stitched decoding, and the checks that gate reporting.

    prompt -> large embed + large blocks 0..j-1 -> adapter -> small blocks
    i..end -> small norm -> small lm_head -> token

Small blocks 0..i-1 and large blocks j..end never run during decode. Both models
keep a KV cache over the layers they do run.

The sliced layer loop (`Stack`), the KV-cached baselines, and the greedy decode
come from `common.decoding`, which both stitching packages share; only the
direction-specific runner and checks live here.

The attention sink now sits on the *small* stream, because that is the stream
being overwritten. Position 0 carries a residual far larger than any other and
the adapter cannot reproduce it, so the path re-derives it by running the
skipped small blocks over that one token — exact, because attention is causal.

Checks
------
The accuracy numbers are worthless if the injection is plumbed wrong, and a
wrong injection usually still produces fluent text, so it does not announce
itself. Four checks gate every report:

  full_stack_*        the hand-rolled loop reproduces HF's own forward
  identity_injection  feeding the small model its OWN layer-i residual at block
                      i reproduces the unmodified small model exactly. This is
                      the one that pins the convention: it fails if `i` is off
                      by one, if the residual is taken after the block instead
                      of before it, or if the resumed slice is wrong.
  offbyone_control    feeding layer i+1's residual at block i must DIFFER.
                      Without it, `identity_injection` could pass on a path that
                      silently ignores the injected value altogether.
  adapter_reload      the saved coefficients round-trip and score finitely.
"""

from __future__ import annotations

import torch

from common.decoding import (
    FullRunner, Stack, check_baseline_matches_hf, check_full_stack, greedy,
)
from common.model_utils import LM, build_prompt_ids
from stitching_large_to_small.adapter import TorchAdapter
from stitching_large_to_small.config import PRESERVE_PREFIX


class LargeToSmallRunner:
    """Large prefix -> adapter -> small suffix, KV-cached on both sides.

    `exit`  the prompt goes through the stitch too: the small model's blocks
            0..i-1 run over the sink position only, and every small KV the
            suffix reads is the adapter's reconstruction of the large model's
            reading of the prompt. This is the mode that actually transfers the
            large model's prompt comprehension.
    `warm`  the small model prefills the prompt itself, so its prompt KVs are
            its own and the adapter supplies the residual only at generated
            positions. Cheaper to get right, but the large model's contribution
            is limited to the tokens being generated.

    Neither mode is fast. Both run j large blocks and (n_small - i) small blocks
    per token, plus a full large-model prefix prefill. Accuracy is the point;
    `evaluate` reports the cost so it cannot be quietly dropped.
    """

    MODES = ("exit", "warm")

    def __init__(self, lm_small: LM, lm_large: LM, adapter: TorchAdapter,
                 preserve_prefix: int = PRESERVE_PREFIX, mode: str = "exit"):
        if mode not in self.MODES:
            raise SystemExit(f"mode must be one of {self.MODES}")
        self.small, self.large = Stack(lm_small), Stack(lm_large)
        self.adapter = adapter
        self.i, self.j = adapter.i, adapter.j
        self.preserve_prefix = preserve_prefix
        self.mode = mode
        self.label = f"stitch-{mode}-largeL{self.j}->smallL{self.i}"
        self.device = lm_small.device
        self.small_cache = self.large_cache = None

    def large_features(self, ids: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """Run large blocks 0..j-1 and return the residual at the input to block
        j — the same quantity `hidden_states[j]` names, which is what capture
        recorded and the adapter was fit on."""
        return self.large.run(self.large.embed(ids), pos, 0, self.j, self.large_cache)

    def prefill(self, ids: torch.Tensor) -> torch.Tensor:
        n_prompt = ids.shape[1]
        pos = torch.arange(n_prompt, device=ids.device)[None]

        # The large prefix cache is needed in both modes: it supplies layer j at
        # every position the stitch decodes.
        self.large_cache = self.large.new_cache()
        xl = self.large_features(ids, pos)
        self.small_cache = self.small.new_cache()

        if self.mode == "warm":
            h = self.small.run(self.small.embed(ids), pos, 0, self.small.n_layers,
                               self.small_cache)
            return self.small.head(h)

        h = self.adapter(xl).to(self.small.dtype)
        n = min(self.preserve_prefix, n_prompt)
        if n > 0:   # the sink, re-derived exactly from the small model's own blocks
            h[:, :n] = self.small.run(self.small.embed(ids[:, :n]), pos[:, :n],
                                      0, self.i, cache=None)
        h = self.small.run(h, pos, self.i, self.small.n_layers, self.small_cache)
        return self.small.head(h)

    def step(self, token_id: int, position: int) -> torch.Tensor:
        ids = torch.tensor([[token_id]], device=self.device)
        pos = torch.tensor([[position]], device=self.device)
        h = self.adapter(self.large_features(ids, pos)).to(self.small.dtype)
        h = self.small.run(h, pos, self.i, self.small.n_layers, self.small_cache)
        return self.small.head(h)

    def active_params(self) -> int:
        """Large prefix (no unembedding — its logits are never formed) + adapter
        + small suffix, per decoded token."""
        return (self.large.n_params(0, self.j, head=False)
                + self.adapter.n_params
                + self.small.n_params(self.i, self.small.n_layers))


# ── direction-specific checks ─────────────────────────────────────────────────
@torch.no_grad()
def _resume_small(lm_small: LM, ids: torch.Tensor, inject: torch.Tensor,
                  i: int) -> torch.Tensor:
    """Run the small model from block i with `inject` as the residual, and
    return the next-token logits. This is exactly the stitched suffix, with the
    adapter's output replaced by whatever the caller supplies."""
    stack = Stack(lm_small)
    pos = torch.arange(ids.shape[1], device=ids.device)[None]
    h = stack.run(inject, pos, i, stack.n_layers, cache=None)
    return stack.head(h)[0].float()


@torch.no_grad()
def check_identity_injection(lm_small: LM, ids: torch.Tensor, i: int) -> dict:
    """Feed the small model its OWN layer-i residual at block i.

    Nothing has been changed, so the logits must match the unmodified model.
    This is what pins down the injection convention — that `hidden_states[i]` is
    the *input* to block i, and that resuming at `i` means running blocks
    i..end. An off-by-one anywhere in that chain breaks it.
    """
    ref = lm_small.model(ids).logits[0, -1].float()
    own = lm_small.model(ids, output_hidden_states=True).hidden_states[i]
    got = _resume_small(lm_small, ids, own, i)
    rel = float((ref - got).norm() / (ref.norm() + 1e-12))
    return {"check": "identity_injection", "layer_i": i, "rel_l2": rel,
            "max_abs_logit_diff": float((ref - got).abs().max()),
            "argmax_match": int(ref.argmax()) == int(got.argmax()),
            "passed": rel < 5e-3 and int(ref.argmax()) == int(got.argmax())}


@torch.no_grad()
def check_offbyone_control(lm_small: LM, ids: torch.Tensor, i: int) -> dict:
    """Negative control: inject layer i+1's residual at block i.

    That is the wrong tensor, so the result must *differ* from the unmodified
    model. If it does not, the resumed slice is ignoring what it was handed and
    `identity_injection` proved nothing.
    """
    ref = lm_small.model(ids).logits[0, -1].float()
    wrong = lm_small.model(ids, output_hidden_states=True).hidden_states[i + 1]
    got = _resume_small(lm_small, ids, wrong, i)
    rel = float((ref - got).norm() / (ref.norm() + 1e-12))
    return {"check": "offbyone_control", "layer_injected": i + 1, "at_block": i,
            "rel_l2": rel, "argmax_match": int(ref.argmax()) == int(got.argmax()),
            # A real difference, not a rounding one. The identity check's
            # tolerance is 5e-3, so anything at least ten times that is
            # unambiguously a different computation.
            "passed": rel > 5e-2}


@torch.no_grad()
def check_prefix_exact(lm_small: LM, ids: torch.Tensor, i: int, n: int) -> dict:
    """The sink splice claims a prefix-only run through small blocks 0..i-1
    equals the small model's own layer-i residual at those positions. Causality
    makes that an identity in exact arithmetic; in bf16 the pass criterion is
    bit-identity against HF's own prefix-only forward, with HF's identical shape
    drift recorded alongside to show whose it is."""
    stack = Stack(lm_small)
    hs = lambda x: lm_small.model(x, output_hidden_states=True).hidden_states[i].float()
    full, hf_pre = hs(ids)[:, :n], hs(ids[:, :n])
    pos = torch.arange(n, device=ids.device)[None]
    ours = stack.run(stack.embed(ids[:, :n]), pos, 0, i, cache=None).float()
    rel = lambda a, b: float((a - b).norm() / (b.norm() + 1e-12))
    return {"check": "prefix_exact", "n_positions": n,
            "rel_l2_vs_hf_prefix": rel(ours, hf_pre),
            "shape_drift_rel_l2": rel(ours, full),
            "hf_own_shape_drift_rel_l2": rel(hf_pre, full),
            "passed": bool((ours == hf_pre).all())}


def run_checks(lm_small: LM, lm_large: LM, i: int, j: int,
               probe: str = "What is the capital city of New Zealand?") -> list[dict]:
    ids = build_prompt_ids(lm_small.tokenizer, probe, lm_small.device)
    return [
        check_full_stack(lm_small, ids, "small"),
        check_full_stack(lm_large, ids, "large"),
        check_baseline_matches_hf(lm_small, probe, "small"),
        check_identity_injection(lm_small, ids, i),
        check_offbyone_control(lm_small, ids, i),
        check_prefix_exact(lm_small, ids, i, max(1, PRESERVE_PREFIX)),
    ]
