"""Step 8 — early-exit stitch: the same injection, but as a *shortcut*.

`stitch.py` (v1) proved the injection point is right. It did so the expensive
way: it ran *both* full models on the whole growing sequence at every decode
step and overwrote the large model's stream mid-flight. That is strictly slower
than either model alone, and latency was explicitly out of scope there.

This module keeps the same map and the same injection index but runs only the
layers the stitched path actually needs:

    prompt -> small embed + small blocks 0..i-1 -> adapter(W, b)
           -> large blocks j..end -> large norm -> large lm_head -> token

Large blocks `0..j-1` never run, and small blocks `i..end` never run. Both
models keep a KV cache over the layers they do run, so decoding is O(n) rather
than v1's O(n^2). For the llama defaults (1B L12 -> 3B L18) that leaves 12 of 16
small blocks and 10 of 28 large blocks on the critical path.

Because HF's `forward` always runs the full stack, the layer loop is
reimplemented here (`Stack.run`) — it is a transcription of `LlamaModel.forward`
in transformers 5.x, restricted to a layer slice. `check_full_stack` asserts
that running the *whole* slice through it reproduces `lm.model(ids).logits`, so
the transcription is verified rather than assumed.

The attention sink, in this setting
-----------------------------------
v1 could preserve the large model's own position-0 residual (PRESERVE_PREFIX)
for free, because it ran the large model's early blocks anyway. Here they are
exactly what we are trying to skip. The fix is cheap and *exact*: attention is
causal, so running only tokens `0..N-1` through large blocks `0..j-1` yields
bit-identical hidden states for those positions as running the full sequence
would. So the fast path does one N-token prefill through the skipped blocks —
N = PRESERVE_PREFIX = 1 token, against a 70+ token prompt — and splices the
result over the adapter's output for those positions. Cost is O(N) once per
prompt, not per token; `check_prefix_exact` verifies the equality.

    python -m q1.stitch_fast --pair llama --check
    python -m q1.stitch_fast --pair llama --prompt "What is the capital of Peru?"
    python -m q1.stitch_fast --pair llama --run-selected

Outputs (in results/<pair>/stitch/): fast_i{i}_j{j}.json/.csv,
fast_checks_i{i}_j{j}.json
"""

from __future__ import annotations

import argparse
import csv
import json
from time import perf_counter

import torch
from transformers import DynamicCache

from q1.config import (
    ModelPair, PAIRS, DEFAULT_PAIR, STITCH_MAX_NEW_TOKENS, PRESERVE_PREFIX,
    results_dir, stitch_dir,
)
from q1.fit_adapter import load_adapter, default_layers, validate_layers
from q1.model_utils import LM, load_pair, build_prompt_ids, generate_answer
from q1.prompts import PROMPTS
from q1.scoring import is_correct
from q1.stitch import Adapter, decoder_stack, stitched_logits

BY_ID = {p["id"]: p for p in PROMPTS}

CHECK_TOKENS = 8        # tokens compared between the cached fast path and v1 lockstep
WARMUP_TOKENS = 3       # throwaway decode steps before timing, to page in weights


def _sync(device: str) -> None:
    """Accelerator queues are asynchronous; timing without this measures nothing."""
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def _eos_ids(lm: LM) -> set[int]:
    """Every id that ends a turn — Llama-Instruct stops on <|eot_id|>, not </s>,
    and `generation_config.eos_token_id` may be a list. Matching what
    `model.generate` would stop on keeps the baseline timings honest."""
    ids = set()
    for src in (getattr(lm.model.generation_config, "eos_token_id", None),
                lm.tokenizer.eos_token_id):
        if src is None:
            continue
        ids.update(src if isinstance(src, (list, tuple)) else [src])
    return ids


# ── one model's decoder stack, sliceable ──────────────────────────────────────
class Stack:
    """Embed / decoder blocks / final norm / lm_head, callable one slice at a time.

    Transcribes `LlamaModel.forward` (transformers 5.x): build the rotary
    embeddings once for the positions in flight, then walk the blocks. Qwen2 has
    the identical structure.
    """

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
        """The attention mask for an unpadded batch of 1.

        SDPA and the flash kernels derive causality from the query length
        themselves when handed `attn_mask=None` (see
        `transformers/integrations/sdpa_attention.py`), which is also the fast
        kernel path — so passing None is both correct and what we want. `eager`
        needs the additive mask spelled out.
        """
        impl = self.config._attn_implementation
        if impl == "sdpa" or impl.startswith("flash"):
            return None
        if impl != "eager":
            raise RuntimeError(f"attn implementation {impl!r} is not supported by the "
                               f"early-exit stack; load the models with sdpa or eager.")
        q = hidden.shape[1]
        if q == 1:
            return None
        rows = torch.arange(q, device=hidden.device) + past_len
        cols = torch.arange(past_len + q, device=hidden.device)
        allowed = cols[None, :] <= rows[:, None]
        mask = torch.zeros_like(allowed, dtype=hidden.dtype)
        mask.masked_fill_(~allowed, torch.finfo(hidden.dtype).min)
        return mask[None, None]

    def run(self, hidden: torch.Tensor, position_ids: torch.Tensor,
            start: int, stop: int, cache: DynamicCache | None) -> torch.Tensor:
        """Push `hidden` through blocks [start, stop). `cache`, when given, is
        indexed by each block's own `layer_idx`, so a suffix run fills slots
        start..stop-1 and leaves the skipped ones empty — which is fine, nothing
        reads them (we pass position_ids explicitly rather than letting the cache
        infer them)."""
        pos_emb = self.base.rotary_emb(hidden, position_ids)
        mask = self._mask(hidden, int(position_ids[0, 0]))
        for layer in self.layers[start:stop]:
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
    """Baseline: one model, all its layers, KV-cached. Deliberately uses the same
    loop as the stitched path so a latency comparison is not a comparison of two
    different decoding harnesses."""

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


class StitchRunner:
    """The early-exit stitch: small blocks 0..i-1, adapter, large blocks j..end.

    Two prefill strategies, identical decode step:

    `exit`  the prompt goes through the stitch too, so large blocks 0..j-1 run
            only over the `preserve_prefix` sink positions. Maximum saving; the
            large model never sees the prompt in its own geometry.
    `warm`  the large model prefills the prompt itself, so the KV the suffix
            blocks attend to is its own and the first generated token is exactly
            the large model's. Prefill FLOPs are *not* saved — only decode is —
            and the sink needs no special handling because position 0's KV comes
            from the large model's own forward pass.
    """

    MODES = ("exit", "warm")

    def __init__(self, lm_small: LM, lm_large: LM, adapter: Adapter,
                 preserve_prefix: int = PRESERVE_PREFIX, mode: str = "exit"):
        if mode not in self.MODES:
            raise SystemExit(f"--mode must be one of {self.MODES}")
        self.small, self.large = Stack(lm_small), Stack(lm_large)
        self.adapter = adapter
        self.i, self.j = adapter.i, adapter.j
        self.preserve_prefix = preserve_prefix
        self.mode = mode
        self.label = f"stitch-{mode}"
        self.device = lm_large.device
        self.small_cache = self.large_cache = None
        # `hidden_states[n_layers]` is the one entry HF overwrites with
        # `model.norm(...)`, so when the diagnosis picks the small model's last
        # layer (qwen: i=24 of 24) the adapter was fit on the *post-norm* state,
        # not on the residual stream. Running blocks 0..i-1 alone would feed it
        # something 0.79 rel-L2 away from what it was fit on.
        self.small_post_norm = self.i == self.small.n_layers

    def small_state(self, ids: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """The small model's layer-i state, in the same convention `extract_states`
        recorded and `fit_adapter` fit against."""
        h = self.small.run(self.small.embed(ids), pos, 0, self.i, self.small_cache)
        return self.small.base.norm(h) if self.small_post_norm else h

    def prefill(self, ids: torch.Tensor) -> torch.Tensor:
        n_prompt = ids.shape[1]
        pos = torch.arange(n_prompt, device=ids.device)[None]

        # The small model's prefix cache is needed either way — it supplies
        # layer i for every position the stitch decodes.
        self.small_cache = self.small.new_cache()
        xs = self.small_state(ids, pos)
        self.large_cache = self.large.new_cache()

        if self.mode == "warm":
            h = self.large.run(self.large.embed(ids), pos, 0, self.large.n_layers,
                               self.large_cache)
            return self.large.head(h)

        h = self.adapter(xs).to(self.large.dtype)
        # The attention sink: re-derive the large model's own residual for the
        # first `preserve_prefix` positions. Causality makes the short run exact.
        n = min(self.preserve_prefix, n_prompt)
        if n > 0:
            pre = self.large.run(self.large.embed(ids[:, :n]), pos[:, :n],
                                 0, self.j, cache=None)
            h[:, :n] = pre
        h = self.large.run(h, pos, self.j, self.large.n_layers, self.large_cache)
        return self.large.head(h)

    def step(self, token_id: int, position: int) -> torch.Tensor:
        # Every decoded position is >= preserve_prefix (prompts are many tokens
        # long), so the sink never has to be re-derived here.
        ids = torch.tensor([[token_id]], device=self.device)
        pos = torch.tensor([[position]], device=self.device)
        h = self.adapter(self.small_state(ids, pos)).to(self.large.dtype)
        h = self.large.run(h, pos, self.j, self.large.n_layers, self.large_cache)
        return self.large.head(h)

    def active_params(self) -> int:
        """Small prefix (no unembedding — its logits are never formed) + the
        adapter + the large suffix. The one-off prefix run through the skipped
        large blocks is per *prompt*, not per token, so it is not counted here;
        `prefill_ms` measures it."""
        return (self.small.n_params(0, self.i, head=False)
                + self.adapter.W.numel() + self.adapter.b.numel()
                + self.large.n_params(self.j, self.large.n_layers))


@torch.no_grad()
def greedy_decode(runner, ids: torch.Tensor, max_new_tokens: int,
                  eos_ids: set[int]) -> dict:
    """Greedy decode with timing. Prefill and each decode step are synchronised
    and timed separately: they are different regimes (compute- vs
    bandwidth-bound) and only the per-step number is comparable across paths
    that stop at different lengths."""
    device = runner.device
    n_prompt = ids.shape[1]

    _sync(device)
    t0 = perf_counter()
    logits = runner.prefill(ids)
    _sync(device)
    prefill_ms = (perf_counter() - t0) * 1e3

    out, step_ms = [], []
    pos = n_prompt
    while len(out) < max_new_tokens:
        nxt = int(logits.argmax(-1))
        out.append(nxt)
        if nxt in eos_ids or len(out) == max_new_tokens:
            break
        _sync(device)
        t0 = perf_counter()
        logits = runner.step(nxt, pos)
        _sync(device)
        step_ms.append((perf_counter() - t0) * 1e3)
        pos += 1

    return {
        "token_ids": out,
        "prefill_ms": prefill_ms,
        "decode_ms_per_token": (sum(step_ms) / len(step_ms)) if step_ms else float("nan"),
        "total_ms": prefill_ms + sum(step_ms),
        "n_generated": len(out),
    }


@torch.no_grad()
def warmup(runner, ids: torch.Tensor) -> None:
    """First call on an accelerator pays for kernel setup and weight paging;
    without this the first path benchmarked looks slower than it is."""
    logits = runner.prefill(ids)
    pos = ids.shape[1]
    for _ in range(WARMUP_TOKENS):
        logits = runner.step(int(logits.argmax(-1)), pos)
        pos += 1
    _sync(runner.device)


def decode_text(lm: LM, token_ids: list[int]) -> str:
    return lm.tokenizer.decode(token_ids, skip_special_tokens=True).strip()


# ── checks: the fast path must compute what v1 computed ───────────────────────
@torch.no_grad()
def check_full_stack(lm: LM, ids: torch.Tensor, tag: str) -> dict:
    """The hand-rolled layer loop, run over *all* layers, must reproduce HF's own
    forward. This is what licenses running it over a slice."""
    stack = Stack(lm)
    ref = lm.model(ids).logits[0, -1]
    pos = torch.arange(ids.shape[1], device=ids.device)[None]
    h = stack.run(stack.embed(ids), pos, 0, stack.n_layers, stack.new_cache())
    got = stack.head(h)[0]
    return _agree(ref, got, f"full_stack_{tag}")


@torch.no_grad()
def check_baseline_matches_hf(lm: LM, question: str, tag: str,
                              max_new_tokens: int = CHECK_TOKENS) -> dict:
    """The latency claim rests on the baselines being real baselines. Both are
    timed through this module's own decode loop rather than `model.generate`, so
    that the stitch is not being compared against a different harness — which
    only holds if the loop *is* `generate`, stop condition included."""
    r = FullRunner(lm, tag)
    ids = build_prompt_ids(lm.tokenizer, question, lm.device)
    mine = decode_text(lm, greedy_decode(r, ids, max_new_tokens, _eos_ids(lm))["token_ids"])
    ref = generate_answer(lm, question, max_new_tokens)
    return {"check": f"baseline_matches_hf_generate_{tag}", "loop": mine,
            "hf_generate": ref, "passed": mine == ref}


@torch.no_grad()
def check_prefix_exact(lm_large: LM, ids: torch.Tensor, j: int, n: int) -> dict:
    """The sink fix runs only the first n tokens through blocks 0..j-1 and claims
    the result is the large model's own layer-j residual for those positions.

    Causality makes that an identity in exact arithmetic, but *not* in bf16: a
    (1, n, d) GEMM tiles differently from a (1, seq, d) one, so a short run and a
    full run disagree slightly. That drift is a property of the library, not of
    this module — so the pass criterion is bit-identity against HF's *own*
    prefix-only forward, and the drift against the full-sequence run is recorded
    alongside HF's identical drift to show whose it is.
    """
    stack = Stack(lm_large)
    hs = lambda x: lm_large.model(x, output_hidden_states=True).hidden_states[j].float()
    full, hf_pre = hs(ids)[:, :n], hs(ids[:, :n])
    pos = torch.arange(n, device=ids.device)[None]
    ours = stack.run(stack.embed(ids[:, :n]), pos, 0, j, cache=None).float()
    rel = lambda a, b: float((a - b).norm() / (b.norm() + 1e-12))
    return {"check": "prefix_exact", "n_positions": n,
            "rel_l2_vs_hf_prefix": rel(ours, hf_pre),
            "max_abs_diff_vs_hf_prefix": float((ours - hf_pre).abs().max()),
            "shape_drift_rel_l2": rel(ours, full),
            "hf_own_shape_drift_rel_l2": rel(hf_pre, full),
            "passed": bool((ours == hf_pre).all())}


@torch.no_grad()
def check_matches_v1(lm_small: LM, lm_large: LM, ids: torch.Tensor, adapter: Adapter,
                     preserve_prefix: int) -> dict:
    """Same arithmetic, fewer layers: the fast prefill's next-token logits must
    match v1's, which computed them by running both full models."""
    ref = stitched_logits(lm_small, lm_large, ids, adapter,
                          preserve_prefix=preserve_prefix)[0, -1]
    got = StitchRunner(lm_small, lm_large, adapter, preserve_prefix).prefill(ids)[0]
    return _agree(ref, got, "fast_equals_v1_prefill")


@torch.no_grad()
def _cache_drift(runner, recompute, ids: torch.Tensor, n_tokens: int):
    """Teacher-forced comparison of a KV-cached path against a from-scratch
    recompute: the cached path picks the tokens, and `recompute` re-derives the
    next-token logits on that exact prefix, so the two are compared position by
    position on identical context.

    Forcing the tokens is the point. Let both run free and a single bf16 rounding
    difference picks a different token, after which the whole divergent
    continuation gets charged to the cache."""
    n_prompt = ids.shape[1]
    cached = [runner.prefill(ids)[0]]
    for k in range(n_tokens):
        nxt = int(cached[-1].argmax())
        ids = torch.cat([ids, torch.tensor([[nxt]], device=ids.device)], dim=1)
        cached.append(runner.step(nxt, n_prompt + k)[0])

    rels, agree = [], 0
    for k, got in enumerate(cached):
        ref = recompute(ids[:, :n_prompt + k]).float()
        rels.append(float((ref - got.float()).norm() / (ref.norm() + 1e-12)))
        agree += int(ref.argmax()) == int(got.argmax())
    return max(rels), agree, len(cached), [int(c.argmax()) for c in cached[:-1]]


@torch.no_grad()
def check_cached_decode(lm_small: LM, lm_large: LM, question: str, adapter: Adapter,
                        preserve_prefix: int, n_tokens: int = CHECK_TOKENS,
                        slack: float = 3.0) -> dict:
    """The KV cache must reproduce what v1 recomputed from scratch every step.

    The tolerance is measured, not chosen: bf16 KV-cached decode is not
    reproducible against a full recompute for *any* of these models (an
    unmodified Llama-3B drifts ~3.9e-2 on MPS, because a 1-token GEMM tiles
    differently from a seq-token one), so the same drift is measured for the
    unmodified large model in the same run and the stitch has to stay within
    `slack` of that floor. Hard-coding a constant here would either fail the
    honest path or hide a real cache bug, depending on the hardware.
    """
    ids = build_prompt_ids(lm_large.tokenizer, question, lm_large.device)
    rel, agree, n, toks = _cache_drift(
        StitchRunner(lm_small, lm_large, adapter, preserve_prefix),
        lambda x: stitched_logits(lm_small, lm_large, x, adapter,
                                  preserve_prefix=preserve_prefix)[0, -1],
        ids, n_tokens)
    floor, floor_agree, _, _ = _cache_drift(
        FullRunner(lm_large, "large"), lambda x: lm_large.model(x).logits[0, -1],
        ids, n_tokens)
    budget = max(slack * floor, 1e-4)
    return {"check": "cached_equals_lockstep", "n_positions": n,
            "max_rel_l2": rel, "argmax_agree": agree,
            "unmodified_large_max_rel_l2": floor,
            "unmodified_large_argmax_agree": floor_agree,
            "budget": budget, "slack": slack,
            "tokens": decode_text(lm_large, toks), "passed": rel <= budget}


@torch.no_grad()
def check_warm_prefill(lm_small: LM, lm_large: LM, ids: torch.Tensor, adapter: Adapter,
                       preserve_prefix: int) -> dict:
    """`warm` mode must hand back the large model's own prompt logits untouched —
    its first generated token is the large model's by construction, and the
    stitch only takes over from the second token on."""
    ref = FullRunner(lm_large, "large").prefill(ids)[0]
    got = StitchRunner(lm_small, lm_large, adapter, preserve_prefix,
                       mode="warm").prefill(ids)[0]
    return _agree(ref, got, "warm_prefill_equals_large")


def _agree(ref: torch.Tensor, got: torch.Tensor, name: str,
           rel_tol: float = 0.05) -> dict:
    """Compare two logit vectors. Argmax alone is too weak a criterion — it let a
    real convention bug through with a 15.6 max logit gap — so the relative
    distance has to be small as well. `rel_tol` is set at the measured bf16
    reproducibility floor for these models (see `check_cached_decode`), not
    tighter: two implementations of the same arithmetic differ by that much."""
    ref, got = ref.float(), got.float()
    rel = float((ref - got).norm() / (ref.norm() + 1e-12))
    same = int(ref.argmax()) == int(got.argmax())
    return {"check": name, "max_abs_logit_diff": float((ref - got).abs().max()),
            "rel_l2": rel, "argmax_match": same, "bit_identical": bool((ref == got).all()),
            "rel_tol": rel_tol, "passed": same and rel < rel_tol}


def run_checks(lm_small: LM, lm_large: LM, adapter: Adapter, preserve_prefix: int,
               probe: str = "What is the capital city of New Zealand?") -> list[dict]:
    ids = build_prompt_ids(lm_large.tokenizer, probe, lm_large.device)
    checks = [
        check_full_stack(lm_small, ids, "small"),
        check_full_stack(lm_large, ids, "large"),
        check_baseline_matches_hf(lm_small, probe, "small"),
        check_baseline_matches_hf(lm_large, probe, "large"),
        check_prefix_exact(lm_large, ids, adapter.j, max(1, preserve_prefix)),
        check_matches_v1(lm_small, lm_large, ids, adapter, preserve_prefix),
        check_cached_decode(lm_small, lm_large, probe, adapter, preserve_prefix),
        check_warm_prefill(lm_small, lm_large, ids, adapter, preserve_prefix),
    ]
    print("\n" + "=" * 72 + "\nEARLY-EXIT CHECKS\n" + "=" * 72)
    for c in checks:
        flag = "PASS" if c["passed"] else "FAIL"
        if "max_abs_logit_diff" in c:
            extra = (f"  relL2={c['rel_l2']:.2e}  max|d|={c['max_abs_logit_diff']:.3e}  "
                     f"argmax_match={c['argmax_match']}")
        elif c["check"].startswith("baseline_matches_hf"):
            extra = f"  {c['loop']!r} == generate() {c['hf_generate']!r}"
        elif c["check"] == "prefix_exact":
            extra = (f"  identical to HF's own {c['n_positions']}-token forward "
                     f"(relL2={c['rel_l2_vs_hf_prefix']:.1e}); bf16 shape drift vs the "
                     f"full run {c['shape_drift_rel_l2']:.1e}, HF's own "
                     f"{c['hf_own_shape_drift_rel_l2']:.1e}")
        else:
            extra = (f"  max relL2={c['max_rel_l2']:.2e} vs a {c['unmodified_large_max_rel_l2']:.2e} "
                     f"bf16 floor (budget {c['budget']:.2e}), argmax agrees "
                     f"{c['argmax_agree']}/{c['n_positions']}  ({c['tokens']!r})")
        print(f"  [{flag}] {c['check']}{extra}")
    return checks


# ── driver ────────────────────────────────────────────────────────────────────
def _pick_prompts(pair: ModelPair, prompts, run_selected, n_divergent, n_control):
    items = [{"prompt_id": None, "set": "custom", "question": q, "gold": None}
             for q in (prompts or [])]
    if items and not run_selected:
        return items
    sel_path = results_dir(pair) / "selection.json"
    if not sel_path.exists():
        raise SystemExit(f"{sel_path} not found — run the select step, or pass --prompt.")
    with open(sel_path) as f:
        sel = json.load(f)
    n_div = None if run_selected else n_divergent
    for pid in sel["divergent_small_wrong_large_right"][:n_div]:
        items.append({"prompt_id": pid, "set": "divergent",
                      "question": BY_ID[pid]["question"], "gold": BY_ID[pid].get("answers")})
    for pid in sel["control_both_right"][:n_control]:
        items.append({"prompt_id": pid, "set": "control",
                      "question": BY_ID[pid]["question"], "gold": BY_ID[pid].get("answers")})
    return items


def _mean(xs):
    xs = [x for x in xs if x == x]                      # drop NaN (0-step decodes)
    return sum(xs) / len(xs) if xs else float("nan")


def run(pair: ModelPair, lm_small: LM | None = None, lm_large: LM | None = None,
        i: int | None = None, j: int | None = None,
        prompts: list[str] | None = None, run_selected: bool = False,
        checks_only: bool = False, max_new_tokens: int = STITCH_MAX_NEW_TOKENS,
        n_divergent: int = 6, n_control: int = 6, modes: tuple[str, ...] = ("exit", "warm"),
        preserve_prefix: int = PRESERVE_PREFIX, skip_checks: bool = False) -> dict:
    if pair.align != "token":
        raise SystemExit(f"pair '{pair.name}' is {pair.align}-aligned; stitching is "
                         f"token-aligned pairs only.")
    if i is None or j is None:
        di, dj = default_layers(pair)
        i, j = (di if i is None else i), (dj if j is None else j)
    validate_layers(pair, i, j)

    arrays, meta = load_adapter(pair, i, j)
    if lm_small is None or lm_large is None:
        lm_small, lm_large = load_pair(pair)
    adapter = Adapter(arrays, meta, lm_large.device)

    # Baselines first: every stitch mode is reported relative to these two.
    runners = {"small": FullRunner(lm_small, pair.small_tag),
               "large": FullRunner(lm_large, pair.large_tag)}
    for m in modes:
        runners[m] = StitchRunner(lm_small, lm_large, adapter, preserve_prefix, mode=m)
    print(f"[stitch-fast] {pair.name}: {pair.small_tag} blocks 0-{i - 1} -> adapter -> "
          f"{pair.large_tag} blocks {j}-{lm_large.n_layers - 1}  "
          f"(skipping {j} large and {lm_small.n_layers - i} small blocks)")

    out_dir = stitch_dir(pair)
    out_dir.mkdir(parents=True, exist_ok=True)

    checks = []
    if not skip_checks:
        checks = run_checks(lm_small, lm_large, adapter, preserve_prefix)
        with open(out_dir / f"fast_checks_i{i:02d}_j{j:02d}.json", "w") as f:
            json.dump({"pair": pair.name, "i": i, "j": j, "checks": checks}, f, indent=2)
        failed = [c["check"] for c in checks if not c["passed"]]
        if failed:
            raise SystemExit(f"early-exit check(s) failed: {failed} — refusing to report "
                             f"latency for a path that does not compute what v1 computed.")
        if checks_only:
            print(f"\nWrote {out_dir}/fast_checks_i{i:02d}_j{j:02d}.json")
            return {"checks": checks}

    items = _pick_prompts(pair, prompts, run_selected, n_divergent, n_control)
    eos = {"small": _eos_ids(lm_small)}
    eos.update({n: _eos_ids(lm_large) for n in runners if n != "small"})

    warm_ids = build_prompt_ids(lm_large.tokenizer, items[0]["question"], lm_large.device)
    for r in runners.values():
        warmup(r, warm_ids)

    print("\n" + "=" * 72 + f"\nBENCHMARK  ({len(items)} prompts, "
          f"<= {max_new_tokens} new tokens, greedy, KV-cached)\n" + "=" * 72)
    rows = []
    for k, it in enumerate(items, 1):
        q = it["question"]
        ids = build_prompt_ids(lm_large.tokenizer, q, lm_large.device)
        res = {n: greedy_decode(r, ids, max_new_tokens, eos[n]) for n, r in runners.items()}
        row = {**it, "n_prompt_tokens": int(ids.shape[1])}
        for name, r in res.items():
            lm = lm_small if name == "small" else lm_large
            row[name] = decode_text(lm, r["token_ids"])
            row[f"{name}_first_token"] = lm.tokenizer.decode(r["token_ids"][:1])
            row[f"{name}_correct"] = (is_correct(row[name], it["gold"])
                                      if it["gold"] else None)
            for m in ("prefill_ms", "decode_ms_per_token", "total_ms", "n_generated"):
                row[f"{name}_{m}"] = r[m]
        for m in modes:
            row[f"{m}_first_token_eq_large"] = (res[m]["token_ids"][:1]
                                                == res["large"]["token_ids"][:1])
            row[f"{m}_first_token_eq_small"] = (res[m]["token_ids"][:1]
                                                == res["small"]["token_ids"][:1])
        rows.append(row)

        tag = f"[{it['set']}]" + (f" {it['prompt_id']}" if it["prompt_id"] else "")
        print(f"\n({k}/{len(items)}) {tag}  {q}")
        if it["gold"]:
            print(f"      gold     : {it['gold']}")
        for name, r in runners.items():
            mark = "" if row[f"{name}_correct"] is None else (
                "  OK " if row[f"{name}_correct"] else "  -- ")
            print(f"      {r.label:>12} :{mark}{row[name]!r}   "
                  f"[{row[f'{name}_prefill_ms']:.0f} ms prefill, "
                  f"{row[f'{name}_decode_ms_per_token']:.1f} ms/tok]")

    # ── aggregate ─────────────────────────────────────────────────────────────
    paths = {}
    for name, runner in runners.items():
        ms_tok = _mean([r[f"{name}_decode_ms_per_token"] for r in rows])
        paths[name] = {
            "label": runner.label,
            "prefill_ms": _mean([r[f"{name}_prefill_ms"] for r in rows]),
            "decode_ms_per_token": ms_tok,
            "tokens_per_second": 1e3 / ms_tok if ms_tok == ms_tok else float("nan"),
            "total_ms": _mean([r[f"{name}_total_ms"] for r in rows]),
            "mean_n_generated": _mean([float(r[f"{name}_n_generated"]) for r in rows]),
            "active_params_per_token": runner.active_params(),
        }
    for name, p in paths.items():
        p["decode_speedup_vs_large"] = (paths["large"]["decode_ms_per_token"]
                                        / p["decode_ms_per_token"])
        p["total_speedup_vs_large"] = paths["large"]["total_ms"] / p["total_ms"]
        p["params_frac_of_large"] = (p["active_params_per_token"]
                                     / paths["large"]["active_params_per_token"])

    by_set = {}
    for s in sorted({r["set"] for r in rows}):
        sub = [r for r in rows if r["set"] == s]
        scored = [r for r in sub if r["gold"]]
        by_set[s] = {
            "n": len(sub), "n_scored": len(scored),
            **{f"{name}_correct": sum(bool(r[f"{name}_correct"]) for r in scored)
               for name in runners},
            **{f"{m}_first_token_eq_{b}": sum(r[f"{m}_first_token_eq_{b}"] for r in sub)
               for m in modes for b in ("large", "small")},
        }

    payload = {
        "pair": pair.name, "modes": list(modes),
        "small_layer_i": i, "large_layer_j": j,
        "preserve_prefix": preserve_prefix, "max_new_tokens": max_new_tokens,
        "device": lm_large.device,
        "skipped_large_blocks": j, "skipped_small_blocks": lm_small.n_layers - i,
        "adapter": {k: meta[k] for k in ("fit_set", "n_fit_rows", "ridge_alpha",
                                         "in_sample", "held_out", "held_out_last_token")},
        "checks": checks, "n_prompts": len(rows),
        "latency": paths, "by_set": by_set, "generations": rows,
    }
    json_path = out_dir / f"fast_i{i:02d}_j{j:02d}.json"
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    csv_path = out_dir / f"fast_i{i:02d}_j{j:02d}.csv"
    fields = ["prompt_id", "set", "question", "gold", "n_prompt_tokens"] + [
        f"{n}{sfx}" for n in runners
        for sfx in ("", "_first_token", "_correct", "_prefill_ms",
                    "_decode_ms_per_token", "_total_ms", "_n_generated")
    ] + [f"{m}_first_token_eq_{b}" for m in modes for b in ("large", "small")]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({**{k: r[k] for k in fields if k != "gold"},
                        "gold": "|".join(r["gold"]) if r["gold"] else ""})

    # ── report ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print(f"LATENCY  (mean over {len(rows)} prompts, {lm_large.device}; ms/token is the "
          f"length-independent number)")
    print(f"  {'path':<13}{'prefill ms':>12}{'ms/token':>11}{'tok/s':>9}"
          f"{'params/token':>15}{'decode vs large':>17}")
    for name, p in paths.items():
        print(f"  {p['label']:<13}{p['prefill_ms']:>12.1f}{p['decode_ms_per_token']:>11.2f}"
              f"{p['tokens_per_second']:>9.1f}"
              f"{p['active_params_per_token'] / 1e9:>13.2f} B"
              f"{p['decode_speedup_vs_large']:>16.2f}x")
    for m in modes:
        p = paths[m]
        print(f"\n  {p['label']}: decode {p['decode_speedup_vs_large']:.2f}x "
              f"{pair.large_tag}-alone ({p['decode_ms_per_token'] / paths['small']['decode_ms_per_token']:.2f}x "
              f"the cost of {pair.small_tag}-alone), multiplying "
              f"{p['params_frac_of_large']:.0%} of the {pair.large_tag}'s per-token weights"
              + ("; prefill is a full large-model pass, so no prefill FLOPs are saved."
                 if m == "warm" else "."))

    print(f"\nACCURACY  (gold-answer match, re-scored here — the sets come from "
          f"selection.json)")
    hdr = f"  {'set':<12}{'n':>4}" + "".join(f"{r.label:>13}" for r in runners.values())
    print(hdr)
    for s, d in by_set.items():
        if not d["n_scored"]:
            continue
        print(f"  {s:<12}{d['n_scored']:>4}"
              + "".join(f"{d[f'{n}_correct']:>13}" for n in runners))
    print(f"\n  first token == baseline  " + "".join(f"{r.label:>13}" for r in runners.values()))
    for b, lbl in (("large", pair.large_tag), ("small", pair.small_tag)):
        cells = "".join(f"{'-':>13}" if n not in modes
                        else f"{sum(d[f'{n}_first_token_eq_{b}'] for d in by_set.values()):>13}"
                        for n in runners)
        print(f"  == {lbl:<22}{cells}")
    div = by_set.get("divergent")
    if div and div["small_correct"]:
        print(f"\n  note: {pair.small_tag} scores {div['small_correct']}/{div['n_scored']} on the "
              f"'divergent' set, which selection.json recorded as 0 — that selection was "
              f"made on a different library/hardware build, so treat these buckets as "
              f"prompt sets, not as re-derived labels.")
    print("=" * 72)
    print(f"\nWrote {json_path}\n      {csv_path}")
    return payload


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair", choices=sorted(PAIRS), default=DEFAULT_PAIR)
    ap.add_argument("--i", type=int, default=None)
    ap.add_argument("--j", type=int, default=None)
    ap.add_argument("--prompt", action="append", default=None,
                    help="question to benchmark (repeatable)")
    ap.add_argument("--run-selected", action="store_true",
                    help="benchmark every divergent prompt from selection.json "
                         "(+ --n-control controls)")
    ap.add_argument("--n-divergent", type=int, default=6,
                    help="divergent prompts to benchmark (default: 6; ignored under "
                         "--run-selected, which uses all of them)")
    ap.add_argument("--n-control", type=int, default=6,
                    help="control prompts to benchmark (default: 6)")
    ap.add_argument("--mode", choices=(*StitchRunner.MODES, "both"), default="both",
                    help="exit: the prompt goes through the stitch too (max saving); "
                         "warm: the large model prefills the prompt itself and only "
                         "decoding is stitched (default: both, reported side by side)")
    ap.add_argument("--check", action="store_true",
                    help="run the early-exit equivalence checks only")
    ap.add_argument("--skip-checks", action="store_true",
                    help="benchmark without first re-verifying against v1 (the checks "
                         "run both full models, so they cost more than the benchmark)")
    ap.add_argument("--preserve-prefix", type=int, default=PRESERVE_PREFIX,
                    help=f"leading positions re-derived from the large model's own early "
                         f"blocks (default {PRESERVE_PREFIX}; 0 overwrites the attention "
                         f"sink and produces noise)")
    ap.add_argument("--max-new-tokens", type=int, default=STITCH_MAX_NEW_TOKENS)
    a = ap.parse_args()
    modes = StitchRunner.MODES if a.mode == "both" else (a.mode,)
    run(PAIRS[a.pair], i=a.i, j=a.j, prompts=a.prompt, run_selected=a.run_selected,
        checks_only=a.check, max_new_tokens=a.max_new_tokens,
        n_divergent=a.n_divergent, n_control=a.n_control, modes=modes,
        preserve_prefix=a.preserve_prefix, skip_checks=a.skip_checks)


if __name__ == "__main__":
    main()
