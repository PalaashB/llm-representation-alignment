"""Synthetic smoke test for the large->small distillation path — no real weights.

Builds a tiny randomly-initialised Llama pair on CPU and drives the machinery
the real run depends on: the trainable adapter, the KL objective, the frozen-LLM
assertions, the warm start, the zero-epoch identity, and the saved-map reload.
The point is to fail in seconds here rather than an hour into an MPS run.

This is the mirror of `smoke_distill.py`, and the difference is the whole reason
it exists as a separate file: here the loss flows through the *small* model's
suffix, the base stream is the small model's own residual, and the teacher is
the large model. Every one of those is a place where a copied-across sign or
argument order would still run and still train, while optimising the wrong path.

What it proves, and what it does not: the plumbing is exercised end to end and
the invariants hold. Nothing here says distillation helps on real models —
random weights have no next-token structure to learn. That question is answered
by the head-to-head bench.

    python3 tests/smoke_distill_l2s.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transformers import LlamaConfig, LlamaForCausalLM  # noqa: E402

from common.decoding import Stack, check_full_stack  # noqa: E402
from common.model_utils import LM  # noqa: E402
from stitching_large_to_small import distill  # noqa: E402

VOCAB, D_SMALL, D_LARGE, N_SMALL, N_LARGE = 128, 32, 48, 6, 8
SEQ, N_SEQ, TOPK = 12, 6, 8


def tiny(dim: int, layers: int, tag: str) -> LM:
    torch.manual_seed(0 if dim == D_SMALL else 1)
    cfg = LlamaConfig(vocab_size=VOCAB, hidden_size=dim, intermediate_size=dim * 2,
                      num_hidden_layers=layers, num_attention_heads=4,
                      num_key_value_heads=4, max_position_embeddings=64)
    model = LlamaForCausalLM(cfg).eval()
    return LM(tag=tag, tokenizer=None, model=model, device="cpu")


def main() -> int:
    lm_small, lm_large = tiny(D_SMALL, N_SMALL, "S"), tiny(D_LARGE, N_LARGE, "L")
    i, j = 3, 4
    rng = np.random.default_rng(0)

    # The sliced layer loop must reproduce HF's own forward before anything
    # built on top of it means anything. Here it is the SMALL model's loop that
    # carries the gradient, so that is the one to check.
    ids = torch.randint(0, VOCAB, (1, SEQ))
    chk = check_full_stack(lm_small, ids, "small")
    assert chk["passed"], f"full_stack check failed: {chk}"
    print(f"[ok] full_stack(small) rel_l2={chk['rel_l2']:.2e}")

    # Synthetic capture: X = large layer j, Y = small layer i, teacher top-K
    # (the LARGE model's distribution), row metadata.
    n = SEQ * N_SEQ
    X = rng.normal(size=(n, D_LARGE)).astype(np.float32)
    Y = rng.normal(size=(n, D_SMALL)).astype(np.float32)
    pid = np.repeat(np.arange(N_SEQ), SEQ).astype(np.int32)
    position = np.tile(np.arange(SEQ), N_SEQ).astype(np.int32)
    is_answer = (position >= SEQ // 2).astype(np.int8)
    t_idx = rng.integers(0, VOCAB, size=(n, TOPK)).astype(np.int32)
    t_vals = np.sort(rng.normal(size=(n, TOPK)).astype(np.float16), axis=1)[:, ::-1].copy()

    base = {"W": rng.normal(scale=0.02, size=(D_LARGE, D_SMALL)).astype(np.float32),
            "b": np.zeros(D_SMALL, np.float32),
            "mu_x": X.mean(0), "sd_x": X.std(0) + 1e-6, "mu_y": Y.mean(0)}

    m, meta = distill.train(
        None, None, i, j, base, lm_small, X, Y, pid, is_answer, position,
        t_vals, t_idx, lm_large=lm_large, epochs=3, lr=1e-2, batch_seqs=2,
        val_frac=0.34, verbose=False)

    # 1. Nothing but the adapter was trainable — and both LLMs were checked,
    #    not just the one the gradient passes through.
    ev = meta["frozen_llm_evidence"]
    assert ev["llm_params_trainable"] == 0, ev
    assert ev["adapter_tensors_trained"] == 2, ev          # W and b only
    print(f"[ok] frozen: {ev['llm_params_total']} LLM tensors across both models, "
          f"0 trainable; {ev['adapter_params_trained']} adapter params trained")

    # 2. No LLM weight actually moved (belt and braces on the assertion above).
    for lm, ref_lm, name in ((lm_small, tiny(D_SMALL, N_SMALL, "S"), "small"),
                             (lm_large, tiny(D_LARGE, N_LARGE, "L"), "large")):
        after = torch.cat([p.flatten() for p in lm.model.parameters()])
        ref = torch.cat([p.flatten() for p in ref_lm.model.parameters()])
        assert torch.equal(after, ref), f"a {name}-model weight changed during training"
    print("[ok] both models' weights bit-identical after training")

    # 3. Warm start: epoch 0 is the ridge map, and training improved on it.
    h = meta["distill_history"]
    assert h[0]["epoch"] == 0 and h[0]["train_loss"] is None
    assert meta["distill_val_loss_best"] <= meta["distill_val_loss_ridge"] + 1e-9
    print(f"[ok] warm start {meta['distill_val_loss_ridge']:.4f} -> "
          f"{meta['distill_val_loss_best']:.4f} @ epoch {meta['distill_best_epoch']} "
          f"(improved: {meta['distill_improved_on_ridge']})")

    # 4. Shape and finiteness: W maps large -> small in this direction.
    assert m["W"].shape == (D_LARGE, D_SMALL) and np.isfinite(m["W"]).all()

    # 5. A zero-epoch run IS the ridge map, bit for bit. That is what makes
    #    "distill with 0 steps must score exactly what ridge scored" a statement
    #    about the file rather than about the benchmark: identical arrays decode
    #    identically, so no GPU time has to be spent proving it.
    z, zmeta = distill.train(
        None, None, i, j, base, lm_small, X, Y, pid, is_answer, position,
        t_vals, t_idx, lm_large=lm_large, epochs=0, batch_seqs=2, val_frac=0.34,
        verbose=False)
    for k in ("W", "b", "mu_x", "sd_x", "mu_y"):
        assert np.array_equal(z[k], base[k].astype(np.float32)), \
            f"0-epoch distill changed {k}"
    assert zmeta["distill_best_epoch"] == 0 and not zmeta["distill_improved_on_ridge"]
    print("[ok] 0-epoch distill returns the ridge map bit-for-bit")

    # 6. The saved arrays reproduce the training-time logits, through a numpy
    #    round-trip and the frozen small suffix.
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "a.npz"
        np.savez(p, **m)
        loaded = {k: v for k, v in np.load(p).items()}
    stack = Stack(lm_small)
    probe = np.flatnonzero(pid == pid[meta["probe_prompt_row0"]])
    probe = probe[np.argsort(position[probe], kind="stable")]
    ans = is_answer[probe] == 1
    with torch.no_grad():
        xs = (X[probe][ans] - loaded["mu_x"]) / loaded["sd_x"]
        inj = torch.from_numpy(xs @ loaded["W"] + loaded["b"]).float()
        hh = torch.from_numpy(Y[probe]).float().unsqueeze(0).clone()
        hh[0, torch.from_numpy(np.flatnonzero(ans)).long()] = inj
        pos = torch.from_numpy(position[probe]).long().unsqueeze(0)
        out = stack.run(hh.to(stack.dtype), pos, i, stack.n_layers, cache=None)
        got = stack.lm_head(stack.base.norm(out))[0][-1].float().numpy()
    head = np.array(meta["probe_logits_sha_head"], np.float32)
    assert int(got.argmax()) == meta["probe_logits_argmax"], "reloaded argmax differs"
    assert np.abs(got[:8] - head).max() < 1e-3, np.abs(got[:8] - head).max()
    print(f"[ok] saved map reproduces training logits (argmax {int(got.argmax())}, "
          f"max|dlogit|={np.abs(got[:8] - head).max():.2e})")

    # 7. The KL objective is a real divergence: zero at identity, positive apart.
    s = torch.randn(5, TOPK)
    tv = torch.from_numpy(t_vals[:5].astype(np.float32))
    ti = torch.arange(TOPK).repeat(5, 1)
    self_kl = float(distill.kl_to_teacher(tv, tv, ti, ce_weight=0.0))
    cross = float(distill.kl_to_teacher(s, tv, ti, ce_weight=0.0))
    assert abs(self_kl) < 1e-5 and cross > 0, (self_kl, cross)
    print(f"[ok] KL(teacher||teacher)={self_kl:.2e}, KL(random||teacher)={cross:.3f}")

    # 8. When no epoch beats the warm start, the warm start is what ships.
    #    Driven with an absurd learning rate so training is guaranteed to make
    #    things worse. The bug this guards against returned the LAST epoch's
    #    weights while printing "shipping ridge unchanged".
    blown, bmeta = distill.train(
        None, None, i, j, base, lm_small, X, Y, pid, is_answer, position,
        t_vals, t_idx, lm_large=lm_large, epochs=2, lr=5.0, batch_seqs=2,
        val_frac=0.34, verbose=False)
    assert not bmeta["distill_improved_on_ridge"], "expected a blown-up run to not improve"
    assert bmeta["distill_best_epoch"] == 0
    for k in ("W", "b"):
        assert np.allclose(blown[k], base[k], atol=1e-6), \
            f"{k} was not restored to the warm start when no epoch improved"
    print("[ok] no-improvement run restores the ridge warm start exactly")

    # 9. The training forward IS the warm-mode inference forward.
    #    Load-bearing: distillation optimises whatever geometry it is run under,
    #    so if training injects the adapter differently from how decoding does,
    #    the map is fit for a path that never runs — and it would be invisible
    #    in the loss curve.
    assert _training_matches_inference(), "training forward != warm inference forward"
    print("[ok] training forward matches warm-mode inference forward")

    print("\nsmoke_distill_l2s: all checks passed")
    return 0


def _training_matches_inference(tol: float = 1e-2) -> bool:
    from stitching_large_to_small.stitch import LargeToSmallRunner

    lm_small, lm_large = tiny(D_SMALL, N_SMALL, "S"), tiny(D_LARGE, N_LARGE, "L")
    i, j, n_prompt, seq = 3, 4, 5, 9
    ids = torch.randint(0, VOCAB, (1, seq))
    stack = Stack(lm_small)
    pos = torch.arange(seq)[None]
    with torch.no_grad():
        X = lm_large.model(ids, output_hidden_states=True).hidden_states[j]
        Y = lm_small.model(ids, output_hidden_states=True).hidden_states[i]
    W = torch.randn(D_LARGE, D_SMALL) * 0.02
    b = torch.zeros(D_SMALL)
    mu_x, sd_x = X[0].mean(0), X[0].std(0) + 1e-6
    amap = lambda x: ((x - mu_x) / sd_x) @ W + b

    with torch.no_grad():                       # training-time geometry
        h = Y.clone().float()
        ans = torch.arange(n_prompt, seq)
        h[0, ans] = amap(X[0, ans].float())
        out = stack.run(h.to(stack.dtype), pos, i, stack.n_layers, cache=None)
        train_logits = stack.lm_head(stack.base.norm(out))[0]

    class _A:                                   # minimal adapter for the runner
        def __init__(self):
            self.i, self.j, self.n_params = i, j, 0

        def __call__(self, x):
            return amap(x.float())

    runner = LargeToSmallRunner(lm_small, lm_large, _A(), mode="warm")
    with torch.no_grad():                       # inference-time geometry
        got = [runner.prefill(ids[:, :n_prompt])[0]]
        for k in range(n_prompt, seq):
            got.append(runner.step(int(ids[0, k]), k)[0])
    diff = float((torch.stack(got[:-1]) - train_logits[n_prompt - 1:seq - 1]).abs().max())
    print(f"     (max |logit diff| training vs warm inference: {diff:.2e})")
    return diff < tol


if __name__ == "__main__":
    raise SystemExit(main())
