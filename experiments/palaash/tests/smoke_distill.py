"""Synthetic smoke test for the distillation training path — no real weights.

Builds a tiny randomly-initialised Llama pair on CPU and drives the machinery
that the real run depends on: the trainable adapter, the KL objective, the
frozen-LLM assertions, the warm start, and the saved-map reload check. The
point is to fail in seconds here rather than thirty minutes into an MPS run.

What it actually proves, and what it does not: the plumbing is exercised
end-to-end and the invariants (nothing but the adapter moves, the loss goes
down, the saved file reproduces the training-time logits) are checked. Nothing
here says distillation helps on real models — random weights have no next-token
structure to learn. That question is answered by the head-to-head bench.

    python tests/smoke_distill.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transformers import LlamaConfig, LlamaForCausalLM, AutoTokenizer  # noqa: E402

from common.decoding import Stack, check_full_stack  # noqa: E402
from common.model_utils import LM  # noqa: E402
from stitching_small_to_large import distill  # noqa: E402

VOCAB, D_SMALL, D_LARGE, N_SMALL, N_LARGE = 128, 32, 48, 4, 6
SEQ, N_SEQ, TOPK = 12, 6, 8


def tiny(dim: int, layers: int, tag: str) -> LM:
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=VOCAB, hidden_size=dim, intermediate_size=dim * 2,
                      num_hidden_layers=layers, num_attention_heads=4,
                      num_key_value_heads=4, max_position_embeddings=64)
    model = LlamaForCausalLM(cfg).eval()
    return LM(tag=tag, tokenizer=None, model=model, device="cpu")


def main() -> int:
    lm_small, lm_large = tiny(D_SMALL, N_SMALL, "S"), tiny(D_LARGE, N_LARGE, "L")
    i, j = 2, 3
    rng = np.random.default_rng(0)

    # The sliced layer loop must reproduce HF's own forward before anything
    # built on top of it means anything.
    ids = torch.randint(0, VOCAB, (1, SEQ))
    chk = check_full_stack(lm_large, ids, "large")
    assert chk["passed"], f"full_stack check failed: {chk}"
    print(f"[ok] full_stack rel_l2={chk['rel_l2']:.2e}")

    # Synthetic capture: X taps, Y true residuals, teacher top-K, row metadata.
    n = SEQ * N_SEQ
    X = rng.normal(size=(n, D_SMALL)).astype(np.float32)
    Y = rng.normal(size=(n, D_LARGE)).astype(np.float32)
    pid = np.repeat(np.arange(N_SEQ), SEQ).astype(np.int32)
    position = np.tile(np.arange(SEQ), N_SEQ).astype(np.int32)
    is_answer = (position >= SEQ // 2).astype(np.int8)
    t_idx = rng.integers(0, VOCAB, size=(n, TOPK)).astype(np.int32)
    t_vals = np.sort(rng.normal(size=(n, TOPK)).astype(np.float16), axis=1)[:, ::-1].copy()

    base = {"W": rng.normal(scale=0.02, size=(D_SMALL, D_LARGE)).astype(np.float32),
            "b": np.zeros(D_LARGE, np.float32),
            "mu_x": X.mean(0), "sd_x": X.std(0) + 1e-6, "mu_y": Y.mean(0)}

    m, meta = distill.train(
        None, None, i, j, base, lm_large, X, Y, pid, is_answer, position,
        t_vals, t_idx, lm_small=lm_small, epochs=3, lr=1e-2, batch_seqs=2,
        val_frac=0.34, verbose=False)

    # 1. Nothing but the adapter was trainable.
    ev = meta["frozen_llm_evidence"]
    assert ev["llm_params_trainable"] == 0, ev
    assert ev["adapter_tensors_trained"] == 2, ev          # W and b only
    print(f"[ok] frozen: {ev['llm_params_total']} LLM tensors, 0 trainable; "
          f"{ev['adapter_params_trained']} adapter params trained")

    # 2. No LLM weight actually moved (belt and braces on the assertion above).
    after = torch.cat([p.flatten() for p in lm_large.model.parameters()])
    ref = torch.cat([p.flatten() for p in tiny(D_LARGE, N_LARGE, "L").model.parameters()])
    assert torch.equal(after, ref), "a large-model weight changed during distillation"
    print("[ok] large-model weights bit-identical after training")

    # 3. Warm start: epoch 0 is the ridge map, and training improved on it.
    h = meta["distill_history"]
    assert h[0]["epoch"] == 0 and h[0]["train_loss"] is None
    assert meta["distill_val_loss_best"] <= meta["distill_val_loss_ridge"] + 1e-9
    print(f"[ok] warm start {meta['distill_val_loss_ridge']:.4f} -> "
          f"{meta['distill_val_loss_best']:.4f} @ epoch {meta['distill_best_epoch']} "
          f"(improved: {meta['distill_improved_on_ridge']})")

    # 4. The map changed shape-compatibly and is finite.
    assert m["W"].shape == (D_SMALL, D_LARGE) and np.isfinite(m["W"]).all()

    # 5. The saved arrays reproduce the training-time logits. Replays the same
    #    probe the real check_adapter_reload replays, through a numpy round-trip.
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "a.npz"
        np.savez(p, **m)
        loaded = {k: v for k, v in np.load(p).items()}
    stack = Stack(lm_large)
    probe = np.flatnonzero(pid == pid[meta["probe_prompt_row0"]])
    probe = probe[np.argsort(position[probe], kind="stable")]
    ans = is_answer[probe] == 1
    with torch.no_grad():
        xs = (X[probe][ans] - loaded["mu_x"]) / loaded["sd_x"]
        inj = torch.from_numpy(xs @ loaded["W"] + loaded["b"]).float()
        hh = torch.from_numpy(Y[probe]).float().unsqueeze(0).clone()
        hh[0, torch.from_numpy(np.flatnonzero(ans)).long()] = inj
        pos = torch.from_numpy(position[probe]).long().unsqueeze(0)
        out = stack.run(hh.to(stack.dtype), pos, j, stack.n_layers, cache=None)
        got = stack.lm_head(stack.base.norm(out))[0][-1].float().numpy()
    head = np.array(meta["probe_logits_sha_head"], np.float32)
    assert int(got.argmax()) == meta["probe_logits_argmax"], "reloaded argmax differs"
    assert np.abs(got[:8] - head).max() < 1e-3, np.abs(got[:8] - head).max()
    print(f"[ok] saved map reproduces training logits "
          f"(argmax {int(got.argmax())}, max|dlogit|="
          f"{np.abs(got[:8] - head).max():.2e})")

    # 6. The KL objective is a real divergence: zero at identity, positive apart.
    s = torch.randn(5, TOPK)
    tv = torch.from_numpy(t_vals[:5].astype(np.float32))
    ti = torch.arange(TOPK).repeat(5, 1)
    self_kl = float(distill.kl_to_teacher(tv, tv, ti, ce_weight=0.0))
    cross = float(distill.kl_to_teacher(s, tv, ti, ce_weight=0.0))
    assert abs(self_kl) < 1e-5 and cross > 0, (self_kl, cross)
    print(f"[ok] KL(teacher||teacher)={self_kl:.2e}, KL(random||teacher)={cross:.3f}")

    # 6b. Regression: when no epoch beats the warm start, the warm start is what
    #     ships. Driven with an absurd learning rate so training is guaranteed to
    #     make things worse. The bug this guards against returned the LAST
    #     epoch's weights while printing "shipping ridge unchanged" — on the real
    #     llama run that was held-out answer R2 = -2.12 shipped in place of
    #     +0.611, with the log claiming the opposite.
    blown, bmeta = distill.train(
        None, None, i, j, base, lm_large, X, Y, pid, is_answer, position,
        t_vals, t_idx, lm_small=lm_small, epochs=2, lr=5.0, batch_seqs=2,
        val_frac=0.34, verbose=False)
    assert not bmeta["distill_improved_on_ridge"], "expected a blown-up run to not improve"
    assert bmeta["distill_best_epoch"] == 0
    for k in ("W", "b"):
        assert np.allclose(blown[k], base[k], atol=1e-6), \
            f"{k} was not restored to the warm start when no epoch improved"
    print("[ok] no-improvement run restores the ridge warm start exactly")

    # 7. The training forward IS the warm-mode inference forward.
    #    Load-bearing: distillation optimises whatever geometry it is run under,
    #    so if training injects the adapter differently from how decoding does,
    #    the map is fit for a path that never runs. This is the same class of bug
    #    as the original prompt-only capture, and it would be invisible in the
    #    loss curve.
    assert _training_matches_inference(), "training forward != warm inference forward"
    print("[ok] training forward matches warm-mode inference forward")

    print("\nsmoke_distill: all checks passed")
    return 0


def _training_matches_inference(tol: float = 1e-2) -> bool:
    from stitching_small_to_large.stitch import StitchRunner

    lm_small, lm_large = tiny(D_SMALL, N_SMALL, "S"), tiny(D_LARGE, N_LARGE, "L")
    i, j, n_prompt, seq = 2, 3, 5, 9
    ids = torch.randint(0, VOCAB, (1, seq))
    stack = Stack(lm_large)
    pos = torch.arange(seq)[None]
    with torch.no_grad():
        Y = lm_large.model(ids, output_hidden_states=True).hidden_states[j]
        X = lm_small.model(ids, output_hidden_states=True).hidden_states[i]
    W = torch.randn(D_SMALL, D_LARGE) * 0.02
    b = torch.zeros(D_LARGE)
    mu_x, sd_x = X[0].mean(0), X[0].std(0) + 1e-6
    amap = lambda x: ((x - mu_x) / sd_x) @ W + b

    with torch.no_grad():                       # training-time geometry
        h = Y.clone().float()
        ans = torch.arange(n_prompt, seq)
        h[0, ans] = amap(X[0, ans].float())
        out = stack.run(h.to(stack.dtype), pos, j, stack.n_layers, cache=None)
        train_logits = stack.lm_head(stack.base.norm(out))[0]

    class _A:                                   # minimal adapter for the runner
        def __init__(self):
            self.i, self.j, self.kind, self.taps, self.n_params = i, j, "linear", (i,), 0

        def __call__(self, x):
            return amap(x.float())

    runner = StitchRunner(lm_small, lm_large, _A(), mode="warm")
    with torch.no_grad():                       # inference-time geometry
        got = [runner.prefill(ids[:, :n_prompt])[0]]
        for k in range(n_prompt, seq):
            got.append(runner.step(int(ids[0, k]), k)[0])
    diff = float((torch.stack(got[:-1]) - train_logits[n_prompt - 1:seq - 1]).abs().max())
    print(f"     (max |logit diff| training vs warm inference: {diff:.2e})")
    return diff < tol


if __name__ == "__main__":
    raise SystemExit(main())
