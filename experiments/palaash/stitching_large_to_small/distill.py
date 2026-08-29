"""Fit the adapter to the large model's next-token distribution, not to the
small model's hidden states.

Why this exists
---------------
The ridge fit minimises L2 between the adapter's output and the *small* model's
layer-i residual. In this direction that is not merely a misaligned proxy, it is
self-defeating, and the failure it produces is the one on record in the README.

Two separate things are wrong with it:

1. **It optimises a quantity the path is not scored on.** What survives to the
   next token is whatever clears the final RMSNorm and the unembedding — a small
   number of directions — while the residual stream's variance is dominated by
   directions the lm_head largely discards. So ridge spends its capacity where
   error is cheapest to reduce, which is not where error costs accuracy. The
   symptom is on disk: held-out R2_all = 0.9999 against R2_answer = 0.427, and
   answer-token R2 does not rank the grid cells by accuracy (L14->L10 has the
   best R2 at 0.498 and scores 34.9%; L14->L8 has 0.427 and scores 57.5%).

2. **Its target is the wrong state.** The regression target is the residual the
   small model would have produced *by itself*. On a divergent prompt — small
   wrong, large right, the only prompts this experiment can gain on — that state
   is precisely the one that decodes to the wrong answer. A map that hits its
   target perfectly reproduces the small model's mistake, and the measured
   generations do exactly that: plutonium 92 (small) instead of 94 (large).
   No amount of R2 fixes this, because the target itself carries the error.

Distillation replaces the target as well as the objective: KL from the *large*
model's own next-token distribution to the stitched path's, backpropagated
through the frozen small-model suffix into the adapter. The quantity being
minimised is now a property of the answer the stitch emits, and the thing being
matched is the model that gets the answer right.

What is trained
---------------
Only the adapter. Every LLM parameter has `requires_grad=False`, and
`_assert_only_adapter_trains` checks both that and that the optimiser's
parameter list is exactly the adapter's tensors — the freeze is asserted at the
two places it could break, not assumed. `mu_x`, `sd_x` and `mu_y` are buffers,
not parameters: they are standardisation constants estimated from the fit set's
own moments, not degrees of freedom the objective should spend.

The training forward reproduces the inference geometry of `train_mode` exactly.
For `warm` (the default) that means prompt positions carry the small model's own
layer-i residual — the same states its own prefill would put in the KV cache —
and only answer positions carry the adapter's output. A map trained under one
injection pattern and run under another is being asked a different question at
inference than it was fit on.

Which positions the loss covers, and why it is not all of them
-------------------------------------------------------------
In `warm` mode the first generated token is decoded from the *last prompt
position*, which the adapter never touches: the small model prefills the prompt
itself. So that token is the small model's own, always, and no adapter can
change it. The loss therefore covers answer positions only — exactly the
positions where the adapter is in the loop.

That is also a hard ceiling on what `warm` can achieve on this bank, and it is
worth stating plainly rather than discovering in the results: when the whole
answer is one token ("94"), `warm` cannot fix it even in principle. `exit` mode
puts the adapter on every position, so its loss adds the final prompt position —
the one that emits the first answer token.
"""

from __future__ import annotations

import numpy as np
import torch

from common.decoding import Stack
from common.model_utils import LM
from stitching_large_to_small.config import (
    DISTILL_BATCH_SEQS, DISTILL_CE_WEIGHT, DISTILL_EPOCHS, DISTILL_LR,
    DISTILL_MAX_GRAD_NORM, DISTILL_TEMPERATURE, DISTILL_TRAIN_MODE,
    DISTILL_TRAIN_MODES, DISTILL_VAL_FRAC, DISTILL_WEIGHT_DECAY, PRESERVE_PREFIX,
    SEED, Bank, Pair,
)

TRAINABLE = ("W", "b")
FROZEN = ("mu_x", "sd_x", "mu_y")


class TrainableAdapter(torch.nn.Module):
    """The saved map as torch parameters, warm-started from a fitted numpy map.

    Splitting the map into trainable and frozen tensors is deliberate. `mu_x`,
    `sd_x` and `mu_y` are standardisation constants; letting them drift would
    make the saved map's `apply_np` semantics differ from the ridge path's for
    no gain, and they are not what the objective should be spending on.
    """

    def __init__(self, m: dict, device: str):
        super().__init__()
        to = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(device, torch.float32)
        self.params = torch.nn.ParameterDict(
            {k: torch.nn.Parameter(to(m[k])) for k in TRAINABLE if k in m})
        for k in FROZEN:
            self.register_buffer(k, to(m[k]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = self.params
        return ((x.to(torch.float32) - self.mu_x) / self.sd_x) @ p["W"] + p["b"]

    def to_numpy(self) -> dict:
        out = {k: v.detach().cpu().numpy().astype(np.float32)
               for k, v in self.params.items()}
        for k in FROZEN:
            out[k] = getattr(self, k).detach().cpu().numpy().astype(np.float32)
        return out


def _assert_only_adapter_trains(models: list[LM], adapter: TrainableAdapter,
                                opt: torch.optim.Optimizer) -> dict:
    """Both halves of "the LLMs are frozen", checked rather than trusted.

    A stitching result is only interesting if no LLM weight moved: the claim is
    that a fitted map can redirect a frozen model, not that a 1B model can be
    fine-tuned on a 3B teacher. Two independent things could break that — a
    parameter left with `requires_grad=True`, or an optimiser handed more than
    the adapter — so both are asserted, and the counts go into the sidecar as
    evidence.
    """
    llm_params = [p for lm in models for p in lm.model.parameters()]
    unfrozen = sum(p.requires_grad for p in llm_params)
    if unfrozen:
        raise SystemExit(f"{unfrozen} LLM parameters still have requires_grad=True; "
                         f"the stitch must train the adapter only.")
    want = {id(p) for p in adapter.parameters()}
    got = {id(p) for g in opt.param_groups for p in g["params"]}
    if want != got:
        raise SystemExit(f"optimiser parameter set is not exactly the adapter's "
                         f"({len(got)} tensors vs {len(want)}).")
    return {"llm_params_total": len(llm_params), "llm_params_trainable": 0,
            "adapter_tensors_trained": len(want),
            "adapter_params_trained": sum(p.numel() for p in adapter.parameters())}


def freeze(*lms: LM) -> None:
    for lm in lms:
        lm.model.eval()
        for p in lm.model.parameters():
            p.requires_grad_(False)


def kl_to_teacher(student_logits: torch.Tensor, t_vals: torch.Tensor,
                  t_idx: torch.Tensor, temperature: float = DISTILL_TEMPERATURE,
                  ce_weight: float = DISTILL_CE_WEIGHT) -> torch.Tensor:
    """KL(teacher || student) over the teacher's top-K support, plus a CE term.

    The teacher distribution is renormalised over its own top-K rather than
    compared against the student's full vocabulary, because that is the support
    that was stored (see `data.topk_teacher`). The student is renormalised over
    the same K tokens, which keeps the quantity a proper KL between two
    distributions on a shared support. Both models share a tokenizer, so the
    stored ids index the student's vocabulary directly.

    The CE term on the teacher's argmax is a small anchor on the token greedy
    decoding will actually emit. KL alone spreads effort over the whole top-K in
    proportion to teacher mass; at temperature 1 that is mostly the argmax
    anyway, so the term is deliberately weak — it sharpens the thing being
    scored without turning the objective into hard-label training.
    """
    sel = student_logits.gather(-1, t_idx.long())
    t_log_p = torch.log_softmax(t_vals.float() / temperature, dim=-1)
    s_log_p = torch.log_softmax(sel / temperature, dim=-1)
    kl = (t_log_p.exp() * (t_log_p - s_log_p)).sum(-1).mean()
    if ce_weight <= 0:
        return kl
    # The teacher's argmax within its own top-K is index 0: topk returns sorted.
    ce = torch.nn.functional.cross_entropy(sel, torch.zeros(
        len(sel), dtype=torch.long, device=sel.device))
    return kl + ce_weight * ce


def _sequences(pid: np.ndarray, is_answer: np.ndarray, position: np.ndarray):
    """Row index groups, one per captured item, ordered by position.

    Whole sequences, not shuffled rows: the suffix blocks attend backwards, so a
    position's logits depend on every earlier position of the same sequence.
    Rows are not independent samples here the way they are for a ridge solve.
    """
    out = []
    for p in np.unique(pid):
        idx = np.flatnonzero(pid == p)
        idx = idx[np.argsort(position[idx], kind="stable")]
        if (is_answer[idx] == 1).any():
            out.append(idx)
    return out


def train(pair: Pair, bank: Bank, i: int, j: int, base_map: dict, lm_small: LM,
          X: np.ndarray, Y: np.ndarray, pid: np.ndarray, is_answer: np.ndarray,
          position: np.ndarray, t_vals: np.ndarray, t_idx: np.ndarray,
          lm_large: LM | None = None, epochs: int = DISTILL_EPOCHS,
          lr: float = DISTILL_LR, batch_seqs: int = DISTILL_BATCH_SEQS,
          weight_decay: float = DISTILL_WEIGHT_DECAY,
          ce_weight: float = DISTILL_CE_WEIGHT,
          temperature: float = DISTILL_TEMPERATURE,
          val_frac: float = DISTILL_VAL_FRAC, seed: int = SEED,
          max_grad_norm: float = DISTILL_MAX_GRAD_NORM,
          train_mode: str = DISTILL_TRAIN_MODE,
          preserve_prefix: int = PRESERVE_PREFIX,
          max_seqs: int | None = None,
          verbose: bool = True) -> tuple[dict, dict]:
    """Distil into the adapter. Returns (map arrays, training metadata).

    `lm_small` is the model the loss flows through: this direction's suffix is
    the *small* model's blocks i..end, so it is the small model that must be
    loaded, frozen, and differentiated through. The large model is not needed
    at all — its contribution is the states in `X` and the distributions in
    `t_vals`, both captured offline — but it is asserted frozen when it happens
    to be loaded.
    """
    if train_mode not in DISTILL_TRAIN_MODES:
        raise SystemExit(f"--distill-train-mode must be one of {DISTILL_TRAIN_MODES}")
    device = lm_small.device
    stack = Stack(lm_small)
    freeze(*(x for x in (lm_small, lm_large) if x is not None))

    adapter = TrainableAdapter(base_map, device)
    opt = torch.optim.AdamW(adapter.parameters(), lr=lr, weight_decay=weight_decay)
    frozen_evidence = _assert_only_adapter_trains(
        [x for x in (lm_small, lm_large) if x is not None], adapter, opt)

    seqs = _sequences(pid, is_answer, position)
    rng = np.random.default_rng(seed + 1)
    order = rng.permutation(len(seqs))
    n_val = max(1, int(round(len(seqs) * val_frac)))
    val_ids = set(order[:n_val].tolist())
    train_seqs = [s for k, s in enumerate(seqs) if k not in val_ids]
    val_seqs = [s for k, s in enumerate(seqs) if k in val_ids]
    if max_seqs:                      # smoke runs only; recorded in the sidecar
        train_seqs = train_seqs[:max_seqs]
        val_seqs = val_seqs[:max(1, max_seqs // 4)]

    to = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(device)

    def _masks(idx: np.ndarray):
        """(positions the adapter writes, positions the loss reads).

        `warm`: the adapter writes answer positions only, and those are exactly
        the positions whose logits it can move. `exit`: it writes everything
        except the preserved sink prefix, so the final prompt position — the one
        that emits the first answer token — becomes a loss position too.
        """
        ans = is_answer[idx] == 1
        if train_mode == "warm":
            return ans, ans
        inject = np.ones(len(idx), bool)
        inject[:min(preserve_prefix, len(idx))] = False
        loss = ans.copy()
        first = int(np.argmax(ans)) if ans.any() else 0
        if first > 0:
            loss[first - 1] = True
        return inject, loss

    def seq_loss(idx: np.ndarray) -> torch.Tensor:
        """One sequence's distillation loss, in `train_mode`'s injection pattern."""
        inject, loss_at = _masks(idx)
        # Non-injected positions keep the small model's own residual — the same
        # states its own forward would have produced, which is what its KV cache
        # holds at those positions during inference.
        h = to(Y[idx]).to(torch.float32).unsqueeze(0).clone()
        h[0, to(np.flatnonzero(inject)).long()] = adapter(to(X[idx][inject]).to(torch.float32))
        pos = to(position[idx]).long().unsqueeze(0)
        out = stack.run(h.to(stack.dtype), pos, i, stack.n_layers, cache=None)
        logits = stack.lm_head(stack.base.norm(out))[0]
        sel = logits[to(np.flatnonzero(loss_at)).long()].float()
        return kl_to_teacher(sel, to(t_vals[idx][loss_at]), to(t_idx[idx][loss_at]),
                             temperature, ce_weight)

    @torch.no_grad()
    def evaluate(which) -> float:
        tot = 0.0
        for idx in which:
            tot += float(seq_loss(idx))
        return tot / max(1, len(which))

    # Epoch 0 is the ridge map exactly (warm start), so it is a real candidate:
    # if no epoch beats it, ridge is what ships and the sidecar says so.
    #
    # The warm start is snapshotted here rather than left implicit. Selecting
    # "epoch 0" has to mean *restoring* those weights, because the adapter object
    # goes on being mutated by the optimiser afterwards — without this the run
    # ships whatever the last epoch produced while reporting that it shipped
    # ridge, which is strictly worse than not distilling and invisible in the log.
    best = ridge_val = evaluate(val_seqs)
    best_ep = 0
    best_state = {k: t.detach().clone() for k, t in adapter.state_dict().items()}
    improved = False
    history = [{"epoch": 0, "val_loss": ridge_val, "train_loss": None}]
    if verbose:
        print(f"  [distill] mode={train_mode}  {len(train_seqs)} train / "
              f"{len(val_seqs)} val sequences, warm start val loss {ridge_val:.4f}")

    for ep in range(1, epochs + 1):
        perm = rng.permutation(len(train_seqs))
        tot, nb = 0.0, 0
        opt.zero_grad(set_to_none=True)
        for k, si in enumerate(perm, 1):
            loss = seq_loss(train_seqs[si]) / batch_seqs
            loss.backward()
            tot += float(loss.detach()) * batch_seqs
            nb += 1
            if k % batch_seqs == 0 or k == len(perm):
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), max_grad_norm)
                opt.step()
                opt.zero_grad(set_to_none=True)
            if verbose and k % 200 == 0:
                print(f"  [distill]   epoch {ep} {k}/{len(perm)} seqs, "
                      f"running train {tot / max(1, nb):.4f}", flush=True)
        v = evaluate(val_seqs)
        history.append({"epoch": ep, "val_loss": v, "train_loss": tot / max(1, nb)})
        if verbose:
            print(f"  [distill] epoch {ep}: train {tot / max(1, nb):.4f}  val {v:.4f}"
                  f"  (best {min(best, v):.4f})", flush=True)
        if v < best:
            best, best_ep, improved = v, ep, True
            best_state = {k: t.detach().clone() for k, t in adapter.state_dict().items()}

    # Always restore, including when the winner is epoch 0.
    adapter.load_state_dict(best_state)
    if verbose and not improved:
        print("  [distill] no epoch beat the ridge warm start — restored the warm "
              "start, so this variant ships the ridge map exactly")

    # Fingerprint for the reload check: the logits the trained map produces on a
    # fixed probe, recorded now so the reload check can prove the saved file
    # reproduces what training ended at.
    probe = (val_seqs or train_seqs)[0]
    with torch.no_grad():
        inject, _ = _masks(probe)
        h = to(Y[probe]).to(torch.float32).unsqueeze(0).clone()
        h[0, to(np.flatnonzero(inject)).long()] = adapter(
            to(X[probe][inject]).to(torch.float32))
        pos = to(position[probe]).long().unsqueeze(0)
        out = stack.run(h.to(stack.dtype), pos, i, stack.n_layers, cache=None)
        probe_logits = stack.lm_head(stack.base.norm(out))[0][-1].float().cpu().numpy()

    meta = {
        "train_method": "distill",
        "distill_train_mode": train_mode,
        "distill_epochs": epochs, "distill_lr": lr,
        "distill_batch_seqs": batch_seqs, "distill_ce_weight": ce_weight,
        "distill_temperature": temperature, "distill_topk": int(t_idx.shape[1]),
        "distill_val_frac": val_frac,
        "distill_n_train_seqs": len(train_seqs), "distill_n_val_seqs": len(val_seqs),
        "distill_val_loss_ridge": ridge_val, "distill_val_loss_best": best,
        "distill_best_epoch": best_ep,
        "distill_improved_on_ridge": improved,
        "distill_history": history,
        "distill_max_seqs": max_seqs,
        "frozen_llm_evidence": frozen_evidence,
        "probe_prompt_row0": int(probe[0]),
        "probe_logits_sha_head": [float(x) for x in probe_logits[:8]],
        "probe_logits_argmax": int(probe_logits.argmax()),
    }
    return adapter.to_numpy(), meta
