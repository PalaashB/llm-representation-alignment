"""Fit the adapter to the teacher's next-token distribution, not to its hidden states.

Why this exists
---------------
The ridge fit minimises L2 between the adapter's output and the large model's
layer-j residual, weighting all 3072 output dimensions equally. That is not the
quantity the experiment is scored on. What survives to the next token is
whatever clears the final RMSNorm and the unembedding — a small number of
directions — and the residual stream's variance is dominated by directions the
lm_head largely discards. So ridge spends its capacity where it is cheapest to
reduce error, which is not where error costs accuracy.

The repo already measured the symptom without naming the cause: held-out
R2_all = 0.999 against R2_answer between -0.20 and +0.375, and, decisively,
answer-token R2 *does not rank the grid cells by accuracy* in either package's
sweep. A proxy that does not order the thing it proxies for is misaligned, not
merely noisy. Both READMEs record it as "a tripwire for total collapse, not a
selection criterion", which is the same observation from the other side.

Distillation optimises the scored quantity directly: KL from the large model's
own next-token distribution to the stitched path's, backpropagated through the
frozen suffix blocks into the adapter.

What is trained
---------------
Only the adapter. Every LLM parameter has `requires_grad=False`, and
`_assert_only_adapter_trains` checks both that and that the optimiser's
parameter list is exactly the adapter's tensors — the freeze is asserted at the
two places it could break, not assumed.

The training forward reproduces `warm` mode exactly: prompt positions carry the
large model's true layer-j residual, answer positions carry the adapter's
output, and the whole sequence goes through blocks j..end with causal attention.
That matters because it is the mode the accuracy wins come from, and because a
map trained under one injection pattern and run under another is being asked a
different question at inference than it was fit on — the mistake this folder
has already made once, with prompt-only capture.

Warm start
----------
The map is initialised at the ridge solution, so training begins at exactly the
baseline's behaviour and step zero is the baseline's loss. Distillation can
therefore only improve on ridge (up to the early-stopping split's judgement),
which makes the head-to-head comparison a statement about the objective rather
than about initialisation luck.
"""

from __future__ import annotations

import numpy as np
import torch

from common.decoding import Stack
from common.model_utils import LM
from stitching_small_to_large.config import (
    DISTILL_BATCH_SEQS, DISTILL_CE_WEIGHT, DISTILL_EPOCHS, DISTILL_LR,
    DISTILL_MAX_GRAD_NORM, DISTILL_TEMPERATURE, DISTILL_VAL_FRAC,
    DISTILL_WEIGHT_DECAY, SEED, Bank, Pair,
)

TRAINABLE = ("W", "b", "W1", "b1", "W2", "b2")
FROZEN = ("mu_x", "sd_x", "mu_y")


class TrainableAdapter(torch.nn.Module):
    """The saved map as torch parameters, warm-started from a fitted numpy map.

    Splitting the map into trainable and frozen tensors is deliberate. `mu_x`,
    `sd_x` and `mu_y` are standardisation constants estimated from the fit set's
    own moments; they are not degrees of freedom the objective should spend, and
    letting them drift would make the saved map's `apply_map` semantics differ
    from the ridge path's for no gain.
    """

    def __init__(self, m: dict, device: str):
        super().__init__()
        to = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(device, torch.float32)
        self.params = torch.nn.ParameterDict(
            {k: torch.nn.Parameter(to(m[k])) for k in TRAINABLE if k in m})
        for k in FROZEN:
            self.register_buffer(k, to(m[k]))
        self.has_mlp = "W1" in self.params

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = self.params
        xs = (x.to(torch.float32) - self.mu_x) / self.sd_x
        out = xs @ p["W"] + p["b"]
        if self.has_mlp:
            h = torch.nn.functional.gelu(xs @ p["W1"] + p["b1"], approximate="tanh")
            out = out + h @ p["W2"] + p["b2"]
        return out

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
    that a fitted map can redirect a frozen model, not that a 3B model can be
    fine-tuned. Two independent things could break that — a parameter left with
    `requires_grad=True`, or an optimiser handed more than the adapter — so both
    are asserted, and the counts go into the sidecar as evidence.
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
    that was stored (see `data.topk_teacher` for why full vocab is not). The
    student is renormalised over the same K tokens, which keeps the quantity a
    proper KL between two distributions on a shared support.

    The CE term on the teacher's argmax is a small anchor on the token that
    greedy decoding will actually emit. KL alone spreads effort over the whole
    top-K in proportion to teacher mass; at temperature 1 that is mostly the
    argmax anyway, so the term is deliberately weak — it sharpens the thing
    being scored without turning the objective into hard-label training.
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
    """Row index groups, one per captured prompt, ordered by position.

    Whole sequences, not shuffled rows: the suffix blocks attend backwards, so a
    position's logits depend on every earlier position of the same prompt. Rows
    are not independent samples here the way they are for a ridge solve.
    """
    out = []
    for p in np.unique(pid):
        idx = np.flatnonzero(pid == p)
        idx = idx[np.argsort(position[idx], kind="stable")]
        if (is_answer[idx] == 1).any():
            out.append(idx)
    return out


def train(pair: Pair, bank: Bank, i: int, j: int, base_map: dict, lm_large: LM,
          X: np.ndarray, Y: np.ndarray, pid: np.ndarray, is_answer: np.ndarray,
          position: np.ndarray, t_vals: np.ndarray, t_idx: np.ndarray,
          lm_small: LM | None = None, epochs: int = DISTILL_EPOCHS,
          lr: float = DISTILL_LR, batch_seqs: int = DISTILL_BATCH_SEQS,
          weight_decay: float = DISTILL_WEIGHT_DECAY,
          ce_weight: float = DISTILL_CE_WEIGHT,
          temperature: float = DISTILL_TEMPERATURE,
          val_frac: float = DISTILL_VAL_FRAC, seed: int = SEED,
          max_grad_norm: float = DISTILL_MAX_GRAD_NORM,
          verbose: bool = True) -> tuple[dict, dict]:
    """Distil into the adapter. Returns (map arrays, training metadata)."""
    device = lm_large.device
    stack = Stack(lm_large)
    freeze(*(x for x in (lm_large, lm_small) if x is not None))

    adapter = TrainableAdapter(base_map, device)
    opt = torch.optim.AdamW(adapter.parameters(), lr=lr, weight_decay=weight_decay)
    frozen_evidence = _assert_only_adapter_trains(
        [x for x in (lm_large, lm_small) if x is not None], adapter, opt)

    seqs = _sequences(pid, is_answer, position)
    rng = np.random.default_rng(seed + 1)
    order = rng.permutation(len(seqs))
    n_val = max(1, int(round(len(seqs) * val_frac)))
    val_ids = set(order[:n_val].tolist())
    train_seqs = [s for k, s in enumerate(seqs) if k not in val_ids]
    val_seqs = [s for k, s in enumerate(seqs) if k in val_ids]

    to = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(device)

    def seq_loss(idx: np.ndarray) -> torch.Tensor:
        """One sequence's distillation loss, in the `warm` injection pattern."""
        ans = is_answer[idx] == 1
        # Prompt positions keep the large model's own residual; answer positions
        # are replaced by the adapter's reconstruction. This is warm mode.
        h = to(Y[idx]).to(torch.float32).unsqueeze(0)
        xa = to(X[idx][ans]).to(torch.float32)
        h = h.clone()
        h[0, to(np.flatnonzero(ans)).long()] = adapter(xa)
        pos = to(position[idx]).long().unsqueeze(0)
        out = stack.run(h.to(stack.dtype), pos, j, stack.n_layers, cache=None)
        logits = stack.lm_head(stack.base.norm(out))[0]
        sel = logits[to(np.flatnonzero(ans)).long()].float()
        return kl_to_teacher(sel, to(t_vals[idx][ans]), to(t_idx[idx][ans]),
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
    # ridge, which is strictly worse than not distilling and invisible in the
    # log. Measured on llama L8->L14 the difference is held-out answer
    # R2 = +0.611 (ridge, correctly restored) against -2.12 (last epoch).
    best = ridge_val = evaluate(val_seqs)
    best_ep = 0
    best_state = {k: t.detach().clone() for k, t in adapter.state_dict().items()}
    improved = False
    history = [{"epoch": 0, "val_loss": ridge_val, "train_loss": None}]
    if verbose:
        print(f"  [distill] {len(train_seqs)} train / {len(val_seqs)} val sequences, "
              f"warm start val loss {ridge_val:.4f}")

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
        v = evaluate(val_seqs)
        history.append({"epoch": ep, "val_loss": v, "train_loss": tot / max(1, nb)})
        if verbose:
            print(f"  [distill] epoch {ep}: train {tot / max(1, nb):.4f}  val {v:.4f}"
                  f"  (best {min(best, v):.4f})")
        if v < best:
            best, best_ep, improved = v, ep, True
            best_state = {k: t.detach().clone() for k, t in adapter.state_dict().items()}

    # Always restore, including when the winner is epoch 0.
    adapter.load_state_dict(best_state)
    if verbose and not improved:
        print("  [distill] no epoch beat the ridge warm start — restored the warm "
              "start, so this variant ships the ridge map exactly")

    # Fingerprint for the reload check: the logits the trained map produces on a
    # fixed probe, recorded now so `check_distill_reload` can prove the saved
    # file reproduces what training ended at.
    probe = (val_seqs or train_seqs)[0]
    with torch.no_grad():
        ans = is_answer[probe] == 1
        h = to(Y[probe]).to(torch.float32).unsqueeze(0).clone()
        h[0, to(np.flatnonzero(ans)).long()] = adapter(to(X[probe][ans]).to(torch.float32))
        pos = to(position[probe]).long().unsqueeze(0)
        out = stack.run(h.to(stack.dtype), pos, j, stack.n_layers, cache=None)
        probe_logits = stack.lm_head(stack.base.norm(out))[0][-1].float().cpu().numpy()

    meta = {
        "train_method": "distill",
        "distill_epochs": epochs, "distill_lr": lr,
        "distill_batch_seqs": batch_seqs, "distill_ce_weight": ce_weight,
        "distill_temperature": temperature, "distill_topk": int(t_idx.shape[1]),
        "distill_val_frac": val_frac,
        "distill_n_train_seqs": len(train_seqs), "distill_n_val_seqs": len(val_seqs),
        "distill_val_loss_ridge": ridge_val, "distill_val_loss_best": best,
        "distill_best_epoch": best_ep,
        "distill_improved_on_ridge": improved,
        "distill_history": history,
        "frozen_llm_evidence": frozen_evidence,
        "probe_prompt_row0": int(probe[0]),
        "probe_logits_sha_head": [float(x) for x in probe_logits[:8]],
        "probe_logits_argmax": int(probe_logits.argmax()),
    }
    return adapter.to_numpy(), meta
