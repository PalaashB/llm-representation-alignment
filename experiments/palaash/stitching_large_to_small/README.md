# stitching_large_to_small — ACCURACY: large early layers → small late layers

**Direction:** large → small. **Goal:** make a small model answer *more
accurately* without fine-tuning either LLM — only a fitted adapter.

For the opposite direction — small → large, aimed at *latency* — see the sibling
package [`stitching_small_to_large/`](../stitching_small_to_large/README.md).
A number from one says nothing about the other.

```
prompt -> large embed + large blocks 0..j-1 -> adapter -> small blocks i..end
       -> small norm -> small lm_head -> token
```

The large model does the reading, the small model does the writing. The question
is whether the large model's mid-stack residual carries enough of the *answer*
that the small model's late blocks can decode it.

> **Status: negative, and the cause is now isolated.** 32 all-ridge cells failed
> in 2026-08-17. The adapter had two real defects in how it was fit; both were
> repaired in 2026-08-23 and the repair is worth **+7 to +10 accuracy points**
> — and the best configuration still loses to running the 1B alone. The binding
> constraint is overwrite-and-continue into a frozen small model, not the
> adapter's objective. See [the 2026-08-23 section](#the-adapter-was-fixed-it-did-not-fix-the-experiment-2026-08-23)
> before proposing another adapter variant.

**This is not a latency win and must not be read as one.** The path runs `j`
large blocks *plus* `n_small - i` small blocks, so it is slower than the small
model alone (~1.9x on llama) and slower than the large model is worth. Latency is
reported anyway, so the cost cannot be quietly dropped from the story.

## Quickstart

```bash
cd experiments/palaash

python -m stitching_large_to_small.run headroom --pair llama          # FIRST

# capture: bank prompts + a generic corpus, varied templates, teacher logits
# ~30 min, ~1.8 GB, 911 items -> 89.5k rows / 26.9k answer rows
python -m stitching_large_to_small.run capture --pair llama \
    --fit-corpus generic --corpus-prompts 700 --corpus-answer-tokens 48 \
    --small-layers 8 10 --large-layers 14 18

# two ways to fit the same map shape
python -m stitching_large_to_small.run fit   --pair llama --i 8 --j 14 --train-method ridge
python -m stitching_large_to_small.run fit   --pair llama --i 8 --j 14 --train-method distill
python -m stitching_large_to_small.run check --pair llama --i 8 --j 14 --train-method distill
python -m stitching_large_to_small.run bench --pair llama --i 8 --j 14 \
    --modes warm --split dev --train-method distill

# head-to-head from disk, no GPU
python -m stitching_large_to_small.run compare --pair llama --i 8 --j 14 --modes warm
```

Everything on disk is scoped by `(pair, bank, train method)`, so runs never
overwrite each other. Token-aligned pairs only (`llama`, `qwen`) — the adapter
is fit position-by-position, which needs both models to see identical tokens.

`--train-method legacy` reads the unscoped artefacts of the original all-ridge
sweep (`adapter_i08_j14.npz`, `bench_i08_j14_dev_warm.json`, …). Those files are
read-only: no fit can write to them, so the published negative result stays on
disk exactly as it was. The capture they were fit on is preserved under
`states_legacy/`.

## The success bar, and why it is interval-based

A configuration counts as useful only if **`accuracy_stitch > accuracy_small`
with non-overlapping 95% Wilson intervals**. Three consequences worth being
explicit about:

* **The small model is the incumbent, not the large one.** This direction runs
  `j` large blocks *plus* `n_small - i` small ones, so it is strictly slower
  than the small model. There is no speed axis to trade against — a
  configuration that is slower and not more accurate is dominated outright.
* **A point estimate above the small model is not a win.** `verdict` has a third
  outcome, `INCONCLUSIVE`, for exactly that case. Reporting an unresolved
  measurement as a success is the error the sibling package made when it read a
  65.7% dev / 85.7% test swing on one unchanged adapter as a 20-point effect.
* **The cost is reported end-to-end, not just per token.** "1.7x the small
  model's ms/token" understates a path that also pays a full large-model prefix
  prefill; measured end-to-end the best cell costs **1.89x**, not 1.72x.

`headroom` gates on interval separation and on split size too, not just on the
raw gap: `factual` offers 8.6 points on a 35-prompt split whose interval is ~23
points wide, so no result on it could ever have been significant.

## Run `headroom` first

A large→small stitch buys back prompts **the small model got wrong and the large
model got right**. A bank without many such prompts cannot show the effect no
matter how good the adapter is, and a null result on it would be a fact about
the bank, not about the method.

`headroom` reports accuracy for both models alone, the gap, and the count of
divergent prompts, then says `usable: YES/NO` against
`MIN_HEADROOM_PTS` (8) and `MIN_DIVERGENT_PROMPTS` (20).

It also prints every prompt **both** models got wrong. Those are the audit queue:
when two models of different sizes agree on an answer the bank calls wrong, the
gold answer is the likely error. This caught three real bank bugs while
`common/prompts_hard_factual.py` was being built — an ambiguous haiku question
(both models answered the per-line syllable count), a missing `neutrophils` alias
for "which blood cells fight infection", and Panama, where the balboa is official
but the US dollar is the circulating legal tender.

## The bank

`common/prompts_hard_factual.py` (`--bank hard_factual`, the default) exists
because the older banks could not support this experiment:

| bank | llama small | llama large | headroom | divergent on dev |
| --- | --- | --- | --- | --- |
| `factual` | 88.6% | 97.1% | 8.6 pts | 3 |
| `list` | 73.3% | 91.1% | 17.8 pts | 9 |
| `hard_factual` | 73.6% | 98.1% | **24.5 pts** | **26** |

It keeps answers short (a name, a symbol, a number) and buys difficulty from
obscurity rather than from length, so the measurement isolates *recall transfer*
rather than mixing it with keeping a long list straight.

Its composition was chosen **empirically, not by intuition**. A first draft
scored +13.8 pts with only 13 divergent prompts; per-category divergence rates
from that run showed where the signal actually was:

| category | divergence rate on llama/dev |
| --- | --- |
| dated events (`In what year …`) | 67% |
| atomic numbers | 50% |
| US state capitals | 19% |
| currencies | 7% |
| **national capitals** | **0%** |
| **element symbols** | **0%** |

Capitals and element symbols contribute nothing — a 1B model knows them cold. So
the bank was expanded along the two productive axes (atomic numbers 26 → 70,
dated events 15 → 45), which took it to 422 prompts, 24.5 points, and 26
divergent cases.

## Plumbing checks gate every report

A mis-plumbed injection still produces fluent text, so it does not announce
itself — it just quietly invalidates the accuracy numbers. `bench` and `sweep`
run the checks first and **refuse to report** unless they pass (`--skip-checks`
overrides, and labels the numbers untrustworthy).

| check | what it rules out |
| --- | --- |
| `full_stack_small` / `full_stack_large` | the hand-rolled sliced layer loop disagreeing with HF's own `forward` |
| `baseline_matches_hf_generate_small` | the baseline being a different decoding harness from the stitch |
| `identity_injection` | **the convention being off by one.** Feeding the small model its OWN layer-`i` residual at block `i` must reproduce the unmodified model exactly. Fails if `hidden_states[i]` is taken after the block instead of before, or if the resumed slice is wrong |
| `offbyone_control` | `identity_injection` passing vacuously. Feeding layer `i+1`'s residual at block `i` must *differ* — if it does not, the path is ignoring what it was handed |
| `prefix_exact` | the attention-sink splice being approximate rather than bit-identical to HF's own prefix forward |
| `adapter_reload` | a map that was saved or reloaded wrong (shapes, dtype, non-finite values) |

Measured on llama at `i=10, j=18`: identity injection `rel_l2 = 3.4e-05` with
matching argmax; off-by-one control `rel_l2 = 0.373` with a *different* argmax.
Both are what they should be.

## Two modes

The decode step is identical; they differ in what the small suffix blocks attend
back to.

- **`exit`** — the prompt goes through the stitch too, so the small model's
  blocks `0..i-1` run over the sink position only and every small KV is the
  adapter's reconstruction of the large model's reading of the prompt. This is
  the mode that could actually transfer the large model's comprehension.
- **`warm`** — the small model prefills the prompt itself, so its prompt KVs are
  its own and the adapter supplies the residual only at generated positions.

## The adapter was fixed. It did not fix the experiment. (2026-08-23)

The 2026-08-17 sweep below had two defects in how the adapter was *fit*, and
both are now repaired. The repair is real and measurable, and it does not
change the conclusion — which is the finding.

**Defect 1 — the fit distribution.** Capture ran the 211 fit prompts under one
chat template, yielding 753 answer rows against 15,555 prompt rows. At
`ANSWER_WEIGHT=4` that is **16%** of the objective's weight on the only
positions the adapter ever faces while decoding, and the number appeared
nowhere on disk. The rest was the same boilerplate 211 times, which the map
duly memorised — hence `R2_all = 0.9999` beside `R2_answer = 0.427`.

**Defect 2 — the target.** Ridge minimises L2 to *the small model's own layer-i
residual*. On a divergent prompt — small wrong, large right, the only prompts
this experiment can gain on — that residual is the state that decodes to the
**wrong** answer. A map that hits its target perfectly reproduces the 1B's
mistake. No amount of R² repairs a target that carries the error.

The fixes: capture now mixes 700 generic-corpus items into the bank prompts and
varies the system prompt and question framing per item (`common/templates.py`,
`common/fit_corpus.py`), and stores the 3B's top-128 next-token distribution at
every position. `--train-method distill` then optimises
`KL(3B next-token ‖ stitched-1B logits)` back through the frozen 1B suffix,
warm-started from ridge (`distill.py`). Both LLMs are asserted frozen; only
`W, b` train.

| capture | items | rows | answer rows | answer weight frac |
| --- | --- | --- | --- | --- |
| original (`states_legacy/`) | 211 | 16,308 | 753 | 16% (unrecorded) |
| rebuilt | 911 | 89,507 | **26,851** | **63.2%** |

`fit` now refuses to ship a map below 50%, records the fraction on every sidecar
and sweep row, and capture on `dev`/`test` is a hard error rather than a warning.

### Ridge vs distill, warm, llama/hard_factual dev (n=106)

Both cells, all methods, same scoring and CI rules. `vs 1B` is a paired
bootstrap over prompts against **that run's own** 1B baseline.

| cell | method | accuracy (95% CI) | vs 1B (bootstrap) | divergent recovered | answer R² | val KL |
| --- | --- | --- | --- | --- | --- | --- |
| L14→L8 | legacy ridge | 57.5% [48.0–66.5] | −16.0 [−23.6, −8.5] | 3.7% (1/27) | 0.427 | — |
| L14→L8 | **ridge (rebuilt)** | **64.2% [54.7–72.6]** | −8.5 [−15.1, −1.9] | 7.1% (2/28) | 0.626 | 0.539 |
| L14→L8 | **distill** | 63.2% [53.7–71.8] | −9.4 [−15.1, −3.8] | 3.6% (1/28) | 0.707 | **0.304** |
| L18→L10 | legacy ridge | 33.0% [24.8–42.4] | −40.6 [−50.0, −31.1] | 0% (0/27) | 0.260 | — |
| L18→L10 | **ridge (rebuilt)** | 43.4% [34.4–52.9] | −29.2 [−38.7, −20.8] | 3.6% (1/28) | 0.631 | 0.491 |
| L18→L10 | **distill** | **44.3% [35.2–53.8]** | −28.3 [−37.7, −18.9] | 7.1% (2/28) | 0.734 | **0.336** |

1B alone 72.6% [63.5–80.2] on the new runs, 73.6% on the 2026-08-17 run — bf16
kernels are not bit-deterministic across processes and one prompt moved. Every
row is bootstrapped against its own run's baseline; ~1 prompt (0.9 pts) is drift.

**What the capture fix bought: +6.7 pts** at L14→L8 (57.5 → 64.2) and **+10.4
pts** at L18→L10 (33.0 → 43.4), with held-out answer R² measured on 6,382 rows
instead of 191. That is a genuine adapter improvement and it is entirely due to
fitting on the right *distribution*.

**What distillation bought: nothing in accuracy.** It won decisively on its own
objective — val KL 0.539 → 0.304 (−44%) and 0.491 → 0.336 (−32%), both
`distill_improved_on_ridge: true`, on a prompt-level split held out of training,
still falling at epoch 4. It converted none of that into accuracy: −1.0 pt at
one cell, +0.9 at the other, i.e. one prompt each way.

So the honest bar — *distill accuracy > ridge at the same (i, j, warm)* — is
**not met**. And the experiment bar — *beat the 1B's 72.6%* — is not met by any
configuration; the paired bootstrap excludes zero in every row, so the stitch is
**significantly worse** than just running the 1B.

### Why: `warm` cannot change the first token

In `warm` mode the 1B prefills the prompt itself, so the first generated token
is decoded from the last **prompt** position — which the adapter never touches.
The stitched warm path emits the 1B's own first token on every prompt, always.

On this bank the answer usually *is* that token. Measured on dev: of 28
divergent prompts, the 1B's first token already differs from the 3B's on **26**.
Only **2/28 (7.1%)** are reachable by a warm stitch *at all*, however good the
adapter. Ridge recovered 2/28 and distill 1/28 at L14→L8 (the reverse at
L18→L10) — both sitting at the structural ceiling, not below it.

The generations make it concrete. Both methods reproduce the 1B's wrong token:

| prompt | gold | 1B | 3B | ridge | distill |
| --- | --- | --- | --- | --- | --- |
| atomic number of plutonium | 94 | 92 | 94 | **92** | **92** |
| atomic number of tungsten | 74 | 83 | 74 | **83** | **83** |
| atomic number of ruthenium | 44 | 88 | 44 | **88** | **88** |
| atomic number of xenon | 54 | 92 | 54 | **92** | **92** |
| year Constantinople fell | 1453 | 1521 | 1453 | 1522 | 1525 |

`bench` reports this ceiling on every warm run, so a recovery rate is read
against what was reachable rather than against 100%.

### Conclusion: the wall is the stitch, not the objective

Three independent facts now point the same way, and the first two were already
in the 2026-08-17 grid:

* `warm` beats `exit` in **every** cell — the *less* the injected residual is
  relied on, the better the result.
* Accuracy falls monotonically as either handoff goes deeper.
* Fixing the fit distribution moved accuracy by +6.7/+10.4 pts; fixing the
  objective on top of that moved it by ~1 prompt, while cutting the loss it
  optimises by 32–44%.

A better adapter is therefore **necessary and not sufficient**. Distillation can
make the 1B's tail prefer the 3B's next token at the positions it trains on; it
cannot make frozen 1B blocks 8–15 treat a mapped 3072→2048 vector as native 1B
thought, and a 6.3M-parameter affine map is a thin interface for that. Combined
with the first-token ceiling, overwrite-and-continue into a frozen small model
is the binding constraint.

**Do not spend more effort on adapter variants.** The next change is to the
stitch itself: add to the residual instead of replacing it, inject at the last
prompt position (so the token that decides the answer is in the loop), or
unfreeze the 1B tail. Each of those changes the hypothesis rather than the
estimator. The full 4×4×2 sweep was deliberately **not** re-run — the question
was the objective, and the objective is now answered.

## Results (llama, hard_factual, MPS, 2026-08-17) — historical, all-ridge

Baselines on dev (106 prompts): small **73.6%**, large **98.1%**, 26 divergent.

| config | mode | accuracy (95% CI) | vs small | divergent recovered | cost vs small (decode / end-to-end) |
| --- | --- | --- | --- | --- | --- |
| best of 16 cells (`L14 → L8`) | `warm` | **57.5% [48.0–66.5]** | −16.0 | 3.7% | 1.72x / **1.89x** |
| best of 16 cells (`L14 → L8`) | `exit` | **40.6% [31.7–50.1]** | −33.0 | 7.4% | 1.74x / **1.84x** |
| depth-matched (`L18 → L10`) | `warm` | 33.0% [24.8–42.4] | −40.6 | 0% | 2.11x / 2.29x |
| depth-matched (`L18 → L10`) | `exit` | 16.0% [10.3–24.2] | −57.5 | 0% | 1.95x / 2.06x |

The end-to-end column is new and is the honest one: the per-token ratio omits
the large-model prefix prefill this path pays on every request (96 ms against
the small model's 43 ms), so the real cost of the best cell is 1.89x, not 1.72x.
The intervals are new too, and here they change nothing — the best cell's upper
bound (66.5%) is still far below the small model's 73.6%, so the failure is
unambiguous rather than a sample-size artefact.

**All 32 configurations failed.** Across `j ∈ {14,18,21,24} × i ∈ {8,10,12,14}`
in both modes, not one beat the small model alone. The best cell reached 57.5%
against the small model's 73.6% — 16 points *worse*, while costing 1.7x its
decode time and a partial large-model forward on top. Full tables in
`results/llama/hard_factual/sweeps/`.

The grid is monotone in a way that says what is happening: accuracy falls as
either handoff gets deeper (`L14→L8` 57.5% → `L21→L14` 15.1%), and `warm` beats
`exit` in every cell. Both point the same way — the *less* the injected residual
is relied on, the better the result. The best configuration is the one that
perturbs the small model least, which is the signature of an injection that adds
noise rather than information.

Inspecting `warm` generations confirms it. The stitched path reproduces the small
model's own answers — including its wrong ones — and corrupts the rest:

| prompt | small | large | stitched |
| --- | --- | --- | --- |
| atomic number of plutonium | 92 ✗ | 94 ✓ | **92** ✗ |
| atomic number of tungsten | 83 ✗ | 74 ✓ | **83** ✗ |
| year Constantinople fell | 1521 ✗ | 1453 ✓ | **1524** ✗ |
| capital of Ivory Coast | Yamoussoukro ✓ | Yamoussoukro ✓ | **Yambe** ✗ |
| capital of Ghana | Accra ✓ | Accra ✓ | **`Acc \|\x08\x08`** ✗ |

Divergent recovery — the metric this experiment exists to move — is **0% in 22
of 32 cells** and never exceeds 11.1% (3 of 26 prompts). The large model's
knowledge is not arriving.

Note the adapter quality column: held-out answer-token R² is **0.50 at best and
negative in 6 cells**, against ~1.00 on all tokens. That all-token-vs-answer-token
gap is the same one that hid the earlier failure in the sibling package.

> **Superseded, 2026-08-23.** This paragraph originally read that gap as *the
> direct explanation* for the failure. It is not. Rebuilding the fit
> distribution lifted answer R² from 0.427 to 0.626 and accuracy from 57.5% to
> 64.2%, and distilling against the 3B's own next-token distribution improved
> the objective by a further 44% — and accuracy still did not reach the 1B.
> Adapter quality was a real defect, worth ~7–10 points, and it was never the
> binding one. See the section above.

Note also that R² does not order the cells: `L14→L10` has the best answer R²
(0.498) and scores 34.9%, while `L14→L8` has 0.427 and scores 57.5%. R² is a
tripwire for total collapse, not a selection criterion.

### Honest limits

- **Still runs a partial large model.** Even had it worked, this is not a
  small-model deployment: you pay `j` large blocks per token. The realistic
  framing is "cheaper than the large model, better than the small one", and it
  currently achieves neither.
- **A linear adapter is a weak instrument.** An MLP correction is the obvious
  next thing to try; the sibling package has one, where it did not help
  (5M parameters moved held-out answer R² from 0.263 to 0.250). That is evidence
  against, not for, but it was measured on the other direction. The 2026-08-23
  result argues against spending it here at all: the *objective* was fixed and
  bought nothing, so widening the same interface is unlikely to be what is
  missing.
- **Small eval splits.** dev and test are ~106 prompts each; one prompt is ~0.9
  points. The gaps reported here are far larger than that noise, but neighbouring
  grid cells are not meaningfully ordered, and the ridge-vs-distill differences
  (±1 prompt) are explicitly *not* resolved.
- **Distillation was stopped at 4 epochs and val KL was still falling.** More
  epochs would lower it further. That is a limit on the KL number, not on the
  conclusion: accuracy was flat across a 32–44% KL improvement, and the
  first-token ceiling caps warm recovery at 2/28 regardless.
- **`exit`-geometry distillation was not run.** `--distill-train-mode exit` is
  wired (it is the only geometry where the adapter is in the loop for the first
  answer token) but was not spent, because the 2026-08-17 grid already shows
  `exit` losing to `warm` in all 16 cells by 10–17 points. It is the honest
  next experiment only if paired with a change to the injection itself.
- **One pair.** Only llama has been run end to end. `--pair qwen` is wired and
  needs its own `capture`.

## Files

| file | role |
| --- | --- |
| `config.py` | pairs, banks, grids, split fractions, paths, train methods, distill hyperparameters, gating thresholds |
| `data.py` | prompt splits; capture of paired states + teacher logits over prompt **+ answer**; answer-weighted row selection |
| `adapter.py` | ridge and distill dispatch, norm matching, save/load, reload check (incl. the distill probe replay) |
| `distill.py` | KL to the 3B's next-token distribution through the frozen 1B suffix; freeze assertions; warm start |
| `stitch.py` | `LargeToSmallRunner` plus the direction-specific checks |
| `evaluate.py` | accuracy (overall + divergent subset), the warm first-token ceiling, latency, verdict, tables |
| `run.py` | CLI, including `compare` (head-to-head from disk, no GPU) |
| `../tests/smoke_distill_l2s.py` | synthetic end-to-end check of the distill path: freeze, warm start, 0-epoch identity, saved-map replay, and that the training forward **is** the warm inference forward |

Outputs land in `results/<pair>/<bank>/`: `states/` and `states_legacy/`
(gitignored), `adapters/`, `checks/`, `benches/`, `sweeps/`, `tables/`,
`headroom_<split>.json`. New artefacts carry the train method in the filename
(`adapter_i08_j14_distill.npz`, `bench_i08_j14_dev_warm_ridge.json`); the
unscoped names belong to the 2026-08-17 run and are never written again.
