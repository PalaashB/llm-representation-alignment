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

**This is not a latency win and must not be read as one.** The path runs `j`
large blocks *plus* `n_small - i` small blocks, so it is slower than the small
model alone (~1.9x on llama) and slower than the large model is worth. Latency is
reported anyway, so the cost cannot be quietly dropped from the story.

## Quickstart

```bash
cd experiments/palaash

python -m stitching_large_to_small.run headroom --pair llama          # FIRST
python -m stitching_large_to_small.run capture  --pair llama          # ~8 min, ~670 MB
python -m stitching_large_to_small.run fit      --pair llama --i 10 --j 18
python -m stitching_large_to_small.run check    --pair llama --i 10 --j 18
python -m stitching_large_to_small.run bench    --pair llama --i 10 --j 18 --modes exit warm
python -m stitching_large_to_small.run sweep    --pair llama --modes exit warm
python -m stitching_large_to_small.run final    --pair llama --i 10 --j 18   # test split
```

Everything on disk is scoped by `(pair, bank)`, so runs never overwrite each
other. Token-aligned pairs only (`llama`, `qwen`) — the adapter is fit
position-by-position, which needs both models to see identical tokens.

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

## Results (llama, hard_factual, MPS, 2026-08-17)

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
gap is the same one that hid the earlier failure in the sibling package, and it is
the direct explanation here. A map explaining half the variance at answer
positions — often none of it — cannot carry a specific fact like "94" through the
small model's remaining blocks. It transmits the *register* of an answer without
its content, and the small model's own priors fill the rest back in.

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
  against, not for, but it was measured on the other direction.
- **Small eval splits.** dev and test are ~106 prompts each; one prompt is ~0.9
  points. The gaps reported here are far larger than that noise, but neighbouring
  grid cells are not meaningfully ordered.
- **One pair.** Only llama has been run end to end. `--pair qwen` is wired and
  needs its own `capture`.

## Files

| file | role |
| --- | --- |
| `config.py` | pairs, banks, grids, split fractions, paths, gating thresholds |
| `data.py` | prompt splits; capture of paired states over prompt **+ answer** |
| `adapter.py` | weighted ridge large→small, norm matching, save/load, reload check |
| `stitch.py` | `LargeToSmallRunner` plus the direction-specific checks |
| `evaluate.py` | accuracy (overall + divergent subset), latency, verdict, tables |
| `run.py` | CLI |

Outputs land in `results/<pair>/<bank>/`: `states/` (gitignored), `adapters/`,
`checks/`, `benches/`, `sweeps/`, `tables/`, `headroom_<split>.json`.
