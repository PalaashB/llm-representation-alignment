# stitching_small_to_large — LATENCY: small early layers → large late layers

**Direction:** small → large. **Goal:** decode faster than the large model.

For the opposite direction — large → small, aimed at *accuracy* rather than
latency — see the sibling package
[`stitching_large_to_small/`](../stitching_large_to_small/README.md).

One question only: **can we decode meaningfully faster than the large model
while giving up only a little accuracy?**

The stitched path is

```
prompt -> small embed + small blocks 0..i-1 -> adapter -> large blocks j..end
       -> large norm -> large lm_head -> token
```

Large blocks `0..j-1` and small blocks `i..end` never run. Both models keep a KV
cache over the layers they do run, so decode is O(n). Per decoded token the path
costs `i / n_small` of the small model plus `(n_large - j) / n_large` of the
large one, so **larger `j` is faster and less faithful** — that frontier is what
`sweep` searches.

This project is independent of the `diagnosis/` project next door. It borrows
four leaf modules from `../common` (two prompt banks, answer scoring, model
loading and chat formatting) and nothing else: no CKA, no divergence grid, no
cross-family pairs. It captures its own states and fits its own adapters, so you
do not need to have run anything in `diagnosis/`.

## Quickstart

```bash
cd experiments/palaash

python -m stitching_small_to_large.run headroom --pair llama                # is this bank worth a sweep?
python -m stitching_small_to_large.run capture  --pair llama                # ~2 min, writes ~750 MB of states
python -m stitching_small_to_large.run fit      --pair llama --i 10 --j 14   # seconds (linear) / ~25 s (mlp)
python -m stitching_small_to_large.run bench    --pair llama --i 10 --j 14   # small vs large vs stitched
python -m stitching_small_to_large.run sweep    --pair llama --modes exit warm    # the grid + Pareto pick
python -m stitching_small_to_large.run report   --pair llama --split dev     # rebuild the table, no GPU work
python -m stitching_small_to_large.run compare  --pair llama --split dev     # adapter variants side by side
python -m stitching_small_to_large.run final    --pair llama --i 10 --j 14 --modes warm  # untouched test split
python -m stitching_small_to_large.run check    --pair llama --j 14          # verify the fast path is exact
```

`--pair qwen` works the same way (Qwen2.5-0.5B into Qwen2.5-3B) and needs its own
`capture`. Only token-aligned (same-family, same-tokenizer) pairs are supported —
the adapter is fit position-by-position.

Four flags choose what is being measured, and **every artefact on disk is scoped
by all four**, so variants never overwrite each other and a table never mixes
them:

| flag | choices | what it changes |
| --- | --- | --- |
| `--bank` | `list_hard` (default), `list`, `factual` | the prompts, and the decode budgets that suit them |
| `--adapter` | `linear` (default), `mlp` | the map's **shape**: affine, or affine + nonlinear correction |
| `--train-method` | `ridge` (default), `distill` | how the map is **fit**: least squares on hidden states, or KL to the teacher's next-token distribution |
| `--n-taps` | `1` (default), 2, 3 | how many small-model layers feed the adapter |

`--bank` requires its own `capture`. The other three need only a refit
(`distill` needs a capture that stored teacher logits, which is the default).

`--adapter` and `--train-method` are orthogonal: either shape can be fit by
either method, so `--adapter mlp --train-method distill` is a valid combination.

## Five defects this folder was rebuilt around

The `factual` campaign below is a negative result, and re-reading it turned up
five reasons the measurement could not have shown a positive one either. They
are worth stating first because most of the design decisions in this folder are
now downstream of them.

1. **The adapter was fit on the wrong objective.** Ridge minimises L2 in
   residual space, weighting all 3072 output dimensions equally, but next-token
   identity lives in the few directions that survive the final norm and
   `lm_head`. The tell was already on record: `R2_all = 0.999` against
   `R2_answer` from −0.20 to +0.375, and answer-token R² not ranking the grid
   cells. `--train-method distill` optimises the scored quantity instead.
2. **The fit distribution was 95% boilerplate.** One chat template and one
   system prompt across every prompt meant 258 answer rows against 5180 prompt
   rows; at `ANSWER_WEIGHT=4` that is **17%** of the objective. Fixed by
   template variety (`common/templates.py`), answer-majority row selection
   (asserted, and recorded as `answer_weight_frac`), and `--fit-corpus generic`
   for volume.
3. **The latency claim was not computed honestly.** `sweep_dev.csv` had no
   end-to-end column at all, so the best-accuracy cell was tabulated by its
   1.11x *decode* speedup while actually running **0.93x end-to-end** — slower
   than the model it was meant to beat. Now every row carries
   `end_to_end_speedup_vs_large`, prefill is printed per row, and the pricing
   source is named.
4. **The benchmark could not show a win.** 35-prompt splits give a ~23-point
   confidence interval across an 8.6-point gap. `list_hard` gives 173.
5. **The bar was set against the wrong incumbent.** Selection ranked stitch
   cells against each other; the thing to beat is the small model, which is
   88.6% at 2.4x. It now competes in the frontier, on end-to-end speed, with
   interval-separated accuracy.

## The benchmark: why there are three banks

`--bank factual` is the original 140-prompt short-answer set ("what is the capital
of New Zealand?"). It is a bad instrument for a latency experiment, in two ways
that both flatter the small model:

1. **Decode barely happens.** Answers are 3-4 tokens against a ~40 token prompt,
   so end-to-end time is dominated by prefill — the part stitching does not help.
2. **The models tie.** Llama-1B answers it nearly as well as Llama-3B, so there
   is almost no accuracy for a stitch to *recover*, and headroom between the two
   models is the only thing a stitch's extra layers can buy.

`--bank list` (180 prompts, `common/prompts_list.py`) asks for a *set* instead —
three capitals, the four DNA bases, the nine countries bordering Germany — and
`common.scoring.is_correct_all` requires **every** item, in any order and with any
wording around it. Both properties push the same way: a longer answer is also a
harder one.

Measured on llama, dev split (`run headroom`):

| bank | n | small | large | headroom | 95% CI width | mean output tokens | decode share |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `factual` | 35 | 88.6% | 97.1% | +8.6 pts | ±23 pts | 3.5 | 65% |
| `list` | 45 | 73.3% | 91.1% | +17.8 pts | ±21 pts | 10.3 | 84% |
| `list_hard` | **173** | — | — | **~+17 pts** | **±11 pts** | ~10 | ~84% |

The CI column is the one that changes the conclusions. On `factual` the interval
is nearly three times the gap being measured, so *no* result on that bank could
have been significant — including the negative one, which is why the negative
result is stated as "no cell was demonstrably better", not "stitching cannot
work". `list` is barely better. Only `list_hard` can resolve a difference of the
size these experiments are looking for.

### `list_hard`: conjunction over audited facts

`common/prompts_list_hard.py` composes three-fact conjunctive questions from
`common/prompts_hard_factual.py`, whose gold answers have already been through
the both-models-wrong audit. That buys three things at once:

* **Size.** 554 prompts, dev = 173 and test = 173, so one prompt is 0.6 points.
* **Headroom, for free.** Conjunction turns a per-fact gap into a per-question
  one multiplicatively — a category where the 1B is right 73% of the time per
  fact is right 39% of the time on three of them, while the 3B stays near 100%.
  No new facts had to be written or verified.
* **Longer answers**, so decode dominates the clock.

The split is over **facts, not composed prompts**. Splitting composed prompts
would put a fact in a `fit` prompt and in a `dev` prompt, and the adapter would
be scored on an answer it was fit on.

Its category mix was chosen by measurement, not intuition, the same way
`hard_factual`'s was. A first version weighted all categories equally and got
+11.0 pts; the per-category breakdown showed atomic numbers (+0.0, both models
100%) and currencies (+0.0) contributing nothing while being 44% of the split,
so those two were down-weighted. That run is kept as
`results/llama/list_hard/headroom_dev_v1_uniform_categories.json`.

Note this does **not** reproduce the single-fact category table in the sibling
README, where national capitals and element symbols showed 0% divergence. Under
conjunction they give +15.0 and +15.8. Which categories are productive depends
on the question shape, so the measurement has to be redone per bank.

So the new bank doubles the accuracy a stitch could win back and moves most of the
clock into the phase stitching actually accelerates. It also quadruples the
adapter's training signal, because `capture` teacher-forces the large model's own
answer: 90 fit prompts yield ~1.1k answer rows on `list` against ~270 on
`factual`.

Building it took two passes, which is worth recording. The first version used
sets of 2-4 famous items; the 3B scored **100%** of the dev split at 8-10 output
tokens, so there was a ceiling instead of a gap. The `*_hard` tier (5-14 items,
entities from the parts of the world a 1B has seen least — borders of Chad, the
capitals of Central Asia, the currencies of the Caucasus) is what produced the
table above. `run headroom` exists so this costs two baseline passes to find out
rather than a whole sweep.

## The thing that makes the adapter work at all

An earlier attempt (since removed) was fast but its generations collapsed:
`Wellington` → `"Well is"`, `Jefferson City` → `"Jeffapolis"`. First token right,
then noise. The adapter reported held-out R² = 0.91 and looked fine.

The cause was the fit set, not the layer choice. States were captured over
**prompt tokens only**, but at decode time the adapter is handed the small
model's state at positions holding *generated answer* tokens — which it had
never seen. On answer tokens that same adapter scored R² = 0.34.

So `capture` runs each prompt through the large model to get its greedy answer,
then teacher-forces `prompt + answer` through **both** models and records every
position, flagging the answer ones.

### That fix was necessary but not sufficient: the fit set was still 95% boilerplate

Capturing answer positions put them in the fit set. It did not make them matter.
On the `factual` capture: **258 answer rows against 5180 prompt rows**, and
because every prompt used the same chat template and the same system prompt,
those 5180 rows are one boilerplate prefix repeated 70 times. At
`ANSWER_WEIGHT = 4` the answer rows carry 258×4 / (258×4 + 5180) = **17%** of
the objective. A ridge solve handed that data does the sensible thing and
memorises the template — which is exactly what `R2_all = 0.999` means — while
the positions the adapter actually faces at decode time are a rounding error.

Three changes, all now asserted rather than hoped for:

* **Template variety** (`common/templates.py`). Several paraphrases of the
  system prompt and of the framing around the question, chosen deterministically
  per item. The question text itself is never altered — the bank's gold answer
  is attached to its exact wording — but the wrapper is, which moves the answer
  tokens' positions and changes what precedes them. Eval always uses the
  canonical template, so this is training-time augmentation only and cannot
  flatter a score.
* **Answer-majority row selection** (`data.select_fit_rows`). Every answer row
  is kept; prompt rows are subsampled, evenly across prompts, to whatever count
  puts answer positions above `MIN_ANSWER_WEIGHT_FRAC` of the total weight. They
  are thinned rather than dropped: `exit` mode hands the adapter prompt
  positions at inference, and they anchor the standardiser. The realised
  fraction is written to the sidecar as `answer_weight_frac` and `fit` refuses
  to run below the threshold. On the old capture this takes 17% → 55%.
* **Volume** (`--fit-corpus generic`). Bank answers are 3-10 tokens, so a bank
  alone cannot produce answer rows in the tens of thousands at any sane prompt
  count. `common/fit_corpus.py` supplies open-ended instructions whose
  continuations run 40-80 tokens, drawn from templates and topics that touch no
  eval bank — and the non-overlap is checked, not asserted, with capture
  refusing a corpus that collides. (`--fit-corpus wikitext` is wired for
  environments with `datasets` installed; it raises rather than silently
  falling back, because a fit set quietly swapped for a different one is
  undetectable in the sidecar.)

Capturing on `dev` or `test` is now a hard error rather than a printed warning.
A warning scrolls off the top of a capture log that runs for half an hour.

`fit` then:

- up-weights answer rows (`--answer-weight`, default 4),
- uses a lighter ridge (`--alpha`, default 0.01 vs the diagnosis fit's 0.1) —
  shrinkage pulls every prediction toward the mean residual, which is exactly the
  "generic continuation" failure,
- rescales the centred prediction by one scalar so its radius matches the
  target's (`--no-norm-match` to disable). **Measured, it barely does anything
  here** — the fitted gain is 1.01-1.03, i.e. the light ridge already left the
  radius intact. It is on by default because it costs nothing, not because it
  earned its keep.

Reported adapter quality is always split into `all` / `answer` / `prompt` rows,
because the all-token number is the one that misled the previous attempt.

## How the map is fit: ridge vs distillation

`--train-method` is orthogonal to `--adapter`. The latter is the map's *shape*;
this is the *objective* it is fit against, and the objective was the bigger
problem.

**Ridge optimises something the model does not read.** It minimises L2 between
the adapter's output and the large model's layer-`j` residual, weighting all
3072 output dimensions equally. But next-token identity survives only in the
directions that clear the final RMSNorm and the unembedding, and the residual
stream's variance is dominated by directions `lm_head` largely discards. Ridge
therefore spends its capacity where error is cheapest to reduce, which is not
where error costs accuracy.

The evidence was already in this repo before the cause was named:

* held-out `R2_all = 0.999` against `R2_answer` from −0.20 to +0.375;
* answer-token R² **does not order the grid cells by accuracy** — in the sibling
  package `L14→L10` has the best answer R² (0.498) and scores 34.9%, while
  `L14→L8` has 0.427 and scores 57.5%.

Both READMEs already described R² as "a tripwire for total collapse, not a
selection criterion". That is the signature of a misaligned proxy, not a noisy
one.

**`--train-method distill` fits the scored quantity directly.** The loss is
KL(teacher ‖ stitched) on the large model's own next-token distribution at
answer positions, backpropagated through the frozen suffix blocks into the
adapter, plus a small cross-entropy term on the teacher's argmax.

Four properties make the comparison meaningful:

* **Only the adapter trains.** Every LLM parameter has `requires_grad=False`,
  and `distill._assert_only_adapter_trains` checks both that *and* that the
  optimiser's parameter list is exactly the adapter's tensors. The counts land
  in the sidecar as `frozen_llm_evidence`.
* **Warm-started from ridge**, so epoch 0 *is* the ridge map and training can
  only improve on it (subject to the early-stopping split). If no epoch beats
  the warm start, ridge ships and `distill_improved_on_ridge: false` says so.
* **Trained in the `warm` injection pattern** — prompt positions carry the large
  model's true residual, answer positions the adapter's output. A map fit under
  one injection pattern and run under another is being asked a different
  question at inference than it was fit on, which is the mistake prompt-only
  capture made.
* **Teacher distributions are top-K** (`DISTILL_TOPK = 128`), because full vocab
  is 128256 floats per row and unstorable at this row count. K is recorded in
  the sidecar so the approximation is never invisible.

Expect held-out R² to **fall** after distillation. That is the point: the map is
no longer optimising L2 in residual space. The sidecar keeps both numbers
(`held_out` and `held_out.after_distill`) so the trade is visible.

The `adapter_reload` check gains a distillation-specific arm: the saved `.npz`
is reloaded and replayed through the suffix on a recorded probe sequence, and
must reproduce the argmax and logits training ended at. Shape and finiteness
checks would not catch a dtype narrowing or a transposed weight in a map that is
still approximately the ridge solution.

## The two ways to improve the map's shape

Held-out answer-row R² at `L10 → L14` on llama/`list`, all four combinations:

| `--adapter` | `--n-taps` | map params | held-out R² (answer) | cosine (answer) |
| --- | --- | --- | --- | --- |
| `linear` | 1 | 6.3M | 0.263 | 0.762 |
| `linear` | **2** | 12.6M | **0.372** | **0.783** |
| `mlp` | 1 | 11.5M | 0.263 | 0.761 |
| `mlp` | 2 | 19.9M | 0.371 | 0.783 |

**`--n-taps 2` is a real win and it is free at inference.** The adapter reads the
small model's residual at layer `i/2` as well as at `i`, concatenated. Those
blocks run on the way to `i` regardless, so the intermediate residual is already
in flight and only has to be kept — no extra compute, just a wider first matmul.
It helps because one layer's residual stream is a lossy summary of the
computation that produced it: attention and MLP writes overwrite as much as they
accumulate, so a feature the large model still needs at `j` may be legible at
`i/2` and gone by `i`.

**`--adapter mlp` does not help, and the reason is instructive.** A
`Linear → GELU → Linear` correction is added to the affine map with its output
layer zero-initialised, so training starts at exactly the ridge solution. It
reduces validation MSE on the residual it is trained against by 64% (0.826 →
0.300) — there *is* nonlinear structure to find — but held-out answer R² does not
move at all. The reduction is almost entirely on prompt positions, which the
affine map already reconstructs at R² = 0.999 and which are 87% of the rows.

It also overfits hard: without a stopping rule, validation MSE bottoms at **epoch
5** and then climbs to 0.634 by epoch 80 while training loss keeps falling to
0.097. ~1.1k answer rows cannot constrain 5M weights. `MLP_VAL_FRAC` holds 20% of
*prompts* (not rows — neighbouring positions in one answer are near-copies) out of
the correction's training set to pick the epoch, and epoch 0 is a legitimate
candidate: if nothing beats the ridge map, the correction is dropped and the
shipped map is affine, recorded as `mlp_correction_kept: false`.

The lesson for anyone extending this: the bottleneck is not the map's *capacity*.
It is that answer positions are a small minority of a small dataset. More answer
rows (a bank with longer answers, more fit prompts) is worth more than a bigger
map.

## Reading the output

`bench` prints one row per path:

```
path                             acc    ms/tok    tok/s   prefill  ms/answer   decode     e2e   params
1B                             XX.X%      24.5     40.8      46ms      298ms    2.45x   2.20x      38%
3B                             XX.X%      60.1     16.6     116ms      735ms    1.00x   1.00x     100%
stitch-warm-t2-linear-L10->L14 XX.X%      ...       ...       ...        ...      ...     ...      ...
```

- **acc** — greedy, EOS-stopping, `bank.max_new_tokens` budget, scored by
  `common.scoring.score`: any-alias for `factual`, all-items for `list`.
- **ms/tok** — measured on a *separate* pass with EOS ignored and a fixed
  `bank.latency_steps` budget on every path, so each path is timed over identical
  work rather than over however many tokens it happened to emit.
- **prefill** — time to first token.
- **ms/answer** — prefill + decode for one whole request, **priced for every path
  at the same output length** (the large model's mean on that split). This is the
  number a user waits for, and pricing it at a fixed length is what stops a path
  from looking fast by stopping early.
- **decode / e2e** — per-token and end-to-end speedup vs the large model. `decode`
  is the mechanism; `e2e` is the consequence, and the two differ sharply between
  the prefill modes.
- **params** — weights multiplied per decoded token, as a fraction of the large
  model's. A hardware-independent sanity check on the wall-clock number
  (embedding lookups excluded, `lm_head` and the adapter included).

Then:

```
stitch vs large: -X.X accuracy pts at 1.XXx decode / 1.XXx end-to-end
stitch vs small: +X.X accuracy pts (gap recovered: XX%)
```

`gap recovered` places the stitch between the two models: 0% = no better than
the small model, 100% = matches the large one. **A stitch below 0% is a failure
regardless of how fast it is** — you could just run the small model.

### The two prefill modes

`--modes exit warm` benchmarks both. The decode step is identical, so both get
the same per-token speedup; they differ only in what the large suffix blocks
attend back to.

- **`exit`** — the prompt is stitched too, so large blocks `0..j-1` run over the
  sink position only. Saves prefill as well, but every prompt KV the suffix
  reads is the adapter's reconstruction.
- **`warm`** — the large model prefills the prompt itself, so prompt KVs are
  exactly its own and the adapter only supplies the residual at generated
  positions. Prefill is *not* saved — it costs a full large prefill **plus** the
  small one, so time-to-first-token is worse than the large model alone. Worth
  it only when output length dominates, which is what the `ms/answer` column
  prices.

### The sweep table

`sweep` (and `report`, which rebuilds it from saved benches with no GPU work)
writes `sweep_<split>[<variant>].csv` / `.json` and prints the grid sorted by
accuracy. `+` marks the Pareto frontier and `*` the recommendation.

**Selection is on end-to-end speed, not decode.** Decode speedup is the
mechanism; end-to-end is what a user waits for, and `warm` mode buys its decode
win by paying a full large prefill *plus* a small one. On `factual` that made the
best-accuracy cell 1.11x per token and **0.93x end-to-end**. Ranking on decode
recommends cells that lose wall-clock time, so the axis is now
`end_to_end_speedup_vs_large` and `MIN_E2E_SPEEDUP` is the floor.

**The small model competes in that frontier.** Ranking stitch cells only against
each other will cheerfully recommend one that is both slower *and* less accurate
than just running the small model, which is always available at its own speedup.
Cells the small model beats on both axes are flagged `(dominated by small)` and
are never recommended.

**Accuracy comparisons use intervals, not point estimates.** A recommendation
must clear `MIN_E2E_SPEEDUP` **and** beat the small model with non-overlapping
95% Wilson intervals. On a 35-prompt split that bar is essentially unreachable,
which is the correct behaviour: `L8→L14` warm scoring 65.7% on dev and 85.7% on
test — same adapter, same configuration — is what point-estimate ranking looks
like when the interval is 23 points wide. If nothing qualifies, `sweep` reports
that no viable point exists rather than dressing up a dominated cell.

Two lines are printed regardless of whether anything is recommendable:

* the best cell that is **genuinely faster end-to-end than the large model**,
  which is a different question from "worth recommending" and was previously
  invisible;
* counts of cells that clear the speed floor, beat the small model at 95%, are
  dominated, and are prefill-suspect.

**Prefill is printed per row** (`pre(st/lg)`), because a prefill regression
otherwise hides behind a decode win. Cells whose prefill is out of line with
their peers in the same mode are flagged `[prefill suspect]` — a cross-cell
test, because the within-cell `latency_suspect` flag is blind to a machine that
was uniformly slow for one cell's whole measurement. That is exactly what
happened at `i=12, j=10` warm: prefill 403.9 ms against ~160 ms for its
neighbours, 2.6x the layer-fraction model, while its decode spread was a tight
1.11 and nothing fired. It is contamination, not cost — there is no mechanism by
which two extra small blocks add 240 ms of prefill while that same cell's decode
ratio stays normal. Two other cells (`i12 j16`, `i14 j10`) were equally affected.

`compare` lines the adapter variants up against each other at fixed geometry,
which is the only way the `--adapter` / `--n-taps` question is legible — each
variant summarised by its own winner would confound the map with the layer pair.

## Results

### The `factual` bank (MPS, 2026-08-16) — negative

Baselines: dev (35) small 88.6% / large 97.1%; test (35) small 88.6% / large 94.3%.

**All 50 llama grid points and all 32 qwen ones were dominated by the small
model** — every one both slower *and* less accurate than simply running the 1B.
The small model decodes 2.4x faster and gives up only 6-9 points, a better trade
than any stitch found.

Re-scored on the untouched `test` split:

| point | acc | vs large | vs small | decode speedup | prefill | params |
| --- | --- | --- | --- | --- | --- | --- |
| `L10→L10` warm | 91.4% | −2.9 | +2.9 | 1.11x | 139ms (large: 114ms) | 88% |
| `L8→L14` warm | 85.7% | −8.6 | −2.9 | 1.34x | 135ms (large: 113ms) | 71% |

**Those decode speedups are not end-to-end speedups, and the table above
originally had no end-to-end column at all.** Re-running `report` (which rebuilds
the table from the saved benches with no GPU work) prices every path at the
large model's measured 3.46-token mean answer and gives a very different
picture on `dev`:

| point | acc (95% CI) | decode | **end-to-end** | prefill st/lg |
| --- | --- | --- | --- | --- |
| `L10→L10` warm | 85.7% [70.6–93.7] | 1.10x | **0.92x** | 160 / 116 ms |
| `L8→L10` warm | 82.9% [67.3–91.9] | 1.15x | **0.96x** | 153 / 115 ms |
| `L6→L10` warm | 74.3% [57.9–85.8] | 1.20x | **1.01x** | 144 / 114 ms |

The best-accuracy cell is **slower than the large model it was supposed to
beat**. Only 3 of 50 cells clear 1.25x end-to-end, against 21 that clear it on
decode, and all 50 are dominated by the small model. Three cells (`i12 j10`,
`i12 j16`, `i14 j10`) are now flagged `[prefill suspect]` — their prefill runs
2.0-2.6x the layer-fraction model while their decode ratios are normal, which is
machine contamination the old within-cell flag could not see.

**Do not read those two rows as a 20-point improvement.** `L8→L14` warm scored
65.7% on dev and 85.7% on test — same configuration, same adapter, two 35-prompt
splits. That swing is roughly what the ±8-point standard error predicts, so the
*ranking* of grid cells is not stable across splits. The robust effects are:

- **`warm` beats `exit` everywhere**, by 17-35 points at identical decode cost.
  Reconstructed prompt KVs are where most of the accuracy goes, not the generated
  positions.
- **Earlier `j` beats later `j`** monotonically, and `j` is also the only real
  source of speedup. That is the bind.
- Answer-token R² does **not** predict accuracy across cells. It is a tripwire for
  total collapse, not a selection criterion.

On qwen the same conclusion held more emphatically: small 77.1% @ **4.82x**, best
stitch `L18→L20` warm at 60.0% (−37.1 vs large, −17.1 vs small) at 1.57x. The
wider size ratio makes the small model a stronger competitor, not a weaker one.

### The `list` bank — see `results/llama/list/`

`sweep_dev*.csv` / `.json` per variant, `headroom_dev.json`, and
`bench/bench_i*_j*_{exit,warm}_dev*.json` with per-prompt generations. The
`headroom` and adapter-quality numbers quoted in the sections above are from these
runs; the sweep tables are the current state of the grid rather than a settled
result, and should be read with the same split-noise caution as the `factual`
tables above (`dev` and `test` are 45 prompts, so ±7 points).

### The `list_hard` bank (MPS, 2026-08-18) — the re-run

`headroom` on llama/dev (n=173) is the first time this folder has had an
instrument that can resolve what it is trying to measure:

| path | accuracy (95% CI) | ms/tok | prefill | out tok | ms/answer |
| --- | --- | --- | --- | --- | --- |
| 1B | 83.2% [77.0–88.1] | 24.0 | 45 ms | 17.4 | 281 ms |
| 3B | 98.3% [95.0–99.4] | 60.2 | 119 ms | 9.8 | 710 ms |

* headroom **+15.0 pts, intervals separated** (on `factual` they overlap);
* decode is **83%** of the large model's end-to-end time (`factual`: 65%);
* the small model's interval is **11.1 pts** wide (`factual`: 23.1);
* 27 of 173 prompts are small-wrong / large-right.
* `usable for a latency claim: YES`.

The bar a stitch has to clear here is the small model at **83.2%** and
**2.5x end-to-end**.

The fit set behind these adapters is the one the row-selection and corpus work
produced: **104,535 rows, 23,588 of them answer positions** (against 258 on
`factual`), **55.0%** of the objective's weight on answer rows (against 17%),
varied templates, and 700 generic-corpus items mixed in. Held-out answer-token
R² moved from 0.19–0.38 to **0.50–0.63** on the ridge maps as a direct result.

| path | accuracy (95% CI) | vs small (paired bootstrap) | decode | e2e | prefill st/lg |
| --- | --- | --- | --- | --- | --- |
| 1B alone | 83.2% [77.0–88.1] | — | 2.45x | **2.46x** | 45 / 116 ms |
| 3B alone | 98.3% [95.0–99.4] | +15.0 | 1.00x | 1.00x | 116 ms |
| `L10→L14` warm, **ridge** | **89.6% [84.2–93.3]** | +6.4 [−0.6, +13.9] | 1.28x | 1.17x | 143 / 114 ms |
| `L10→L14` warm, **distill** | **89.6% [84.2–93.3]** | +6.4 [−0.6, +13.3] | 1.28x | 1.16x | 145 / 116 ms |
| `L8→L14` warm, **ridge** | 75.7% [68.8–81.5] | −7.5 [−15.6, +0.6] | 1.34x | 1.22x | 137 / 115 ms |
| `L8→L14` warm, **distill** | **86.1% [80.2–90.5]** | +2.9 [−4.6, +10.4] | 1.34x | 1.21x | 144 / 115 ms |

**Distillation vs ridge at fixed geometry.** At `L8→L14` — identical layers,
identical decode cost, the only difference being the objective the same 6.3M-
parameter affine map was fit against — distillation is worth
**+10.4 accuracy points, paired bootstrap [+5.8, +15.6], excluding zero**. Its
KL validation loss fell 0.2296 → 0.1689 and its held-out answer R² rose 0.611 →
0.696. This is the clearest single result in the folder and it is a statement
about the *objective*, not about capacity: the map's shape and parameter count
did not change.

At `L10→L14` distillation makes **no difference at all** (89.6% both ways),
despite reducing KL 0.1600 → 0.1361 and raising answer R² 0.632 → 0.717. The
honest reading is that ridge is already good enough at that geometry for greedy
decoding to land on the same tokens; distillation buys the most where ridge is
worst, and buys nothing where ridge is already adequate.

**Still no viable point, and now for a clearly stated reason.** The best cell
recovers 42% of the small→large accuracy gap and runs 1.17x faster end-to-end
than the *large* model — but the *small* model runs 2.46x faster than large,
i.e. roughly twice the stitch's speed, and the stitch's +6.4-point accuracy edge
over it does not exclude zero. So the stitch loses outright on the speed axis
and is unresolved on the accuracy axis. It is much closer than the `factual`
campaign suggested (where every cell was dominated on both axes), but it is not
a win, and `sweep` reports `NO VIABLE POINT` rather than dressing one up.

## Honesty notes

- **Split discipline.** Each bank is split 50/25/25 into `fit` / `dev` / `test`
  (`list`: 90/45/45). The adapter is fit only on `fit`; the sweep selects on
  `dev`; `final` re-scores the winner on `test`, which nothing has touched. Trust
  the `test` number, not the sweep's best `dev` cell.
- **Every accuracy number carries a 95% Wilson interval, and the selection rule
  refuses to rank cells whose intervals overlap.** Wilson rather than the normal
  approximation because several cells score 0% and the large model sometimes
  scores 100%, where the textbook interval has zero width. On `factual`/`list`
  (35-45 prompts) that interval is 21-23 points wide, which is wider than almost
  every difference in those tables — those grids should be read as "nothing here
  was demonstrably better", not as rankings. `list_hard` brings it to ~11 points.
- **`list_hard` prompts are not fully independent.** Each fact appears in up to
  11 composed questions within its own split, so a Wilson interval computed as
  if the 173 prompts were independent trials is somewhat narrower than the truth.
  The effect is bounded — any two prompts share at most one of their three
  items, and a prompt only scores if all three are right — but `n=173` is worth
  a little less than 173 independent prompts. It is worth far more than the 35
  it replaces.
- **Conjunctive scoring is harsh on purpose.** On the `list` bank a generation
  naming four of five required items scores zero. `n_items_found` is recorded per
  generation so "nearly right" and "collapsed into noise" stay distinguishable —
  they are the same accuracy but very different failures.
- **Timing contamination is flagged, not hidden.** Decode cost is
  prompt-independent, so a wide spread across timing samples means the machine
  interfered. Cells where the stitch or the baseline spread exceeds
  `LATENCY_SPREAD_FLAG` are printed as `[latency suspect: ...]`; treat their
  speedups as unusable rather than as results.
- **Latency is this machine's.** Wall-clock on whatever `pick_device()` finds
  (MPS here). The `params` column travels; the milliseconds do not.
- **The baselines are real baselines.** Both run through the *same* decode loop
  as the stitch, not `model.generate`, so a comparison is not a comparison of two
  harnesses. `run.py check` asserts (a) the hand-rolled sliced layer loop
  reproduces HF's own `forward` logits, (b) that loop's greedy output matches
  `model.generate` token for token on both models, and (c) the attention-sink
  splice is bit-identical to HF's own prefix forward.
- **`mlp` inference is duplicated in numpy and torch.** `adapter.apply_map` scores
  the fit; `adapter.TorchAdapter` runs the decode. Both use the tanh GELU
  approximation deliberately so they compute the same function; they agree to
  ~1e-6 in float32.

## Files

| file | what it does |
| --- | --- |
| `config.py` | pairs, banks, sweep grids, adapter variants, training methods, thresholds, paths |
| `data.py` | prompt splits; capture of paired states + teacher logits; answer-majority row selection |
| `adapter.py` | weighted ridge, norm matching, MLP correction, save/load, the reload check |
| `distill.py` | KL-to-teacher training through the frozen suffix; the frozen-LLM assertions |
| `stitch.py` | sliceable decoder `Stack` (with taps), `FullRunner`, `StitchRunner`, checks |
| `evaluate.py` | accuracy + latency passes, end-to-end pricing, CIs, prefill anomalies, Pareto pick |
| `run.py` | CLI (`headroom / capture / fit / bench / sweep / report / compare / final / check`) |

Shared, in `../common/`: `stats.py` (Wilson intervals, paired bootstrap, the
CI-separation rule), `templates.py` (capture-time prompt variety),
`fit_corpus.py` (the generic fitting corpus), `prompts_list_hard.py` (the
`list_hard` bank), plus the pre-existing `decoding.py`, `scoring.py` and
`model_utils.py`.

Tests: `tests/smoke_distill.py` exercises the distillation path on a tiny
randomly-initialised pair — no real weights, runs in seconds — and asserts the
invariants that matter: no LLM weight moves, the warm start is the ridge map,
the saved file reproduces the training-time logits, and **the training forward
is numerically identical to the warm-mode inference forward** (1e-7). That last
one is load-bearing: a map fit under a different injection pattern than it is
decoded under would show a perfectly healthy loss curve.

Outputs land in `results/<pair>/` for the `factual` bank and
`results/<pair>/<bank>/` otherwise: `states/` (gitignored, large), `adapters/`,
`bench/`, `sweep_*.csv/.json`, `headroom_*.json`, `checks.json`. Adapter and bench
filenames carry the variant suffix (`_mlp`, `_lineart2`, `_mlpt2`; plain `linear`
with one tap has none).

## Why `preserve_prefix = 1`

Position 0 is the attention sink: its residual at large-model depth is ~30x the
norm of any other position, and the adapter cannot reproduce it — overwriting it
destroys attention globally and the model emits noise. The fix is exact rather
than approximate: attention is causal, so running only tokens `0..N-1` through
the skipped large blocks gives exactly the hidden states a full-sequence run
would give those positions. That is one 1-token forward per prompt, spliced over
the adapter's output. It is not a tuning knob.
