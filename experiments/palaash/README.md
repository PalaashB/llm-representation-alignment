# Aligning small ↔ large LLM hidden states

Research code for studying representational alignment between a small and a
large instruction-tuned model. Four pairs are configured
(`diagnosis/config.py`):
two same-family pairs (token-level alignment) and two cross-family pairs
(prompt-level alignment — see below):

| pair | small model | large model | alignment |
|------|-------------|-------------|-----------|
| `llama` | Llama-3.2-1B-Instruct (16 layers, 2048-d) | Llama-3.2-3B-Instruct (28 layers, 3072-d) | token |
| `qwen` | Qwen2.5-0.5B-Instruct (24 layers, 896-d) | Qwen2.5-3B-Instruct (36 layers, 2048-d) | token |
| `llama2qwen` | Llama-3.2-1B-Instruct (16 layers, 2048-d) | Qwen2.5-3B-Instruct (36 layers, 2048-d) | prompt |
| `qwen2llama` | Qwen2.5-0.5B-Instruct (24 layers, 896-d) | Llama-3.2-3B-Instruct (28 layers, 3072-d) | prompt |

## Question 1 — can alignment identify the root causes of hallucinations?

> **Experiment:** Run prompts where the small model fails but the large model
> succeeds. Train layer-wise Direct Matching (DM) adapters to isolate the exact
> hidden layer where the smaller model's representations permanently diverge.

### Idea in one paragraph

In a same-family pair the two models share the **same tokenizer**, so a given
prompt produces the identical token sequence in both — their hidden states
line up position-by-position. A **Direct Matching adapter** is an affine map
`Ŷ = X·W + b` fit by ridge regression that tries to translate a small-model
hidden layer `X` into a large-model hidden layer `Y`. The *held-out* R² of that
map measures how translatable the small representation is into the large
geometry. We fit a DM adapter for **every (small layer i, large layer j)
pair**; each small layer's best R² over a **depth-matched band** of large
layers tells us whether that layer still lives in a large-translatable
subspace. The small-model layer where this best-match R² collapses — on
hallucination prompts but **not** on a matched control set — is the layer where
the small model's representation *permanently diverges*: the candidate
root-cause layer.

**Why the target layer j is restricted.** An unrestricted `argmax` over j is
degenerate. Early residual-stream layers are near-deterministic functions of
token identity, and the small model retains that information at *every* depth,
so "best over all j" collapses onto the shallow end: for the llama pair, `j=0`
won for **16 of 17** small layers (mean R² 0.91 at `j=0` vs a median of 0.67
across `j≥1`), and the qwen pair pinned to `j=3`. That inflates the curve, hides
the fork, and returns a target layer no one would actually stitch into. We
therefore constrain j to within `DEPTH_BAND` layers of i's *relative* depth
(`i/n_small × n_large`), exclude `j=0`, and exclude the final hidden state —
which is the output of `model.norm`, not a residual stream, and is
scale-discontinuous with the rest (llama-3B RMS climbs smoothly 0.018→0.525 over
layers 0–27, then jumps to 1.622 at layer 28). Naive half-fixes are worse than
nothing: merely excluding `j=0` (or `j≤3`) moves the llama answer to layer 16,
the last layer — the usual signature of a degenerate criterion. All parameters
live in the selection block of `diagnosis/config.py`.

### Cross-family pairs: prompt-level (final-token) alignment

Llama and Qwen use **different tokenizers**, so for the `llama2qwen` and
`qwen2llama` pairs the same prompt yields different token sequences of
different lengths — token-position alignment is impossible. These pairs
(`align="prompt"` in `diagnosis/config.py`) instead align at the **prompt level**:
both models answer the same question, and we keep exactly **one hidden-state
row per prompt per model — the final answer-generating token position**
(`is_last`). The DM ridge fit is unchanged in form; it just pairs rows by
prompt instead of by token position.

**Sample-size caveat (important).** One row per prompt means the divergent set
gives only ~12–43 rows against 896–2048 input dimensions — the ridge fit is
badly **underdetermined** (the pipeline prints an explicit `[warn]` when a
set has fewer rows than input dims). Treat the cross-family numbers as an
indicative **translatability curve**, not a well-powered divergent-vs-control
test. Ways to strengthen it:

* widen the row set — fit the DM grid on the much larger `control_both_right`
  set (~90–120 prompts) or on the **full prompt bank**, and read the result as
  a layer-wise cross-family translatability curve;
* grow the divergent set with harder prompts the large model still answers;
* interpret the divergent-vs-control *gap* only qualitatively at this sample
  size.

### CKA cross-check (no fitted map, well-defined at any sample size)

The DM adapter asks a *predictive* question — can a ridge-fit affine map
translate layer i into layer j? — and that fit degrades exactly where the
question matters most here (few rows vs many dims). As a complementary
readout the `cka` step computes **debiased linear CKA** (Centered Kernel
Alignment; Kornblith et al. 2019, with the unbiased HSIC estimator of Song et
al. 2012) for every (small layer, large layer) pair, on the same saved states:

* **no fitted map and no train/test split** — CKA compares the two layers'
  Gram matrices directly, so it cannot be underdetermined the way a
  regression can;
* **dimension-agnostic and symmetric** — well-suited to cross-family pairs
  with unrelated widths and tokenizers (rows are paired the same way as for
  DM: by token position within-family, by final answer token cross-family);
* **debiased estimator** — the divergent and control sets have very different
  sizes, and the biased estimator's O(1/n) offset would masquerade as a
  divergent-vs-control gap.

The readout mirrors the DM one (best-match curve per small layer, divergent
vs control), except there is no "onset": CKA measures geometry similarity,
not translatability collapse, so the single readout is the layer with the
most negative divergent-minus-control gap. Layer 0 of the last-token rows is
degenerate by construction (every row is the same chat-template token, so the
embedding Gram has ~zero variance) and is reported as NaN.

## Layout

Two independent projects share one folder, plus a standalone timing benchmark:

```
common/              leaf utilities all projects import
  model_utils.py     model loading / chat formatting / generation / hidden states
  decoding.py        sliceable layer loop, KV-cached baselines, greedy decode
  scoring.py         answer scoring: ANY-alias and ALL-items (conjunctive)
  stats.py           Wilson intervals, paired bootstrap, the CI-separation rule
  templates.py       system-prompt / question-framing variety for state capture
  fit_corpus.py      generic instruction corpus for fitting, disjoint from evals
  prompts.py             bank `factual`      — 140 short factual questions
  prompts_list.py        bank `list`         — 180 multi-item, conjunctive
  prompts_hard_factual.py bank `hard_factual` — 422 obscure single facts
  prompts_list_hard.py   bank `list_hard`    — 554 conjunctive 3-fact questions
                         composed from the audited hard_factual pool; dev/test
                         173 each, split over FACTS so none crosses a boundary
diagnosis/           Question 1 — where do the two models' representations diverge?
  run.py             pipeline entry point (--pair llama|qwen|llama2qwen|qwen2llama)
  config.py          model pairs (incl. align mode), hyperparameters, result paths
  select_prompts.py  step 1 — bucket prompts into divergent vs control
  extract_states.py  step 2 — paired hidden states (token- or prompt-aligned)
  train_dm.py        step 3 — fit the full layer x layer DM ridge-regression grid
  analyze.py         step 4 — verdict (printed + verdict.txt) + figures
  cka.py             step 5 — debiased linear CKA grid (map-free cross-check)
  fit_adapter.py     step 6 — materialise + save the selected DM map W, b
  results/<pair>/    outputs, one folder per pair
stitching_small_to_large/   LATENCY: small early → adapter → large late
  run.py             headroom / capture / fit / bench / sweep / report / compare / final / check
  config.py data.py adapter.py stitch.py evaluate.py
  distill.py         KL-to-teacher training through the frozen suffix blocks
  results/<pair>/<bank>/     states, adapters, benches, sweep tables
  README.md          method, how to read the numbers, measured results
stitching_large_to_small/   ACCURACY: large early → adapter → small late
  run.py             headroom / capture / fit / check / bench / sweep / final
  config.py data.py adapter.py stitch.py evaluate.py
  results/<pair>/<bank>/     states, adapters, checks, benches, sweeps, tables
  README.md          method, plumbing checks, honest limits
benchmark/           standalone model load / inference timing across GPUs
tests/
  smoke_synthetic.py synthetic extract->train->analyze->cka->fit_adapter (no weights)
  smoke_distill.py   synthetic distillation: frozen LLMs, warm start restored,
                     saved map reproduces training logits, and the training
                     forward == the warm-mode inference forward (no weights)
```

The two projects are independent: `stitching_small_to_large/` does not read anything under
`diagnosis/results/`, and both import from `common/` rather than from each other.

Each `diagnosis/results/<pair>/` folder has the same shape:

| Step | Output |
|------|--------|
| `select` | `generations.csv`, `selection.json` — greedy-generate + score both models; bucket into **divergent** (small-wrong / large-right) and **control** (both-right) |
| `extract` | `states/*.npz` — paired hidden states, all layers of both models: token-aligned pairs keep the last 64 positions/prompt; prompt-aligned pairs keep 1 row/prompt (the final answer-generating token) |
| `train` | `dm/*.npz`, `dm/dm_summary.json` — (small+1)x(large+1) DM residual grid, prompt-level train/test split; best-match R² per small layer + divergence layer |
| `analyze` | `figures/*.png` — divergence curve (divergent vs control) and R²(i→j) heatmaps — plus `verdict.txt`, the per-layer table and printed verdict saved alongside the figures |
| `cka` | `cka/*.npz`, `cka/cka_summary.json`, `figures/cka_*.png`, `verdict_cka.txt` — debiased linear CKA(i,j) grids and best-match curves, same divergent-vs-control readout without any fitted map |
| `fit_adapter` | `adapters/adapter_i{i}_j{j}.npz` (`W`, `b`, `mu_x`, `sd_x`, `mu_y`) + `.json` sidecar with provenance and held-out map quality |

Stitching results live under each package's own `results/<pair>/<bank>/` — see
those folders' READMEs for their layout.

## Running

Set up the environment once from the repository root (see the top-level
`README.md`). Both projects are packages, so run them as modules from inside
`experiments/palaash/`:

```bash
cd experiments/palaash

# ── diagnosis ────────────────────────────────────────────────────────────────
python -m diagnosis.run                    # the five diagnosis steps, llama pair
python -m diagnosis.run --pair qwen        # the five diagnosis steps, qwen pair
python -m diagnosis.run --pair llama2qwen  # cross-family, prompt-aligned
python -m diagnosis.run train analyze      # re-fit + re-plot from saved states
python -m diagnosis.run cka --pair qwen    # CKA cross-check from saved states
python -m diagnosis.fit_adapter --pair llama   # materialise the selected map

# ── stitching, small → large: LATENCY ────────────────────────────────────────
python -m stitching_small_to_large.run headroom --pair llama
python -m stitching_small_to_large.run sweep    --pair llama --modes exit warm
python -m stitching_small_to_large.run report   --pair llama --split dev  # no GPU

# ── stitching, large → small: ACCURACY ───────────────────────────────────────
python -m stitching_large_to_small.run headroom --pair llama  # is the bank usable?
python -m stitching_large_to_small.run capture  --pair llama
python -m stitching_large_to_small.run sweep    --pair llama

python tests/smoke_synthetic.py            # synthetic smoke test, no weights
python tests/smoke_distill.py              # synthetic distillation smoke test
```

`fit_adapter` is not in the default diagnosis run: it acts on the diagnosis
rather than producing it, and it applies to token-aligned pairs only. It
defaults to the (i, j) the analysis selected — `divergence_layer_small` and its
depth-matched target from `dm_summary.json` (llama: **1B L12 → 3B L18**).
Override with `--i` / `--j`; `j=0` and `j=n_layers_large` are rejected outright,
since neither is an injectable residual stream.

The `select` and `extract` steps download/use the models via your Hugging Face
cache (the Llama models are gated — request access on the model pages; the Qwen
models are open). They share a single model load when run together. Individual
steps also run as modules with the default pair, e.g. `python -m diagnosis.train_dm`.

Note that `*.npz` is gitignored, so the large state and adapter tensors under
either project's `results/` are not tracked. Re-generate them with
`python -m diagnosis.run select extract` and `python -m stitching_small_to_large.run capture`.

## How to read the result

* **`results/<pair>/figures/divergence_curve.png`** is the headline. Best-match
  R² starts high (early layers of the small model are linearly translatable
  into the large one), then on the divergent set it falls off at some layer
  while the control set stays higher. That fork is the divergence / root-cause
  layer, also printed by the analyze step and stored as
  `divergence_layer_small` in `dm_summary.json`.
* The **heatmaps** show *which* large-model layer each small-model layer maps
  into best — early layers map to early layers along a diagonal; the diagonal
  breaks where the representations stop corresponding.

## Results from the bundled runs

Prompt bank = 140 facts; headline metric = held-out R² at the final answer
token, DM grids averaged over 6 prompt-splits.

### llama (1B vs 3B)

Greedy accuracy **1B 87.1% / 3B 94.3%** → **12 divergent** prompts, 120
both-right controls.

| 1B layer | → 3B layer | best-R² divergent | best-R² control | gap |
|---:|---:|---:|---:|---:|
| 1–8 | 1–11 | 0.97–1.00 | 0.98–1.00 | −0.008…+0.003 |
| 9 | 12 | 0.950 | 0.958 | −0.008 ← onset |
| 10 | 14 | 0.775 | 0.835 | −0.060 |
| 11 | 16 | 0.561 | 0.708 | −0.147 |
| **12** | **18** | **0.392** | **0.556** | **−0.163 ← max (root-cause candidate)** |
| 13–14 | 19–21 | 0.23–0.33 | 0.39–0.49 | −0.156…−0.157 |
| 15–16 | 27 | 0.16–0.18 | 0.27–0.28 | −0.094…−0.126 |

(Layer 0 is the embedding table; it has no meaningful depth-matched counterpart,
so the onset scan starts at layer 1. See `ONSET_SCAN_START` in `diagnosis/config.py`.)

**Answer to Q1 — qualified yes.** Alignment *localises* where the small model
goes wrong: through layers 1–8 the 1B representation is near-perfectly linearly
translatable into the depth-matched 3B geometry on **both** hallucination and
control prompts (the early representations are *not* the cause); translatability
collapses from layer 9 for all prompts (a generic depth/specialisation effect);
and on hallucination prompts it collapses **further than on matched controls**,
peaking at layer 12 — the candidate root-cause layer, whose depth-matched
counterpart is **3B layer 18**, the layer an adapter would stitch into. The
robust part is the *structure* (flat-then-fork); the gap rests on only 12
divergent prompts.

Layer 12 is the same answer the earlier unrestricted-`argmax` rule gave, but it
now rests on a real signal rather than on shallow-layer leakage: the gap at the
peak is **−0.163 rather than −0.048**, the curve decays smoothly (0.998 at
layer 1 → 0.176 at layer 16) instead of sitting flat near 1.0, and the target
layer is 18 rather than the leakage-selected 11.

### qwen (0.5B vs 3B)

Greedy accuracy **0.5B 65.0% / 3B 94.3%** → **43 divergent** prompts, 89
both-right controls (the much weaker 0.5B gives ~3.5× more divergent cases than
the llama pair, so the gap estimate is better supported).

| 0.5B layer | → 3B layer | best-R² divergent | best-R² control | gap |
|---:|---:|---:|---:|---:|
| 0–14 | 1–18 | 0.993–0.999 | 0.994–1.000 | ~0 |
| 15–16 | 21 | 0.959–0.968 | 0.973–0.984 | −0.014…−0.015 |
| 17 | 22 | 0.943 | 0.962 | −0.018 ← onset |
| 18–20 | 24–27 | 0.86–0.91 | 0.93–0.96 | −0.050…−0.069 |
| 21 | 28 | 0.702 | 0.792 | −0.090 |
| 22–23 | 35 | −0.05…−0.03 | 0.10–0.13 | −0.151…−0.156 |
| **24** | **35** | **−0.029** | **0.200** | **−0.229 ← max (root-cause candidate)** |

The qwen pair **replicates the llama finding**: early/mid layers are
near-perfectly translatable on both sets, then the curves fork in the last
quarter of the network — onset at layer 17 of 24 (71% depth; llama: 9 of 16,
56%), with the hallucination-specific gap growing monotonically to a maximum
near the top. The root-cause signature — a late-layer, hallucination-specific
loss of translatability — appears in both model families, and under the
depth-matched rule the effect is several times larger in both (qwen −0.229,
llama −0.163) than the ≈−0.05 the unrestricted `argmax` reported.

Two differences from llama worth naming: qwen's collapse is **sharper and
later** (translatability holds above 0.94 until layer 17, then falls to ~0
between layers 21 and 22), and its peak gap lands on layer **24, the last small
layer**. A criterion peaking at the final layer is usually a warning sign, but
here it is not an artefact of the boundary — the divergent curve has already
gone to zero by layer 22 while the control curve retains 0.10–0.20, so the fork
is genuine across the whole 22–24 block rather than an edge effect at 24.

### Cross-family results (indicative — prompt-level alignment)

Both cross-family pairs were run on the same 140-fact bank with prompt-level
(final-token) alignment: one held-out answer row per test prompt, fit against
896–2048 input dims, so these are **underdetermined** — read them as a
qualitative cross-family translatability curve, not a powered divergent-vs-control
test. Absolute R² is far below the same-family pairs (~0.5–0.8 vs ~0.99) because
the two families have unrelated tokenizers and geometries; the only question is
whether the **divergent curve sits below the control curve more in the late
layers** than early, as it does within-family.

**`qwen2llama` (Qwen-0.5B → Llama-3B) — weak but present late-layer fork.**
Accuracy 65.0% / 93.6% → **45 divergent**, 86 control (14 held-out answer rows —
the best-powered cross-family direction).

| Qwen-0.5B layer | → Llama-3B layer | best-R² divergent | best-R² control | gap |
|---:|---:|---:|---:|---:|
| 1–4 | 1 | 0.45–0.60 | 0.54–0.67 | −0.07…−0.10 |
| 5–8 | 3–6 | 0.45–0.55 | 0.66–0.68 | −0.12…−0.21 ← widening |
| **9** | **8** | **0.461** | **0.688** | **−0.227 ← max** |
| 10–16 | 8–15 | 0.36–0.49 | 0.57–0.67 | −0.15…−0.22 |
| 17–24 | 16–25 | 0.00–0.24 | 0.13–0.40 | −0.10…−0.23 |

The divergent-minus-control gap is smallest in the first fifth (~−0.07 to −0.10),
roughly **doubles by layer 9** (−0.227) and stays in the −0.13…−0.23 band for the
rest of the network while *both* curves decay toward zero. Even translating
*across families* the extra loss of translatability on hallucination prompts
shows up early and persists — but note this pair's peak sits at layer 9 of 24
rather than late, so it does **not** reproduce the same-family late-layer
localisation; what survives crossing families is the existence of a
hallucination-specific gap, not its depth.

**`llama2qwen` (Llama-1B → Qwen-3B) — too underpowered to read.** Accuracy
87.9% / 94.3% → only **12 divergent**, 120 control, which leaves ~**4 held-out
answer rows**. Control best-R² is ~0.36–0.80 and divergent ~0.02–0.27, but the
gap is a roughly **uniform ≈−0.34…−0.59 across every layer with no fork** — that
offset is the cross-family geometry mismatch on a 4-row test set, not a
hallucination-specific divergence, and the per-layer "max gap at layer 1" is
noise. This direction needs a wider row set (fit on the control set / full bank,
or grow the divergent set with harder prompts the large model still answers)
before its curve means anything. It is included for completeness and to make the
underdetermination concrete.

**Takeaway.** A hallucination-specific loss of translatability survives crossing
model families in the better-powered `qwen2llama` direction, but its *depth*
does not — under depth-matched selection that pair peaks at layer 9 of 24, not
in the late layers — and `llama2qwen` is too small to interpret at all. The
late-layer localisation is a same-family result; that is the robust one.

### CKA results (map-free cross-check on the same states)

Debiased linear CKA on the final answer-token rows, best match over all large
layers, divergent-minus-control gap averaged over depth thirds of the small
model (full per-layer tables in `results/<pair>/verdict_cka.txt`):

| pair | n divergent | gap: early third | mid third | late third | max gap (layer) |
|------|---:|---:|---:|---:|---|
| `qwen` | 43 | −0.015 | −0.038 | −0.069 | **−0.095 (L22)** |
| `qwen2llama` | 45 | −0.001 | −0.031 | −0.075 | **−0.106 (L24)** |
| `llama2qwen` | 12 | +0.102 | −0.011 | −0.040 | −0.051 (L15) |
| `llama` | 12 | +0.019 | +0.020 | +0.026 | −0.015 (L9) |

Three observations:

* **`qwen`: the DM finding is corroborated by an independent method.** The
  CKA gap widens monotonically with depth and is largest at **layer 22 — the
  layer at which the depth-matched DM curve collapses** (divergent best-R² falls
  from 0.702 at layer 21 to −0.054 at layer 22; the DM gap then keeps widening
  to its maximum at layer 24). Two very different estimators (a fitted ridge
  translator vs a map-free geometry statistic) put the hallucination-specific
  divergence in the same 22–24 block. Note the CKA numbers below were computed
  with CKA's own unrestricted best-match over j (`diagnosis/cka.py`), which is *not*
  subject to the DM collapse — its argmax already increases monotonically with
  depth — so it is an independent check, not the same rule reapplied.
* **Cross-family pairs become interpretable.** CKA needs no fitted map, so
  the underdetermination that made the DM cross-family numbers "indicative
  only" does not apply. `qwen2llama` shows the same clean late-layer widening
  (max −0.106 at layer 24), and even `llama2qwen` — unreadable under DM (a
  uniform −0.55 offset on a 4-row test set) — now shows the late-layer sign:
  the gap moves from *positive* in the early third to −0.04/−0.05 in the
  final quarter (still only 12 divergent prompts, so read it as a trend).
* **`llama` is the honest null.** With only 12 divergent prompts the CKA gap
  is small and slightly *positive* at all depths — CKA neither confirms nor
  contradicts the DM fork for this pair; 12 final-token rows is simply below
  what a geometry statistic can resolve. This is the right caveat to attach
  to the llama DM gap (−0.163) as well.

Overall the CKA cross-check *strengthens* the headline claim where the data is
adequate (qwen, qwen2llama: late-layer hallucination-specific divergence,
peaking at the same depth as DM) and correctly exposes the llama-pair sample
size as the weak point.

## Two stitching projects, in opposite directions

Stitching used to live in this folder as two `diagnosis` pipeline steps
(`stitch`, `stitch_fast`); both were removed. It is now **two sibling packages
that point in opposite directions and optimise different things**:

| package | direction | primary goal | is it faster? |
| --- | --- | --- | --- |
| [`stitching_small_to_large/`](stitching_small_to_large/README.md) | small → large | **latency** — decode faster than the large model | yes, that is the point |
| [`stitching_large_to_small/`](stitching_large_to_small/README.md) | large → small | **accuracy** — beat the small model without fine-tuning either LLM | **no** — it runs part of both models |
| [`diagnosis/`](diagnosis/) | neither | analysis — where the two models' representations diverge | n/a |

```
small_to_large:  prompt → small blocks 0..i-1 → adapter → large blocks j..end → token
large_to_small:  prompt → large blocks 0..j-1 → adapter → small blocks i..end → token
```

Both are KV-cached on both sides, both are token-aligned pairs only (`llama`,
`qwen`), and both capture their own states and fit their own adapters — neither
reads anything `diagnosis/` produces. They share `common/`: the prompt banks,
the scorer, model loading, and `common/decoding.py`, which holds the sliced
layer loop both need (one verified copy, since it is a hand transcription of
`LlamaModel.forward` and duplicating it would duplicate the risk).

**Read them as separate claims.** `small_to_large` trades accuracy for speed;
`large_to_small` trades speed for accuracy. A number from one says nothing about
the other.

### The lesson both inherit from the retired code

The old `stitch_fast` collapsed after the first generated token — `Wellington` →
`"Well is"`, `Jefferson City` → `"Jeffapolis"` — while its adapter reported a
healthy held-out R² of 0.91. The fit set was the bug, not the layer choice:
states were captured over **prompt tokens only**, but at decode time the adapter
is handed the model's state at positions holding *generated answer* tokens. On
those, the same adapter scored **R² = 0.34**. Both packages now capture prompt
**+ teacher-forced answer** positions, and both report answer-token quality
separately from all-token quality, because the all-token number is the one that
hid the failure.

### Current results, including the negative one

**`small_to_large` (latency): negative, and it stays on the record.** Across a
50-point llama grid and a 32-point qwen grid, *every* configuration was dominated
by the small model — both slower and less accurate than just running the 1B (or
0.5B). The small model decodes 2.4x (llama) / 4.8x (qwen) faster than the large
one while giving up only 6-9 / 20 accuracy points, and no stitch beat that trade.

**`large_to_small` (accuracy): also negative.** On a purpose-built bank with
real headroom (`hard_factual`: llama small 73.6%, large 98.1%, 26 divergent
prompts), all 32 grid cells failed. The best reached 57.5% — 16 points *below*
the small model alone — while costing 1.7x its decode time plus a partial
large-model forward, and recovering 0-11% of the divergent prompts. The stitched
path reproduces the small model's own wrong answers and corrupts some it would
otherwise get right; the adapter's held-out answer-token R² tops out at 0.50 and
goes negative in 6 cells, which is not faithful enough to carry a specific fact
through the small model's remaining blocks.

That bank exists because the older ones gave a 1B model too little to get wrong:
the original `factual` bank leaves 8.6 points of headroom, of which 3 prompts on
a 35-prompt split are divergent, which cannot support any conclusion either way.
`headroom` is the command that checks this before a sweep is worth running.

**Both directions are negative on these pairs.** That is a real result about
linear stitching between a 1B and a 3B, not a bug: the plumbing is verified by
checks that gate every report (identity-injection at `rel_l2 = 3.4e-05`, an
off-by-one control that correctly differs), so what failed is the method, not
the harness.

### …but the instrument that produced them could not have shown a positive one

Re-reading those campaigns turned up five defects in the *measurement*, four of
which are independent of whether stitching works. They are fixed now, and the
negative results above are retained unchanged as the evidence they were
measured against.

| # | defect | fix |
| --- | --- | --- |
| 1 | The adapter was fit on the wrong objective — ridge minimises L2 over all 3072 residual dimensions, but next-token identity lives in the few that survive `norm` + `lm_head`. `R2_all = 0.999` vs `R2_answer` ∈ [−0.20, 0.375], and answer R² does not rank the cells. | `--train-method distill`: KL to the teacher's next-token distribution, backpropagated through the frozen suffix, warm-started from ridge. |
| 2 | The fit set was 95% boilerplate: 258 answer rows against 5180 prompt rows of one repeated template, i.e. **17%** of the objective at `ANSWER_WEIGHT=4`. | Template variety, answer-majority row selection (asserted, recorded as `answer_weight_frac`), and `--fit-corpus generic` for volume. |
| 3 | The latency claim was not computed honestly: the sweep CSV had **no end-to-end column**, so the best cell was tabulated at 1.11x decode while running **0.93x end-to-end**. | Every row carries `end_to_end_speedup_vs_large` with a named pricing source; prefill printed per row; a cross-cell prefill-anomaly flag. |
| 4 | The benchmark could not show a win: 35-prompt splits give a ~23-point interval across an 8.6-point gap. | `list_hard`, 173 prompts per eval split (~11-point interval), +15.0 pts of headroom with **separated** intervals. |
| 5 | The bar was set against the wrong incumbent — cells were ranked against each other, not against the small model. | The small model competes in the Pareto frontier, on end-to-end speed, with interval-separated accuracy. No recommendation is emitted otherwise. |

The clearest single symptom of defects 3–5 together: the `factual` grid's
best-accuracy cell, `L10→L10` warm, was reported as a 1.11x speedup. Priced
end-to-end at the large model's true 3.46-token mean answer it runs **0.92x** —
slower than the model it was meant to beat — and its accuracy interval
(70.6–93.7%) overlaps the small model's so heavily that the comparison was never
resolvable. All 50 cells were dominated by the small model on both axes.

### Re-run on the repaired instrument (llama, `list_hard`, MPS, 2026-08-18)

| path | accuracy (95% CI) | vs small (paired bootstrap) | decode | end-to-end |
| --- | --- | --- | --- | --- |
| 1B alone | 83.2% [77.0–88.1] | — | 2.45x | **2.46x** |
| 3B alone | 98.3% [95.0–99.4] | +15.0 | 1.00x | 1.00x |
| best ridge (`L10→L14` warm) | 89.6% [84.2–93.3] | +6.4 [−0.6, +13.9] | 1.28x | 1.17x |
| best distill (`L10→L14` warm) | 89.6% [84.2–93.3] | +6.4 [−0.6, +13.3] | 1.28x | 1.16x |
| `L8→L14` warm, ridge | 75.7% [68.8–81.5] | −7.5 [−15.6, +0.6] | 1.34x | 1.22x |
| `L8→L14` warm, **distill** | 86.1% [80.2–90.5] | +2.9 [−4.6, +10.4] | 1.34x | 1.21x |

**Distillation beats ridge by +10.4 points at identical geometry and identical
decode cost** (`L8→L14`, paired bootstrap [+5.8, +15.6], excludes zero) — same
6.3M-parameter affine map, only the objective changed. At `L10→L14` it makes no
difference, because ridge is already good enough there for greedy decoding to
pick the same tokens.

**No configuration beats the small model on both axes**, and the reason is now
legible rather than hidden: the best stitch is 1.17x faster than the *large*
model but the small model is 2.46x faster, so the stitch loses the speed axis
outright, and its +6.4-point accuracy edge does not exclude zero. That is a far
narrower failure than the `factual` campaign's — where every cell lost on both
axes — but it is still a failure, and `sweep` says `NO VIABLE POINT`.

`diagnosis/fit_adapter.py` remains, and is **not** stitching code: it
materialises the map the DM analysis selected (the diagnostic grid solves for R²
through a hat matrix and never forms `W`), so the selected map can be inspected
and regression-tested.

## Honest caveats

* The DM adapter is a *linear* translator. Low R² means "not linearly
  translatable," which is the standard operationalisation of representational
  divergence, but a nonlinear map could recover more. (Question 2's nonlinear
  projection head is the natural follow-up.) The CKA step partially hedges
  this: it is also a linear-kernel statistic, but it requires no fit at all.
* The llama pair's DM gap (−0.163 at layer 12) is **not corroborated by the
  CKA cross-check** — with 12 divergent prompts the CKA gap is small and
  slightly positive throughout. Treat the llama root-cause layer as the
  weakest of the headline numbers; the qwen pair (43 divergent prompts, DM and
  CKA agreeing on the 22–24 block) is the one to lead with.
* **The llama pair's answer-token sample is very small.** `n_test_last_tokens`
  is **4** for the divergent set — four held-out answer positions, from 12
  divergent prompts — and `r2_last` is the only metric that shows the effect
  (`r2_all` gaps span only −0.010…+0.025 and turn *positive* from layer 12
  onward — the divergent set fits slightly better than the control there — so
  that metric cannot localise anything and should not be quoted for the fork;
  the figures now plot `r2_last`, matching the verdict). The depth-matched
  −0.163 is a far better signal than the −0.048 the old rule gave, but it still
  rests on very little data. Before building on "layer 12 → 18", expand
  `common/prompts.py`: at the current 87% small-model accuracy, reaching 50+
  divergent cases needs roughly **400–600 prompts**. This is independent of the
  broad fitting corpus an adapter would need, and it is the change most likely
  to move these numbers.
* Target-layer selection is a **choice**, not a measurement. `DEPTH_BAND=3` is
  a reasonable default (it produces a smoothly decaying curve and picks
  plausible stitching targets), but the peak layer's exact index is somewhat
  sensitive to it; the flat-then-fork *structure* is not.
* The chat prompt repeats a fixed system message, so some token positions are
  shared across prompts; that inflates absolute R² roughly uniformly across
  layers but does not move the *fork* between the divergent and control curves,
  which is what the verdict relies on. The headline metric is evaluated on the
  held-out **final answer token** of each test prompt.
* Absolute R² depends on the ridge strength (`RIDGE_ALPHA` in `diagnosis/config.py`);
  the cross-layer shape and the divergent-vs-control gap are the robust signals.
* The cross-family pairs (`llama2qwen`, `qwen2llama`) use prompt-level
  alignment: one row per prompt, so their fits are far more underdetermined
  than the token-aligned pairs — see the caveat in the prompt-level alignment
  section above.
* When the small model is a strong factual recaller the divergent set is small
  (12 prompts for llama). Split-averaging stabilises the curve, but the gap
  magnitude should be treated as indicative. To strengthen it, add harder
  prompts that the large model still gets right (grow the divergent set) and/or
  a held-out prompt domain.
