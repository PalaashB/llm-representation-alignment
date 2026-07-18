# Aligning small ↔ large LLM hidden states

Research code for studying representational alignment between a small and a
large instruction-tuned model. Four pairs are configured (`q1/config.py`):
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
pair**; each small layer's best R² over all large layers tells us whether that
layer still lives in a large-translatable subspace. The small-model layer where
this best-match R² collapses — on hallucination prompts but **not** on a
matched control set — is the layer where the small model's representation
*permanently diverges*: the candidate root-cause layer.

### Cross-family pairs: prompt-level (final-token) alignment

Llama and Qwen use **different tokenizers**, so for the `llama2qwen` and
`qwen2llama` pairs the same prompt yields different token sequences of
different lengths — token-position alignment is impossible. These pairs
(`align="prompt"` in `q1/config.py`) instead align at the **prompt level**:
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

```
run_q1.py            pipeline entry point (--pair llama|qwen|llama2qwen|qwen2llama)
q1/
  config.py          model pairs (incl. align mode), hyperparameters, result paths
  prompts.py         bank of 140 checkable factual questions (shared by all pairs)
  scoring.py         normalised substring answer scoring
  model_utils.py     model loading / generation / hidden-state extraction
  select_prompts.py  step 1 — bucket prompts into divergent vs control
  extract_states.py  step 2 — paired hidden states (token- or prompt-aligned)
  train_dm.py        step 3 — fit the full layer×layer DM ridge-regression grid
  analyze.py         step 4 — verdict (printed + verdict.txt) + figures
  cka.py             step 5 — debiased linear CKA grid (map-free cross-check)
tests/
  smoke_synthetic.py synthetic extract→train→analyze→cka smoke test (no model weights)
results/
  llama/  qwen/            outputs for the same-family pairs
  llama2qwen/  qwen2llama/ outputs for the cross-family pairs (same folder shape)
```

Each `results/<pair>/` folder has the same shape:

| Step | Output |
|------|--------|
| `select` | `generations.csv`, `selection.json` — greedy-generate + score both models; bucket into **divergent** (small-wrong / large-right) and **control** (both-right) |
| `extract` | `states/*.npz` — paired hidden states, all layers of both models: token-aligned pairs keep the last 64 positions/prompt; prompt-aligned pairs keep 1 row/prompt (the final answer-generating token) |
| `train` | `dm/*.npz`, `dm/dm_summary.json` — (small+1)×(large+1) DM residual grid, prompt-level train/test split; best-match R² per small layer + divergence layer |
| `analyze` | `figures/*.png` — divergence curve (divergent vs control) and R²(i→j) heatmaps — plus `verdict.txt`, the per-layer table and printed verdict saved alongside the figures |
| `cka` | `cka/*.npz`, `cka/cka_summary.json`, `figures/cka_*.png`, `verdict_cka.txt` — debiased linear CKA(i,j) grids and best-match curves, same divergent-vs-control readout without any fitted map |

## Running

Set up the environment once from the repository root (see the top-level
`README.md`), then run this pipeline from inside `experiments/palaash/`:

```bash
cd experiments/palaash
python run_q1.py                       # all four steps, llama pair
python run_q1.py --pair qwen           # all four steps, qwen pair
python run_q1.py --pair llama2qwen     # cross-family, prompt-aligned
python run_q1.py --pair qwen2llama     # cross-family, prompt-aligned
python run_q1.py train analyze         # re-fit + re-plot from saved states
python run_q1.py cka --pair qwen       # CKA cross-check from saved states
python tests/smoke_synthetic.py        # synthetic smoke test, no weights
```

Equivalently, from the repository root: `python experiments/palaash/run_q1.py …`
(Python adds the script's folder to the import path, so `import q1` resolves).

The `select` and `extract` steps download/use the models via your Hugging Face
cache (the Llama models are gated — request access on the model pages; the Qwen
models are open). They share a single model load when run together. Individual
steps also run as modules with the default pair, e.g. `python -m q1.train_dm`.

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

| 1B layer | best-R² divergent | best-R² control | gap |
|---:|---:|---:|---:|
| 0–9 | ~0.99–1.00 | ~0.99–1.00 | ~0 |
| 10 | 0.904 | 0.925 | −0.020 ← onset |
| 11 | 0.863 | 0.894 | −0.032 |
| **12** | **0.759** | **0.807** | **−0.048 ← max (root-cause candidate)** |
| 13–16 | 0.73–0.77 | 0.75–0.81 | −0.005…−0.045 |

**Answer to Q1 — qualified yes.** Alignment *localises* where the small model
goes wrong: through layers 0–9 the 1B representation is near-perfectly linearly
translatable into the 3B geometry on **both** hallucination and control prompts
(the early representations are *not* the cause); translatability collapses from
layer 10 for all prompts (a generic depth/specialisation effect); and on
hallucination prompts it collapses **further than on matched controls**,
peaking at layer 12 — the candidate root-cause layer. The robust part is the
*structure* (flat-then-fork); the absolute gap is small and rests on only 12
divergent prompts.

### qwen (0.5B vs 3B)

Greedy accuracy **0.5B 65.0% / 3B 94.3%** → **43 divergent** prompts, 89
both-right controls (the much weaker 0.5B gives ~3.5× more divergent cases than
the llama pair, so the gap estimate is better supported).

| 0.5B layer | best-R² divergent | best-R² control | gap |
|---:|---:|---:|---:|
| 0–14 | ~0.997–1.00 | ~0.998–1.00 | ~0 |
| 15–17 | 0.97–0.98 | 0.97–0.99 | −0.004…−0.008 |
| 18 | 0.945 | 0.972 | −0.027 ← onset |
| 19–21 | 0.87–0.95 | 0.89–0.97 | −0.013…−0.024 |
| **22** | **0.708** | **0.757** | **−0.049 ← max (root-cause candidate)** |
| 23–24 | 0.69–0.75 | 0.72–0.76 | −0.013…−0.029 |

The qwen pair **replicates the llama finding**: early/mid layers are
near-perfectly translatable on both sets, then the curves fork in the final
quarter of the network — onset at layer 18 of 24 (llama: 10 of 16, both ≈75%
depth) with the hallucination-specific gap peaking near the top (layer 22,
−0.049; llama: layer 12, −0.048). The root-cause signature — a late-layer,
hallucination-specific loss of translatability of strikingly similar magnitude
— appears in both model families.

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

| Qwen-0.5B layer | best-R² divergent | best-R² control | gap |
|---:|---:|---:|---:|
| 1–9 | 0.45–0.61 | 0.54–0.69 | −0.06…−0.10 |
| 10–12 | 0.57–0.59 | 0.67–0.68 | −0.08…−0.11 |
| 13–14 | 0.46–0.54 | 0.66–0.68 | −0.14…−0.20 ← widening |
| 15–22 | 0.37–0.48 | 0.59–0.65 | −0.18…−0.22 |
| **23** | **0.369** | **0.592** | **−0.223 ← max** |
| 24 | 0.369 | 0.590 | −0.221 |

The divergent-minus-control gap is small and roughly flat through the first half
(~−0.07 to −0.10), then roughly **doubles across the final third** (layers 13→24,
peaking −0.223 at layer 23). Even translating *across families* the extra loss of
translatability on hallucination prompts concentrates in the late layers —
qualitatively the same late-layer signature as the same-family pairs, at much
lower absolute R².

**`llama2qwen` (Llama-1B → Qwen-3B) — too underpowered to read.** Accuracy
87.9% / 94.3% → only **12 divergent**, 120 control, which leaves ~**4 held-out
answer rows**. Control best-R² is ~0.69–0.80 and divergent ~0.13–0.27, but the
gap is a roughly **uniform ≈−0.55 across every layer with no fork** — that offset
is the cross-family geometry mismatch on a 4-row test set, not a
hallucination-specific divergence, and the per-layer "max gap at layer 1" is
noise. This direction needs a wider row set (fit on the control set / full bank,
or grow the divergent set with harder prompts the large model still answers)
before its curve means anything. It is included for completeness and to make the
underdetermination concrete.

**Takeaway.** The late-layer, hallucination-specific loss of translatability
survives crossing model families in the better-powered `qwen2llama` direction,
but only qualitatively and at low absolute R²; the `llama2qwen` direction is too
small to interpret. The robust result remains the same-family one.

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
  exact layer the DM grid flagged as the root-cause candidate**. Two very
  different estimators (a fitted ridge translator vs a map-free geometry
  statistic) agree on where the hallucination-specific divergence lives.
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
  to the llama DM gap (−0.048) as well.

Overall the CKA cross-check *strengthens* the headline claim where the data is
adequate (qwen, qwen2llama: late-layer hallucination-specific divergence,
peaking at the same depth as DM) and correctly exposes the llama-pair sample
size as the weak point.

## Honest caveats

* The DM adapter is a *linear* translator. Low R² means "not linearly
  translatable," which is the standard operationalisation of representational
  divergence, but a nonlinear map could recover more. (Question 2's nonlinear
  projection head is the natural follow-up.) The CKA step partially hedges
  this: it is also a linear-kernel statistic, but it requires no fit at all.
* The llama pair's DM gap (−0.048 at layer 12) is **not corroborated by the
  CKA cross-check** — with 12 divergent prompts the CKA gap is small and
  slightly positive throughout. Treat the llama root-cause layer as the
  weakest of the headline numbers; the qwen pair (43 divergent prompts,
  DM and CKA agreeing on layer 22) is the one to lead with.
* The chat prompt repeats a fixed system message, so some token positions are
  shared across prompts; that inflates absolute R² roughly uniformly across
  layers but does not move the *fork* between the divergent and control curves,
  which is what the verdict relies on. The headline metric is evaluated on the
  held-out **final answer token** of each test prompt.
* Absolute R² depends on the ridge strength (`RIDGE_ALPHA` in `q1/config.py`);
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
