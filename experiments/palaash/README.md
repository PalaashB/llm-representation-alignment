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
live in the selection block of `q1/config.py`.

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
  fit_adapter.py     step 6 — materialise + save one DM map W, b for stitching
  stitch.py          step 7 — inject small layer i into the large model at block j
  stitch_fast.py     step 8 — the same injection as an early exit: skip large
                              blocks 0..j-1, KV-cached, latency + accuracy report
tests/
  smoke_synthetic.py synthetic extract→train→analyze→cka→fit_adapter smoke test (no weights)
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
| `fit_adapter` | `adapters/adapter_i{i}_j{j}.npz` (`W`, `b`, `mu_x`, `sd_x`, `mu_y`) + `.json` sidecar with provenance and held-out map quality |
| `stitch` | `stitch/checks_i{i}_j{j}.json` — injection plumbing checks; `stitch/stitch_i{i}_j{j}.json/.csv` — small-alone / large-alone / stitched generations |
| `stitch_fast` | `stitch/fast_checks_i{i}_j{j}.json` — early-exit equivalence checks; `stitch/fast_i{i}_j{j}.json/.csv` — per-prompt latency (prefill ms, ms/token) and gold-answer accuracy for small / large / both stitch modes |

## Running

Set up the environment once from the repository root (see the top-level
`README.md`), then run this pipeline from inside `experiments/palaash/`:

```bash
cd experiments/palaash
python run_q1.py                       # the five diagnosis steps, llama pair
python run_q1.py --pair qwen           # the five diagnosis steps, qwen pair
python run_q1.py --pair llama2qwen     # cross-family, prompt-aligned
python run_q1.py --pair qwen2llama     # cross-family, prompt-aligned
python run_q1.py train analyze         # re-fit + re-plot from saved states
python run_q1.py cka --pair qwen       # CKA cross-check from saved states
python tests/smoke_synthetic.py        # synthetic smoke test, no weights
```

Stitching is **not** part of the default run — it acts on the diagnosis rather
than producing it, and it only applies to token-aligned pairs. Ask for it by name:

```bash
python run_q1.py fit_adapter stitch          # save the adapter, then stitch with it
python run_q1.py fit_adapter stitch_fast     # ... then benchmark the early-exit stitch
python -m q1.fit_adapter --pair llama        # just fit + save W, b  (numpy only)
python -m q1.stitch --pair llama --check     # injection plumbing checks only
python -m q1.stitch --pair llama --prompt "What is the capital of New Zealand?"
python -m q1.stitch --pair llama --run-selected --n-control 6
```

There are **two** stitch steps and they make different claims. `stitch` is the
mechanism check: it runs both full models every decode step and is slower than
either alone. `stitch_fast` is the one whose point is speed — it skips large
blocks `0..j-1` entirely and decodes with a KV cache:

```bash
python -m q1.stitch_fast --pair llama --check          # early-exit equivalence checks
python -m q1.stitch_fast --pair llama                  # latency + accuracy benchmark
python -m q1.stitch_fast --pair qwen --run-selected    # every divergent prompt
python -m q1.stitch_fast --pair llama --mode exit      # one mode instead of both
```

Both default to the (i, j) the analysis selected — `divergence_layer_small` and
its depth-matched target from `dm_summary.json` (llama: **1B L12 → 3B L18**).
Override with `--i` / `--j`; `j=0` and `j=n_layers_large` are rejected outright,
since neither is an injectable residual stream.

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
so the onset scan starts at layer 1. See `ONSET_SCAN_START` in `q1/config.py`.)

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
  with CKA's own unrestricted best-match over j (`q1/cka.py`), which is *not*
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

## Stitching, part 1 (`stitch`) — the mechanism check

> **This path is a correctness demonstration, not a system.** It runs both full
> models at every decode step, so it is slower than either model alone; that was
> the accepted cost of making the injection obviously correct. The
> latency-improving version is [part 2](#stitching-part-2-stitch_fast--the-early-exit-path)
> below. Nothing here is superseded — `stitch` is still what establishes that the
> injection index and the adapter are right, and `stitch_fast` is checked
> against it.

The diagnosis says the 1B's layer 12 stops being translatable into the 3B's
layer 18. Stitching tests that claim by *doing* it: run the 1B, map its layer-12
residual stream through the adapter, overwrite the 3B's residual stream at the
input to block 18, and finish the 3B forward pass.

`train_dm.py` never materialises `W` — it solves each ridge system once and
pushes the *targets* through a hat matrix, which gives R² for all target layers
at once but discards the map. So `fit_adapter.py` re-solves the same ridge
problem explicitly for one (i, j) and saves `Y_hat = ((X - mu_x)/sd_x) @ W + b`.
Unlike the diagnostic it fits on **every** row of the chosen set (default:
control, 7680 rows) because the goal is a usable map, not an estimate; a
separate prompt-level refit reports honest held-out quality.

**Injection point.** In transformers 5.x `output_hidden_states=True` is served by
capture hooks that record the first layer's *input*, then each layer's *output*,
with the last entry overwritten by `model.norm(...)`. So `hidden_states[j]` is
exactly the input to block `j` for `j < n_layers` — which is what a forward
*pre-hook* on `model.layers[j]` replaces — and index `n_layers` is post-norm.
That is the same fact `DROP_POST_NORM_LAYER` encodes on the analysis side, here
reached from the other direction.

### Plumbing checks (all pass; `results/llama/stitch/checks_i12_j18.json`)

| check | result |
|---|---|
| **null hook** — hook registered, returns stream unchanged | bit-identical to baseline (max abs Δlogit `0.0`) |
| **identity injection** — inject the 3B's own `hidden_states[18]` | bit-identical to baseline (max abs Δlogit `0.0`) |
| **off-by-one** (negative control) — inject `hidden_states[19]` at block 18 | differs (max abs Δlogit `10.84`), so the identity check is not vacuous |
| **adapter quality** on a held-out prompt split | cosine `0.972`, rel-L2 `0.087`, R² `0.910` |

The off-by-one control matters: without it, "identity injection reproduces the
baseline" would also pass if the hook were silently doing nothing.

### The attention sink breaks naive full-stream injection

Overwriting **all** positions produces pure noise (`'Lri and and and and unhing…'`)
on *every* prompt, controls included. The cause is a single position. At 3B layer
18 the position-0 residual has norm **≈762** against **≈12–23** everywhere else —
the attention-sink / massive-activation token — and the adapter under-predicts it
by more than 2× (≈327). It is also outside the fit: `LAST_K=64` keeps only the
final 64 rows per prompt while these prompts run 72–73 tokens, so BOS was never
a fitted row. Overwriting it destroys attention globally.

Preserving position 0 alone fully restores coherent output. Sweeping the
preserved prefix over `{0, 1, 2, 4, 8, 16, 32}` gives noise at 0 and *identical*
output for every value ≥ 1, so `PRESERVE_PREFIX = 1` is a correctness
requirement, not a tuned hyperparameter. The `sink_position` check records the
norm ratio on each run.

### Result (llama, 1B L12 → 3B L18, 12 divergent + 6 control prompts)

v1 is scoped to prefill + the first generated token, so that is what is scored:

| | n | first token == 3B | first token == 1B | full text == 3B |
|---|---:|---:|---:|---:|
| divergent | 12 | 4 | **0** | 0 |
| control | 6 | 6 | 6 | 1 |

**The stitch path works and moves computation toward the large model.** On the
divergent prompts — the only ones that discriminate, since 1B and 3B agree on
controls by construction — the stitched model reproduces the 1B's own first
token **0 out of 12 times**, and the 3B's **4 out of 12**. Text past the first
token drifts (`'Ott…'`→`'Ottfoundland'`, `'An…'`→`'Anstanbul'`), which is what a
lossy linear reconstruction of one layer should do.

**This is not hallucination fixing and is not claimed as such.** The stitched
output lands on neither model's answer in 8 of 12 divergent cases. What is
established is narrower and is the actual v1 goal: the adapter persists and
reloads, the injection point is verified against three independent checks, and a
stitched forward pass runs end-to-end and demonstrably carries large-model
information rather than passing the small model through. Latency/accuracy
benchmarking is deliberately out of scope here.

Known v1 limits, in rough priority order:

* **Ridge shrinkage systematically under-predicts the residual norm** (mean
  ‖pred‖ 15.8 vs ‖true‖ 23.1 on a live prompt) — a norm-matching or
  whitened-fit variant is the obvious next change. *Still open, and now the
  binding constraint: part 2 shows the plumbing is no longer what limits
  quality.*
* **`LAST_K=64` truncation** means early positions of longer prompts were never
  fitted; extraction should cover full sequences. *Still open.*
* **No KV cache.** Greedy decoding re-runs both models over the whole growing
  sequence each step. Correct by construction but O(n²). *Addressed in part 2.*
* Single (i, j) pair, token-aligned pairs only, and the adapter is fit on control
  rows only. *Still open.*

## Stitching, part 2 (`stitch_fast`) — the early-exit path

Part 1 proved the injection point. It bought that proof by running *more*
compute than the large model alone. Part 2 keeps the same map and the same
injection index and runs only the layers the stitched path actually needs:

```
prompt -> small embed + small blocks 0..i-1 -> adapter(W, b)
       -> large blocks j..end -> large norm -> large lm_head -> token
```

Large blocks `0..j-1` never run and small blocks `i..end` never run. Both models
keep a KV cache over the blocks they do run, so decoding is O(n) rather than
part 1's O(n²). For the llama defaults that leaves 12 of 16 small blocks and 10
of 28 large blocks on the critical path; for qwen, 24 of 24 and 1 of 36.

Since HF's `forward` always runs the full stack, `Stack.run` reimplements the
layer loop over a slice (a transcription of `LlamaModel.forward` in transformers
5.x; Qwen2 is structurally identical). That transcription is verified, not
assumed — see the checks below.

### Two prefill strategies, one decode step

| mode | prompt goes through | first generated token | saves |
|---|---|---|---|
| `exit` | the stitch (large blocks `0..j-1` run only on the sink prefix) | the stitch's | prefill **and** decode |
| `warm` | the large model's own full stack | exactly the large model's | decode only |

`warm` exists because prefill happens once per prompt and decode happens once
per token, so most of the latency in a long generation is decode. It gives up
the prefill saving to hand the suffix blocks the large model's own KV — and
**its prefill runs the entire large model, so it saves no prefill FLOPs.** Both
are reported side by side by default; neither is a fallback for the other.

### The attention sink, in this setting

Part 1 could preserve the large model's own position-0 residual (`PRESERVE_PREFIX`)
for free, because it ran the large model's early blocks anyway. Here those
blocks are exactly what we are skipping, so the constant needed a new
implementation.

The fix is cheap and exact. Attention is causal, so running *only* tokens
`0..N-1` through large blocks `0..j-1` produces the same layer-j hidden states
for those positions as running the whole prompt would. The fast path therefore
does one N-token prefill through the skipped blocks — **N = 1 token against a
50–80 token prompt** — and splices the result over the adapter's output there.
The cost is O(N) once per prompt, not per token. In `warm` mode the question
does not arise: position 0's KV comes from the large model's own forward pass.

### Checks (all pass on both pairs; `results/<pair>/stitch/fast_checks_*.json`)

| check | what would break without it | llama | qwen |
|---|---|---|---|
| **full stack** (small, large) | the hand-rolled layer loop silently differs from HF's `forward` — everything downstream is then meaningless | rel-L2 `3.9e-05` / `2.4e-05` | `2.9e-05` / `1.3e-05` |
| **baseline == `generate`** (small, large) | the baselines the speedup is measured against are a weaker decoder than HF's | identical text | identical text |
| **prefix exact** | the sink splice isn't the large model's own state | bit-identical to HF's own 1-token forward | bit-identical |
| **fast == v1 prefill** | the early exit computes something *other* than what part 1 verified | rel-L2 `1.8e-02` | `3.7e-03` |
| **cached == lockstep** | the KV cache drifts from a from-scratch recompute | `2.8e-02` vs a `4.1e-02` floor | `5.4e-02` vs a `4.6e-02` floor |
| **warm prefill == large** | `warm` mode isn't handing back the large model's own logits | bit-identical (`0.0`) | bit-identical |

Two of these needed care to state honestly:

* **bf16 is not bit-reproducible across batch shapes**, so "prefix exact" cannot
  be tested as equality against the full-sequence run: a `(1, 1, d)` GEMM tiles
  differently from a `(1, seq, d)` one and the two disagree by rel-L2 `1.2e-04`
  (llama) to `8.4e-03` (qwen, at layer 35). That drift belongs to the library,
  not to this module — **HF's own forward reproduces it exactly** — so the pass
  criterion is bit-identity against HF's own prefix-only forward, with the
  library's drift recorded next to ours.
* **The cache tolerance is measured, not chosen.** An *unmodified* Llama-3B
  doing KV-cached decode drifts rel-L2 `3.9e-02` from a full recompute on this
  hardware, for the same GEMM-shape reason. So the check measures that floor for
  the unmodified large model in the same run and requires the stitch to stay
  within 3× of it. The comparison is also teacher-forced: left free-running, one
  rounding difference picks a different token and the whole divergent
  continuation gets charged to the cache.

### A convention bug this surfaced

`hidden_states[n_layers]` is the one entry HF overwrites with `model.norm(...)`
— the same fact `DROP_POST_NORM_LAYER` encodes on the analysis side. The qwen
diagnosis picks `i = 24` of 24 small layers, so the adapter was fit on the small
model's **post-norm** state. Running blocks `0..i-1` and stopping feeds the
adapter the raw residual stream instead, which is rel-L2 `0.79` away from what
it was fit on. Caught by the equivalence check (max logit gap `15.6`, which
argmax agreement alone would have waved through — hence the relative-distance
criterion in `_agree`). `StitchRunner.small_state` applies the final norm when
`i == n_layers_small`.

### Latency (Apple M5, MPS, bf16, batch 1, 12 prompts, greedy)

`ms/token` is the headline: prefill and decode are different regimes, and total
time is not comparable across paths that stop at different lengths.

**llama — 1B L12 → 3B L18** (skipping 18 of 28 large blocks)

| path | prefill ms | ms/token | tok/s | params/token | decode vs 3B |
|---|---:|---:|---:|---:|---:|
| 1B alone | 44.7 | 24.18 | 41.4 | 1.24 B (38%) | 2.43× |
| 3B alone | 114.0 | 58.75 | 17.0 | 3.21 B (100%) | 1.00× |
| stitch-exit | 106.9 | **41.44** | 24.1 | 2.14 B (67%) | **1.42×** |
| stitch-warm | 141.9 | **41.12** | 24.3 | 2.14 B (67%) | **1.43×** |

**qwen — 0.5B L24 → 3B L35** (skipping 35 of 36 large blocks)

| path | prefill ms | ms/token | tok/s | params/token | decode vs 3B |
|---|---:|---:|---:|---:|---:|
| 0.5B alone | 17.4 | 12.42 | 80.5 | 0.49 B (16%) | 4.56× |
| 3B alone | 83.7 | 56.65 | 17.7 | 3.09 B (100%) | 1.00× |
| stitch-exit | 70.1 | **17.38** | 57.5 | 0.75 B (24%) | **3.26×** |
| stitch-warm | 98.3 | **17.58** | 56.9 | 0.75 B (24%) | **3.22×** |

**The latency goal is met.** Decode is 1.42× the large model on llama and 3.26×
on qwen, and the wall-clock ratios track the weights actually multiplied per
token (67% and 24% of the large model's) — this is a real early exit, not a
re-labelled full forward. `params/token` is reported alongside every timing for
exactly that reason. The qwen speedup is much larger only because its diagnosed
`j = 35` skips all but the final large block, which is also why its accuracy is
the worst of the two.

### Accuracy — the tradeoff, stated plainly

Gold-answer match on 6 divergent + 6 control prompts, re-scored at run time:

| pair | set | small | large | stitch-exit | stitch-warm |
|---|---|---:|---:|---:|---:|
| llama | control | 6/6 | 6/6 | **1/6** | **1/6** |
| llama | divergent | 2/6 | 6/6 | **1/6** | **1/6** |
| qwen | control | 6/6 | 6/6 | **1/6** | **1/6** |
| qwen | divergent | 0/6 | 6/6 | **0/6** | **0/6** |

**The accuracy goal is not met, and the gap is not small.** The success
criterion set for this work was "better than the small model alone"; the
stitched path is **much worse than the small model alone** on both pairs
(llama 2/12 vs 8/12, qwen 1/12 vs 6/12). Paying 1.4–1.7× the small model's
decode cost to lose most of its accuracy is not a good trade at the current
adapter quality. Reported here rather than buried because it is the actual
result.

What the numbers do show is *where* it breaks. First-token agreement with the
large model is high — 6/6 control for llama `exit`, and 12/12 overall for `warm`
by construction — and then the answer falls apart at roughly the second token:

```
Ottawa    -> 'Ott'  + 'foundland'
Ankara    -> 'An'   + 'stanbul'
Brasília  -> 'Bras' + ' Salvador'
Thimphu   -> 'Th'   + '0ThThe'
```

The stitch reliably reproduces the large model's *first* decision and then loses
the thread, which is the signature of a lossy per-token reconstruction rather
than of a broken decode path — and the decode path is independently verified
above. `warm` mode buys a guaranteed-correct first token and, on these prompts,
nothing after it, which is a fairly precise measurement of how much of the
answer lives in one linear map of one layer.

The cause is the adapter, and the adapter's own sidecar predicts this: held-out
quality is good on *all* token positions (llama cosine `0.972`, R² `0.910`) and
poor on exactly the positions that matter, the **answer** tokens — llama R²
`0.343` on 30 rows, qwen R² **`−0.037`** on 22 rows. A map that fails to beat
predicting the mean on answer tokens cannot carry an answer, so the qwen result
(0/6 divergent) is what its own fit report implies. The three open v1 limits
above — norm-matching instead of ridge shrinkage, `LAST_K` truncation, and
fitting on control rows only — all bear directly on this, and none of them are
plumbing.

### Honest scope of part 2

* **What is established:** the early exit computes what part 1 computed
  (verified eight ways), skips the layers it claims to skip (verified by
  parameter accounting *and* wall clock), and is faster than the large model.
* **What is not:** that stitching is a useful accuracy/latency trade at this
  adapter quality. It is not, on either pair.
* Latency is one machine (M5 / MPS / bf16 / batch 1) and 12 prompts. The
  `params/token` column is the hardware-independent claim; the milliseconds are
  not. Batch-1 decode is bandwidth-bound, so on a GPU with different
  compute/bandwidth balance the ratios will move.
* Both baselines are timed through the *same* hand-rolled loop as the stitch, so
  the comparison is not measuring two different decoding harnesses — and the
  loop is not a weaker `generate`: it reproduces HF `generate` token-for-token
  on **48/48** runs (12 benchmark prompts × 2 models × 2 pairs), including the
  `<|eot_id|>`-vs-`</s>` stop condition. The `baseline_matches_hf_generate`
  check re-tests this on every run.
* The `divergent` / `control` buckets come from `selection.json`, which was
  produced on a different library build: the llama 1B now answers 2 of its 6
  recorded "divergent" prompts correctly. They are prompt sets here, not
  re-derived labels, and the benchmark prints a warning when it detects the
  drift.

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
  `q1/prompts.py`: at the current 87% small-model accuracy, reaching 50+
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
