"""Central configuration: model pairs, hyperparameters, paths.

The experiment compares a *small* model against a *large* model. Same-family
pairs share a tokenizer, so hidden states align token-by-token
(align="token"). Cross-family pairs have different tokenizers, so they align
at the prompt level instead: one hidden-state row per prompt — the final
answer-generating token position — per model (align="prompt").
Each pair gets its own results folder: results/<pair-name>/.

Deliberately torch-free so the numpy-only steps (train_dm, analyze) can import
it without pulling in the ML stack.
"""

from dataclasses import dataclass
from pathlib import Path


# ── Model pairs ───────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ModelPair:
    name: str            # short key, also the results subfolder name
    small_id: str        # HF id of the small (hallucination-prone) model
    large_id: str        # HF id of the large (reference) model
    small_tag: str       # human-readable size labels for logs/plots
    large_tag: str
    # Known geometry (asserted at load time so a model swap fails loudly).
    dim_small: int
    n_layers_small: int  # transformer blocks; hidden-state tensors = n+1 (embeddings)
    dim_large: int
    n_layers_large: int
    # "token"  — same tokenizer, hidden states pair position-by-position
    # "prompt" — different tokenizers, one final-token row per prompt per model
    align: str = "token"


PAIRS = {
    "llama": ModelPair(
        name="llama",
        small_id="meta-llama/Llama-3.2-1B-Instruct",
        large_id="meta-llama/Llama-3.2-3B-Instruct",
        small_tag="1B", large_tag="3B",
        dim_small=2048, n_layers_small=16,    # 17 hidden-state tensors
        dim_large=3072, n_layers_large=28,    # 29 hidden-state tensors
    ),
    "qwen": ModelPair(
        name="qwen",
        small_id="Qwen/Qwen2.5-0.5B-Instruct",
        large_id="Qwen/Qwen2.5-3B-Instruct",
        small_tag="0.5B", large_tag="3B",
        dim_small=896, n_layers_small=24,     # 25 hidden-state tensors
        dim_large=2048, n_layers_large=36,    # 37 hidden-state tensors
    ),
    # Cross-family pairs: different tokenizers, so prompt-level alignment.
    "llama2qwen": ModelPair(
        name="llama2qwen",
        small_id="meta-llama/Llama-3.2-1B-Instruct",
        large_id="Qwen/Qwen2.5-3B-Instruct",
        small_tag="Llama-1B", large_tag="Qwen-3B",
        dim_small=2048, n_layers_small=16,    # 17 hidden-state tensors
        dim_large=2048, n_layers_large=36,    # 37 hidden-state tensors
        align="prompt",
    ),
    "qwen2llama": ModelPair(
        name="qwen2llama",
        small_id="Qwen/Qwen2.5-0.5B-Instruct",
        large_id="meta-llama/Llama-3.2-3B-Instruct",
        small_tag="Qwen-0.5B", large_tag="Llama-3B",
        dim_small=896, n_layers_small=24,     # 25 hidden-state tensors
        dim_large=3072, n_layers_large=28,    # 29 hidden-state tensors
        align="prompt",
    ),
}
DEFAULT_PAIR = "llama"


# ── Prompting ─────────────────────────────────────────────────────────────────
# SYSTEM_PROMPT and MAX_NEW_TOKENS live in common.model_utils, next to the
# functions that apply them, so both projects prompt identically.

# ── Hidden-state extraction ───────────────────────────────────────────────────
LAST_K = 64             # keep at most this many final prompt-token positions

# ── DM adapter training ───────────────────────────────────────────────────────
RIDGE_ALPHA = 0.1       # ridge strength, relative to per-column signal
TEST_FRAC = 0.30        # fraction of prompts held out
N_SPLITS = 6            # average the residual grid over this many prompt-splits
TRAIN_TOK_CAP = 1200    # cap fitted train tokens per split (bounds compute)
TEST_TOK_CAP = 800      # cap held-out non-final tokens per split (last tokens always kept)
SEED = 0

# ── Target-layer selection (which large layer j a small layer i is matched to) ─
# A bare argmax over all j is degenerate: early residual-stream layers are
# near-deterministic functions of token identity, which the small model retains
# at every depth, so "best over all j" collapses toward the shallow end (for the
# llama pair on r2_last, j=0 wins for 16 of 17 small layers; mean R² is 0.91 at
# j=0 vs a median of 0.67 across j>=1). We therefore restrict j to a band around
# i's *relative depth*, which is also the only regime where the chosen j is a
# plausible stitching target.
DEPTH_BAND = 3          # allow |j - i/n_small*n_large| <= this many large layers
MIN_TARGET_LAYER = 1    # never match into the large model's embedding layer
DROP_POST_NORM_LAYER = True
# ^ HF returns n_layers+1 hidden states, but the final one is the output of
#   model.norm, not a residual stream. Its scale is discontinuous with the rest
#   (llama-3B: RMS climbs smoothly 0.018 -> 0.525 over layers 0-27, then jumps
#   to 1.622 at layer 28), and it is not a valid injection point, so exclude it.

ONSET_THRESH = 0.95     # best-match R² below this counts as "no longer translatable"
ONSET_SCAN_START = 1
# ^ Start the onset scan at layer 1. Layer 0 is the small model's embedding
#   table, which has no sensible depth-matched counterpart (its band sits at
#   large layers 1-3, giving R²~0.86) and would otherwise trip the threshold
#   immediately and report onset=0 for every selection rule.

# ── CKA ───────────────────────────────────────────────────────────────────────
CKA_ROW_CAP = 1200      # cap rows used for the all-token CKA grid (Gram matrices are n x n)

# ── Materialising the DM map (fit_adapter) ────────────────────────────────────
# The DM grid is a *diagnostic*: train_dm never materialises W (it solves for R²
# through a hat matrix). fit_adapter re-solves the ridge normal equations
# explicitly for one (i, j) and saves W/b, so the selected map can be inspected.
ADAPTER_FIT_SET = "control"   # "control" | "all" | "divergent" — rows the saved map is fit on
ADAPTER_TEST_FRAC = 0.25      # prompt-level holdout, used only to *report* map quality

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent      # experiments/palaash/diagnosis
RESULTS_ROOT = PROJECT_ROOT / "results"


def results_dir(pair: ModelPair) -> Path:
    return RESULTS_ROOT / pair.name


def states_dir(pair: ModelPair) -> Path:
    return results_dir(pair) / "states"


def dm_dir(pair: ModelPair) -> Path:
    return results_dir(pair) / "dm"


def cka_dir(pair: ModelPair) -> Path:
    return results_dir(pair) / "cka"


def figures_dir(pair: ModelPair) -> Path:
    return results_dir(pair) / "figures"


def adapters_dir(pair: ModelPair) -> Path:
    return results_dir(pair) / "adapters"


def adapter_path(pair: ModelPair, i: int, j: int) -> Path:
    return adapters_dir(pair) / f"adapter_i{i:02d}_j{j:02d}.npz"
