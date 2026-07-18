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
SYSTEM_PROMPT = (
    "You are a precise factual assistant. Answer with ONLY the specific fact "
    "requested — a name, number, or short phrase. Do not add explanation."
)
MAX_NEW_TOKENS = 24     # greedy generation budget per answer

# ── Hidden-state extraction ───────────────────────────────────────────────────
LAST_K = 64             # keep at most this many final prompt-token positions

# ── DM adapter training ───────────────────────────────────────────────────────
RIDGE_ALPHA = 0.1       # ridge strength, relative to per-column signal
TEST_FRAC = 0.30        # fraction of prompts held out
N_SPLITS = 6            # average the residual grid over this many prompt-splits
TRAIN_TOK_CAP = 1200    # cap fitted train tokens per split (bounds compute)
TEST_TOK_CAP = 800      # cap held-out non-final tokens per split (last tokens always kept)
SEED = 0

# ── CKA ───────────────────────────────────────────────────────────────────────
CKA_ROW_CAP = 1200      # cap rows used for the all-token CKA grid (Gram matrices are n x n)

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = REPO_ROOT / "results"


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
