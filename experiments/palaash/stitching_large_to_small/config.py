"""Configuration for the accuracy-oriented stitching experiment (large -> small).

The stitched path is the mirror of the sibling package's:

    prompt -> large embed + large blocks 0..j-1 -> adapter -> small blocks
    i..end -> small norm -> small lm_head -> token

The large model does the reading, the small model does the writing. The question
is whether the large model's mid-stack representation carries enough of the
*answer* that the small model's late blocks can decode it — i.e. whether a small
model can be made more accurate without fine-tuning either LLM, by paying for a
partial large-model forward instead.

This is not a latency play and must not be read as one: the path runs j large
blocks *plus* (n_small - i) small blocks, so it is slower than the small model
alone and usually slower than the large model alone. Latency is reported anyway,
because a cost that is never printed is a cost that gets forgotten.

Torch-free on purpose so the numpy-only steps can import it cheaply.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from common.prompts import PROMPTS
from common.prompts_hard_factual import HARD_FACTUAL_PROMPTS
from common.prompts_list import LIST_PROMPTS, LIST_SYSTEM_PROMPT
from common.prompts_list_hard import LIST_HARD_PROMPTS, LIST_HARD_SYSTEM_PROMPT

HERE = Path(__file__).resolve().parent


# ── model pairs ───────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Pair:
    name: str
    small_id: str
    large_id: str
    small_tag: str
    large_tag: str
    dim_small: int
    n_layers_small: int
    dim_large: int
    n_layers_large: int
    grid_j: tuple[int, ...]   # large-model exit layers (source of the residual)
    grid_i: tuple[int, ...]   # small-model continue-from layers (injection point)

    @property
    def depth_ratio(self) -> float:
        return self.n_layers_large / self.n_layers_small

    def depth_matched_i(self, j: int) -> int:
        """The small layer at the same *relative depth* as large layer j.

        The neutral reference point: exiting the large model 60% of the way up
        and resuming the small model 60% of the way up preserves how much
        processing the residual has had, which is the least presumptuous thing
        to do when the two stacks have different depths.
        """
        return max(1, min(self.n_layers_small - 1, round(j / self.depth_ratio)))

    def depth_matched_j(self, i: int) -> int:
        return max(1, min(self.n_layers_large - 1, round(i * self.depth_ratio)))


PAIRS = {
    # Token-aligned pairs only: the adapter is fit position-by-position, which
    # needs both models to see the identical token sequence. Cross-family is out
    # of scope for v1.
    "llama": Pair(
        name="llama",
        small_id="meta-llama/Llama-3.2-1B-Instruct",
        large_id="meta-llama/Llama-3.2-3B-Instruct",
        small_tag="1B", large_tag="3B",
        dim_small=2048, n_layers_small=16,
        dim_large=3072, n_layers_large=28,
        # Deep enough into the large model that the answer has been retrieved,
        # and far enough up the small model that its remaining blocks are doing
        # surface realisation rather than recall.
        grid_j=(14, 18, 21, 24),
        grid_i=(8, 10, 12, 14),
    ),
    "qwen": Pair(
        name="qwen",
        small_id="Qwen/Qwen2.5-0.5B-Instruct",
        large_id="Qwen/Qwen2.5-3B-Instruct",
        small_tag="0.5B", large_tag="3B",
        dim_small=896, n_layers_small=24,
        dim_large=2048, n_layers_large=36,
        grid_j=(18, 24, 28, 32),
        grid_i=(12, 16, 18, 20),
    ),
}
DEFAULT_PAIR = "llama"


# ── prompt banks ──────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Bank:
    name: str
    prompts: tuple[dict, ...]
    system: str | None            # None -> common.model_utils.SYSTEM_PROMPT
    max_new_tokens: int
    teacher_answer_tokens: int
    latency_steps: int
    note: str = ""

    def __len__(self) -> int:
        return len(self.prompts)


BANKS = {
    "hard_factual": Bank(
        name="hard_factual", prompts=tuple(HARD_FACTUAL_PROMPTS), system=None,
        max_new_tokens=24, teacher_answer_tokens=16, latency_steps=32,
        note="348 obscure single-fact questions; built for small-wrong/large-right headroom",
    ),
    "factual": Bank(
        name="factual", prompts=tuple(PROMPTS), system=None,
        max_new_tokens=24, teacher_answer_tokens=16, latency_steps=32,
        note="the original short-answer bank; too easy for the small model to show much",
    ),
    "list": Bank(
        name="list", prompts=tuple(LIST_PROMPTS), system=LIST_SYSTEM_PROMPT,
        max_new_tokens=64, teacher_answer_tokens=48, latency_steps=64,
        note="conjunctive multi-item answers; large headroom but mixes recall with list-keeping",
    ),
    "list_hard": Bank(
        name="list_hard", prompts=tuple(LIST_HARD_PROMPTS),
        system=LIST_HARD_SYSTEM_PROMPT,
        max_new_tokens=48, teacher_answer_tokens=32, latency_steps=48,
        note="604 conjunctive 3-fact questions composed from this bank's own "
             "audited facts; dev=172 / test=161 for CI-resolvable comparisons",
    ),
}
DEFAULT_BANK = "hard_factual"
# ^ `hard_factual` is the bank this experiment is designed around. An accuracy
#   claim needs prompts the small model actually gets wrong and the large model
#   gets right; `factual` has 3 such prompts on a 35-prompt split, which cannot
#   support any conclusion. Run `headroom` on a new bank before trusting a sweep.

MIN_HEADROOM_PTS = 8.0      # below this, a bank cannot support an accuracy claim
MIN_DIVERGENT_PROMPTS = 20  # small-wrong/large-right cases needed on the eval split
MAX_CI_WIDTH_PTS = 15.0
# ^ A bank must also be big enough to *resolve* the gap it offers. `factual`
#   clears the headroom bar on paper (8.6 pts) while its 35-prompt split carries
#   a ~23-point Wilson interval, so no result on it could ever have been
#   significant. `hard_factual` at n=106 gives ~13.6 pts; `list_hard` at n=172
#   gives ~10.7.


# ── prompt splits ─────────────────────────────────────────────────────────────
SEED = 0
SPLIT_FRACS = {"fit": 0.50, "dev": 0.25, "test": 0.25}
# fit  — prompts the adapter is fit on (states captured here)
# dev  — prompts the (i, j) sweep selects on
# test — scored once, at the end

# ── adapter fit ───────────────────────────────────────────────────────────────
TRAIN_METHODS = ("ridge", "distill")
TRAIN_METHOD = "ridge"
LEGACY_TRAIN_METHOD = "legacy"
# ^ How the map's parameters are chosen.
#
#   `ridge`   closed-form weighted least squares in residual space: minimise L2
#             to the *small* model's own layer-i residual.
#   `distill` gradient descent on the *large* model's next-token distribution,
#             backpropagated through the frozen small-model suffix into the
#             adapter.
#
#   Ridge optimises the wrong thing here, and in this direction it is not a
#   subtle mismatch — it is self-defeating. The regression target is the state
#   the small model would have produced by itself, and on exactly the prompts
#   this experiment exists to fix, that state is the one that decodes to the
#   *wrong* answer. A map that hits its target perfectly reproduces the small
#   model's error. The sweep says so: warm beats exit in every cell, accuracy
#   falls as more of the large model's residual is injected, and the stitched
#   generations copy the small model's mistakes (plutonium 92, not 94).
#
#   `distill` targets the distribution the path is scored on, so the thing being
#   optimised is the thing being measured. It is warm-started from the ridge
#   solution, so it begins at exactly the ridge map's behaviour and ships ridge
#   unchanged if no epoch improves on it.
#
#   `legacy` is not a fit method: it is the read-only name of the pre-2026-08-23
#   artefacts, which were fit by ridge on a capture with 753 answer rows and no
#   template variety. Those files keep their unscoped filenames so the negative
#   result stays on disk exactly as it was published.

RIDGE_ALPHA = 0.01
ANSWER_WEIGHT = 4.0      # answer positions are the ones the adapter sees while decoding
NORM_MATCH = True
ADAPTER_TEST_FRAC = 0.25

# ── row selection for the fit ─────────────────────────────────────────────────
MIN_ANSWER_WEIGHT_FRAC = 0.5
# ^ Answer rows must carry more than half the fit objective's total weight, and
#   `adapter.fit` refuses to ship a map that misses this. The original capture
#   was nowhere near: 753 answer rows against 15555 prompt rows at
#   ANSWER_WEIGHT=4 is 753*4 / (753*4 + 15555) = 16%. Every prompt shared one
#   chat template and one system prompt, so the other 84% of the objective was
#   the same boilerplate repeated 211 times — which the map memorises (hence
#   held-out R2_all = 0.9999) while the positions it actually faces at decode
#   time are a rounding error in the loss.
PROMPT_ROW_KEEP = "auto"
# ^ How many prompt-position rows to keep. "auto" subsamples them (deterministic,
#   seeded) to whatever count makes answer rows clear MIN_ANSWER_WEIGHT_FRAC;
#   "all" keeps every row (the old behaviour); an integer keeps that many.
#   Prompt rows are thinned, never dropped: `exit` mode hands the adapter prompt
#   positions at inference, and they anchor the standardiser.

# ── capture composition ───────────────────────────────────────────────────────
CORPUS_ANSWER_TOKENS = 48
# ^ Generation budget for `--fit-corpus` items, separate from the bank's.
#   `hard_factual` answers are a name or a number and its teacher budget is 16
#   tokens, which is right for the bank and useless as a source of answer rows
#   in bulk: 211 fit prompts yield ~750. The corpus items are open-ended
#   instructions whose continuations run 40-80 tokens, so they only earn their
#   capture time at a budget long enough to let them finish a thought.

# ── distillation ──────────────────────────────────────────────────────────────
DISTILL_EPOCHS = 4
DISTILL_BATCH_SEQS = 4      # sequences per optimiser step (whole sequences: the
                            # suffix blocks attend, so rows are not independent)
DISTILL_LR = 2e-6
# ^ Small because the map is warm-started, not initialised. Adam's update is
#   ~lr per parameter per step regardless of gradient scale, so a large lr
#   discards the warm start within one epoch (measured in the sibling package:
#   val loss 0.2296 -> 0.4779 at lr=1e-4, never recovered). At 2e-6 an epoch
#   moves the weights a few percent of their magnitude, which is the regime
#   where "can distillation improve on ridge" is the question being asked.
DISTILL_WEIGHT_DECAY = 0.0
DISTILL_TOPK = 128
# ^ Teacher distributions are stored as top-K logits per position. Full vocab is
#   128256 floats per row — 80k rows is 20 GB at fp16 — while the top 128 tokens
#   hold essentially all the mass of a distribution the teacher would greedily
#   decode from. KL is computed over that support, renormalised.
DISTILL_CE_WEIGHT = 0.1     # small CE term on the teacher's argmax, alongside KL
DISTILL_TEMPERATURE = 1.0
DISTILL_VAL_FRAC = 0.2      # prompts held out to choose the epoch
DISTILL_MAX_GRAD_NORM = 1.0
DISTILL_TRAIN_MODES = ("warm", "exit")
DISTILL_TRAIN_MODE = "warm"
# ^ Which inference geometry the training forward reproduces. A map trained
#   under one injection pattern and run under another is being asked a different
#   question at inference than it was fit on, so this must match the mode the
#   bench reports. `warm` is the default because it is the mode the measured
#   best cell used.

# ── decoding ──────────────────────────────────────────────────────────────────
LATENCY_PROMPTS = 4
LATENCY_REPEATS = 2
LATENCY_SPREAD_FLAG = 1.5

PRESERVE_PREFIX = 1
# ^ Position 0 is the attention sink on the *small* stream here, since that is
#   the stream being overwritten. Its residual norm dwarfs every other position
#   and the adapter cannot reproduce it, so the fast path re-derives it by
#   running the skipped small blocks over that one token — exact, because
#   attention is causal. `check` verifies the splice.

# ── paths ─────────────────────────────────────────────────────────────────────
# Everything is scoped by (pair, bank) so runs never overwrite each other.
RESULTS_ROOT = HERE / "results"


def results_dir(pair: Pair, bank: Bank) -> Path:
    return RESULTS_ROOT / pair.name / bank.name


def states_dir(pair: Pair, bank: Bank) -> Path:
    """Where `capture` writes. One capture per (pair, bank), overwritten in place.

    The pre-2026-08-23 capture — bank prompts only, one chat template, 753
    answer rows — was moved aside to `states_legacy/` rather than deleted, so
    the `legacy` adapters can still be re-scored against the rows they were
    actually fit on.
    """
    return results_dir(pair, bank) / "states"


def adapters_dir(pair: Pair, bank: Bank) -> Path:
    return results_dir(pair, bank) / "adapters"


def variant_tag(train_method: str | None) -> str:
    """Filename suffix identifying how an adapter was fit.

    Empty for `legacy` (and for `None`, which means the same thing), so the
    artefacts of the published all-ridge failure keep the exact filenames they
    were written under and no new run can overwrite one. Everything fit after
    the capture was rebuilt carries its method in the name — including ridge,
    because "ridge on the old 753-answer-row capture" and "ridge on the new one"
    are different maps and the whole exercise is comparing them.
    """
    if train_method in (None, LEGACY_TRAIN_METHOD):
        return ""
    if train_method not in TRAIN_METHODS:
        raise SystemExit(f"--train-method must be one of "
                         f"{TRAIN_METHODS + (LEGACY_TRAIN_METHOD,)}, got {train_method!r}")
    return f"_{train_method}"


def adapter_path(pair: Pair, i: int, j: int, bank: Bank,
                 train_method: str | None = None) -> Path:
    return (adapters_dir(pair, bank)
            / f"adapter_i{i:02d}_j{j:02d}{variant_tag(train_method)}.npz")


def checks_path(pair: Pair, bank: Bank, i: int, j: int,
                train_method: str | None = None) -> Path:
    return (results_dir(pair, bank) / "checks"
            / f"checks_i{i:02d}_j{j:02d}{variant_tag(train_method)}.json")


def bench_path(pair: Pair, i: int, j: int, split: str, mode: str, bank: Bank,
               train_method: str | None = None) -> Path:
    return (results_dir(pair, bank) / "benches"
            / f"bench_i{i:02d}_j{j:02d}_{split}_{mode}{variant_tag(train_method)}.json")


def sweep_path(pair: Pair, bank: Bank, split: str, mode: str,
               train_method: str | None = None) -> Path:
    return (results_dir(pair, bank) / "sweeps"
            / f"sweep_{split}_{mode}{variant_tag(train_method)}.json")


def table_path(pair: Pair, bank: Bank, split: str, mode: str,
               train_method: str | None = None) -> Path:
    return (results_dir(pair, bank) / "tables"
            / f"sweep_{split}_{mode}{variant_tag(train_method)}.csv")


def capture_layers(pair: Pair) -> tuple[list[int], list[int]]:
    """Which layers capture stores. The roles are reversed relative to the
    sibling package: large layers are the adapter's *input*, small layers its
    *target*, so both grids plus their depth-matched partners are kept."""
    large = set(pair.grid_j) | {pair.depth_matched_j(i) for i in pair.grid_i}
    small = set(pair.grid_i) | {pair.depth_matched_i(j) for j in pair.grid_j}
    return (sorted(L for L in small if 1 <= L < pair.n_layers_small),
            sorted(L for L in large if 1 <= L < pair.n_layers_large))


def validate_layers(pair: Pair, i: int, j: int) -> None:
    if not 1 <= i < pair.n_layers_small:
        raise SystemExit(
            f"--i {i} out of range for {pair.small_tag}: use 1..{pair.n_layers_small - 1} "
            f"(0 is the embedding table, so injecting there discards the prompt's own "
            f"embedding; {pair.n_layers_small} is the post-norm state, not a residual "
            f"stream you can resume from)")
    if not 1 <= j < pair.n_layers_large:
        raise SystemExit(
            f"--j {j} out of range for {pair.large_tag}: use 1..{pair.n_layers_large - 1} "
            f"(0 is the embedding table; {pair.n_layers_large} is post-norm)")
