"""Configuration for the latency-oriented stitching experiment.

Scope: make a small+large stitch that decodes measurably faster than the large
model while staying close to its accuracy. Nothing here is about diagnosis,
CKA, or hallucination localisation — the layer pair (i, j) is a
speed/accuracy knob, not a finding.

The stitched path is

    prompt -> small embed + small blocks 0..i-1 -> adapter -> large blocks
    j..end -> large norm -> large lm_head -> token

so the per-token cost is (i / n_small) of the small model plus
((n_large - j) / n_large) of the large one. Larger j is faster and less
accurate; the sweep is over that frontier.

Torch-free on purpose so the numpy-only steps can import it cheaply.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from common.prompts import PROMPTS
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
    grid_i: tuple[int, ...]        # small-model handoff layers to sweep
    grid_j: tuple[int, ...]        # large-model injection layers to sweep

    @property
    def depth_ratio(self) -> float:
        return self.n_layers_large / self.n_layers_small

    def depth_matched_j(self, i: int) -> int:
        """The large layer at the same *relative depth* as small layer i. Not a
        constraint — just the neutral reference point the sweep brackets."""
        return int(round(i * self.depth_ratio))


PAIRS = {
    # Both pairs are token-aligned (shared tokenizer within a family), which is
    # what lets the adapter be fit position-by-position. Cross-family pairs are
    # deliberately out of scope here.
    "llama": Pair(
        name="llama",
        small_id="meta-llama/Llama-3.2-1B-Instruct",
        large_id="meta-llama/Llama-3.2-3B-Instruct",
        small_tag="1B", large_tag="3B",
        dim_small=2048, n_layers_small=16,
        dim_large=3072, n_layers_large=28,
        grid_i=(6, 8, 10, 12, 14),
        grid_j=(10, 12, 14, 16, 18),
        # Small blocks are cheap here (~1.5 ms each against ~2.1 for a large
        # block), so `i` buys accuracy at little latency cost while `j` is where
        # the speedup actually comes from. The grid is weighted accordingly.
    ),
    "qwen": Pair(
        name="qwen",
        small_id="Qwen/Qwen2.5-0.5B-Instruct",
        large_id="Qwen/Qwen2.5-3B-Instruct",
        small_tag="0.5B", large_tag="3B",
        dim_small=896, n_layers_small=24,
        dim_large=2048, n_layers_large=36,
        grid_i=(6, 10, 14, 18),
        grid_j=(12, 16, 20, 24),
    ),
}
DEFAULT_PAIR = "llama"


# ── prompt banks ──────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Bank:
    """A prompt set plus the decode budgets it needs.

    Budgets belong to the bank rather than to the module because they are
    properties of the answers: a bank of 4-token answers and a bank of 40-token
    answers cannot share a generation cap, a teacher-forcing length, or a
    latency step count without one of them being measured wrong.
    """
    name: str
    prompts: tuple[dict, ...]
    system: str | None            # None -> common.model_utils.SYSTEM_PROMPT
    max_new_tokens: int           # accuracy runs: greedy, stop at EOS
    teacher_answer_tokens: int    # answer positions captured for the adapter fit
    latency_steps: int            # latency runs: forced step count, EOS ignored
    note: str = ""

    def __len__(self) -> int:
        return len(self.prompts)


BANKS = {
    "factual": Bank(
        name="factual", prompts=tuple(PROMPTS), system=None,
        max_new_tokens=24, teacher_answer_tokens=16, latency_steps=32,
        note="short factual recall; 3-5 token answers, ANY-alias scoring",
    ),
    "list": Bank(
        name="list", prompts=tuple(LIST_PROMPTS), system=LIST_SYSTEM_PROMPT,
        max_new_tokens=64, teacher_answer_tokens=48, latency_steps=64,
        note="multi-item answers; 20-60 token answers, ALL-items scoring",
    ),
    "list_hard": Bank(
        name="list_hard", prompts=tuple(LIST_HARD_PROMPTS),
        system=LIST_HARD_SYSTEM_PROMPT,
        max_new_tokens=48, teacher_answer_tokens=32, latency_steps=48,
        note="604 conjunctive 3-fact questions composed from the audited "
             "hard_factual bank; dev=172 / test=161, ALL-items scoring",
    ),
}
DEFAULT_BANK = "list_hard"
# ^ A latency claim needs decode to be most of the clock and needs splits large
#   enough to rank anything, and only `list_hard` has both.
#
#   `factual` fails on both counts: 3.5-token answers put 65% of end-to-end time
#   in prefill, which stitching does not help, and its 35-prompt splits give a
#   ~23-point confidence interval across an 8.6-point small-vs-large gap. The
#   whole 50-point grid was measured through that instrument.
#
#   `list` fixes the answer length (10.3 tokens) but not the splits: 45 prompts,
#   ~21-point interval. `list_hard` gives 172/161-prompt eval splits at ~11
#   points, and conjunction over audited facts widens the model gap instead of
#   narrowing it. Both older banks stay selectable, and their results stay on
#   disk under their own names.


# ── prompt splits ─────────────────────────────────────────────────────────────
SEED = 0
SPLIT_FRACS = {"fit": 0.50, "dev": 0.25, "test": 0.25}
# fit  — prompts the adapter is fit on (states captured here)
# dev  — prompts the (i, j) sweep selects on
# test — prompts touched only once, for the final honest number
# `factual` is 140 prompts and `list` is 135, so dev and test are ~34 each: one
# prompt is ~3 accuracy points and the standard error on a 70%-ish rate is ~8
# points. Small accuracy differences between neighbouring grid points are not
# meaningful; large ones (the small-vs-large gap, or a collapse to near zero) are.

# ── state capture ─────────────────────────────────────────────────────────────
# The single most important choice in this folder is that states are captured
# over prompt AND the large model's own greedy answer, teacher-forced through
# both models (`Bank.teacher_answer_tokens` sets the answer length). The earlier
# early-exit attempt (since removed) fit only on prompt positions and collapsed
# after the first generated token — held-out R² was 0.91 on all tokens but 0.34
# on answer tokens, and generations read "Wellington" -> "Well is",
# "Jefferson City" -> "Jeffapolis". Answer positions are the ones the adapter
# actually sees while decoding, so they have to be in the fit.
#
# `teacher_answer_tokens` is a cap, not a target: the teacher generation stops at
# EOS, so the realised count per prompt is the answer's own length. Post-EOS
# positions are deliberately not captured — decoding stops there too, so they are
# rows the adapter is never asked about at inference.
#
# That cap is still why the `list` bank helps the fit as much as the benchmark.
# Measured on llama/dev, `factual` answers run 3.5 tokens and `list` answers 10.3,
# so 90 fit prompts yield ~1.0k answer rows to constrain a 2048x3072 map where the
# old bank yielded ~270.

# ── adapter fit ───────────────────────────────────────────────────────────────
ADAPTER_KINDS = ("linear", "mlp")
ADAPTER_KIND = "linear"  # `mlp` adds a nonlinear correction on top; see adapter.py

TRAIN_METHODS = ("ridge", "distill")
TRAIN_METHOD = "ridge"
# ^ How the map's parameters are chosen. Orthogonal to ADAPTER_KIND, which is
#   the map's *shape*: either shape can be fit by either method.
#
#   `ridge`   closed-form weighted least squares in residual space. Minimises
#             L2 to the large model's layer-j hidden state.
#   `distill` gradient descent on the large model's next-token distribution,
#             backpropagated through the frozen suffix blocks into the adapter.
#
#   Ridge optimises the wrong thing, and the repo's own numbers say so. Next-token
#   identity survives only in the few directions that clear the final norm and
#   lm_head, but ridge weights all 3072 output dimensions equally, so it spends
#   its capacity on the directions the unembedding discards. The symptom is on
#   record: R2_all = 0.999 while R2_answer ranges -0.20..0.375, and answer-token
#   R2 does not order the grid cells by accuracy. A metric that does not rank
#   what you care about is a misaligned proxy, not a noisy one.
#
#   `distill` targets the distribution directly, so what it optimises is what is
#   scored. It is warm-started from the ridge solution, so it begins at the
#   baseline's exact behaviour and any movement is an improvement on it.

RIDGE_ALPHA = 0.01       # lighter than the diagnosis fit's 0.1: shrinkage
                         # is what flattens answers into generic continuations
ANSWER_WEIGHT = 4.0      # row weight for answer positions relative to prompt positions
NORM_MATCH = True        # undo ridge's radial shrinkage (see adapter.fit)
ADAPTER_TEST_FRAC = 0.25 # prompt-level holdout, reporting only

# ── row selection for the fit ─────────────────────────────────────────────────
MIN_ANSWER_WEIGHT_FRAC = 0.5
# ^ Answer rows must carry more than half the fit objective's total weight.
#   They did not: on the `factual` capture, 258 answer rows against 5180 prompt
#   rows at ANSWER_WEIGHT=4 gives 258*4 / (258*4 + 5180) = 17%. Every prompt
#   shares one chat template and one system prompt, so the other 83% is the same
#   boilerplate repeated 70 times, which the map can memorise (hence
#   R2_all = 0.999) while the positions that actually matter at decode time are
#   a rounding error in the loss.
PROMPT_ROW_KEEP = "auto"
# ^ How many prompt-position rows to keep. "auto" subsamples them (deterministic,
#   seeded) to whatever count makes answer rows clear MIN_ANSWER_WEIGHT_FRAC;
#   "all" keeps every row (the old behaviour); an integer keeps that many.
#   Prompt rows are not dropped entirely: the map still has to produce a sane
#   residual at prompt positions in `exit` mode, and they anchor the standardiser.

# ── distillation ──────────────────────────────────────────────────────────────
DISTILL_EPOCHS = 4
DISTILL_BATCH_SEQS = 4      # sequences per optimiser step (whole sequences: the
                            # suffix blocks attend, so rows are not independent)
DISTILL_LR = 2e-6
# ^ Small because the map is warm-started, not initialised. Adam's update is
#   ~lr per parameter per step regardless of gradient scale, so at lr=1e-4 over
#   ~180 steps/epoch the cumulative drift is ~0.018 against ridge weights whose
#   own scale is ~0.02 — i.e. the first epoch discards the warm start. Measured:
#   val loss 0.2296 -> 0.4779 after one epoch, and it never recovered. At 2e-6
#   an epoch moves the weights a few percent of their magnitude, which is the
#   regime where "can distillation improve on ridge" is actually being asked.
DISTILL_WEIGHT_DECAY = 0.0
DISTILL_TOPK = 128
# ^ Teacher distributions are stored as top-K logits per answer position. Full
#   vocab would be 128256 floats per row — 20k rows is 5 GB at fp16 — while the
#   top 128 tokens hold essentially all the mass of a greedy-decodable
#   distribution. KL is computed over that support, renormalised.
DISTILL_CE_WEIGHT = 0.1     # small CE term on the teacher's argmax, alongside KL
DISTILL_TEMPERATURE = 1.0
DISTILL_VAL_FRAC = 0.2      # prompts held out to choose the epoch, as for the MLP
DISTILL_MAX_GRAD_NORM = 1.0

N_TAPS = 1
# ^ How many small-model layers feed the adapter. 1 = layer i alone (the original
#   map). Extra taps are read from evenly spaced layers below i and cost nothing
#   at inference: blocks 0..i-1 run anyway, so an intermediate residual is
#   already in flight and only has to be kept. A single layer's residual is a
#   lossy summary — the residual stream overwrites as much as it accumulates —
#   so reading two depths gives the map strictly more to work with per token.
CAPTURE_MAX_TAPS = 3     # tap layers stored at capture time, so --n-taps up to
                         # this needs no re-capture

# MLP correction (only used when kind == "mlp").
MLP_HIDDEN = 1024        # ~5.2M extra weights on llama, i.e. 0.2% of the large
                         # model's per-token cost — invisible in the latency
MLP_EPOCHS = 80
MLP_BATCH = 512
MLP_LR = 1e-3
MLP_WEIGHT_DECAY = 1e-4
MLP_VAL_FRAC = 0.2
# ^ Prompts held out *inside* the training rows to pick the epoch to stop at.
#   Without it the correction memorises: on llama/list it drove weighted MSE from
#   0.83 to 0.10 in sample while held-out answer R² went 0.263 -> 0.250, i.e. it
#   spent 5M parameters getting slightly worse at the only thing that matters.
#   ~1.1k answer rows cannot constrain that many weights, so the stopping point
#   has to be chosen on data the fit cannot see.

# ── decoding ──────────────────────────────────────────────────────────────────
# Accuracy and latency are measured in separate passes. Accuracy stops at EOS;
# latency runs a fixed `bank.latency_steps` on every path, so each path is timed
# over identical work and ms/token does not rest on however many tokens that
# particular path happened to emit before stopping.
LATENCY_PROMPTS = 4      # prompts used for the timing loop (it is prompt-insensitive)
LATENCY_REPEATS = 2      # passes over those prompts, so samples are spread over time
# ^ Contamination on this machine arrives in windows lasting tens of seconds. One
#   pass over 4 prompts can fall entirely inside such a window while the baseline
#   measured seconds earlier did not, which yields a ratio that is pure artefact
#   (seen on qwen: identical stitch geometry timed 37 ms at one point in the
#   sweep and 109 ms at another). Spreading the samples makes a clean minimum
#   much more likely; LATENCY_SPREAD_FLAG below reports when it still failed.
LATENCY_SPREAD_FLAG = 1.5
# ^ max/min across timing samples above which a cell's latency is marked
#   `latency_suspect`. Decode cost here is genuinely prompt-independent, so a
#   wide spread means interference, not a property of the configuration.

PRESERVE_PREFIX = 1
# ^ Position 0 is the attention sink; its residual norm at large-model depth is
#   ~30x every other position and the adapter cannot reproduce it. The fast path
#   re-derives it by running the *skipped* large blocks over that one token —
#   exact, because attention is causal — and splices it in. Costs one 1-token
#   forward per prompt. Verified by `python -m stitching_small_to_large.run check`.

# ── sweep selection ───────────────────────────────────────────────────────────
MIN_DECODE_SPEEDUP = 1.25   # per-token floor, reported but no longer the selector
MIN_E2E_SPEEDUP = 1.25
# ^ The axis a recommendation is actually made on. Decode speedup is the
#   mechanism; end-to-end is the thing a user experiences, and the two came
#   apart badly: `warm` mode pays a full large prefill *plus* a small one, so
#   the best-accuracy cell on `factual` ran 1.11x per token and 0.93x
#   end-to-end — slower than the large model it was supposed to beat. Selecting
#   on decode had no way to see that.

# ── what makes a bank usable for a latency claim ──────────────────────────────
MIN_HEADROOM_PTS = 10.0
MIN_DECODE_SHARE = 0.75
MAX_CI_WIDTH_PTS = 15.0
# ^ `headroom` gates on all three, plus separated small/large intervals. Each
#   corresponds to a way the `factual` campaign was unmeasurable before anyone
#   fit an adapter:
#     headroom     8.6 pts of small-vs-large gap is not enough to spend layers on.
#     decode share only 65% of end-to-end was decode, so even a large per-token
#                  win could not move the wall clock much (and `warm` mode's
#                  extra prefill then ate the rest).
#     CI width     35 prompts gives a ~23-point interval, so the 8.6-point gap
#                  the bank did offer was not resolvable, let alone the
#                  differences between neighbouring grid cells.

PREFILL_ANOMALY_FACTOR = 1.5
# ^ A cell is flagged when its prefill exceeds this multiple of the *median*
#   measured/modelled prefill ratio among cells sharing its mode. Cross-cell,
#   because the within-cell decode-spread flag (LATENCY_SPREAD_FLAG) is blind to
#   sustained interference: at i=12, j=10 warm the prefill was 403.9 ms against
#   ~160 ms for its neighbours — 2.6x the layer-fraction model, while that same
#   cell's decode ratio was a perfectly normal 1.11 and its decode spread 1.11,
#   so nothing fired. Two extra small blocks cannot cost 240 ms of prefill; the
#   machine was slow for the duration of that cell.

# ── adapter variants ──────────────────────────────────────────────────────────
def tap_layers(i: int, n_taps: int = N_TAPS) -> tuple[int, ...]:
    """Small-model layers whose residuals feed the adapter, shallowest first.

    Always ends at i (the handoff depth). Extra taps sit at evenly spaced
    fractions of i, which is the arrangement that costs nothing: those blocks run
    on the way to i regardless. Duplicates collapse, so a shallow i can yield
    fewer than n_taps layers — the fit records how many it actually got.
    """
    if n_taps < 1:
        raise SystemExit(f"--n-taps must be >= 1, got {n_taps}")
    return tuple(sorted({max(1, round(i * k / n_taps)) for k in range(1, n_taps + 1)}))


def variant_tag(kind: str = ADAPTER_KIND, n_taps: int = N_TAPS,
                train_method: str = TRAIN_METHOD) -> str:
    """Filename suffix identifying an adapter variant.

    Empty for the original single-tap linear ridge map, so the artefacts from
    earlier runs keep their names and a new variant never silently overwrites a
    result it should be compared against. The training method is part of the
    scope for the same reason the map's shape is: `distill` vs `ridge` at fixed
    geometry is the comparison the whole exercise turns on, and it is only a
    comparison if both are on disk at once.
    """
    if kind not in ADAPTER_KINDS:
        raise SystemExit(f"--adapter must be one of {ADAPTER_KINDS}, got {kind!r}")
    if train_method not in TRAIN_METHODS:
        raise SystemExit(f"--train-method must be one of {TRAIN_METHODS}, "
                         f"got {train_method!r}")
    tag = ""
    if not (kind == "linear" and n_taps == 1):
        tag = "_" + kind + (f"t{n_taps}" if n_taps > 1 else "")
    if train_method != "ridge":
        tag += f"_{train_method}"
    return tag


# ── paths ─────────────────────────────────────────────────────────────────────
# Everything is scoped by (pair, bank, adapter variant) so results from different
# benchmarks or different adapters can coexist and be compared. The default bank
# maps to the unscoped directory the earlier runs wrote to.
RESULTS_ROOT = HERE / "results"


def results_dir(pair: Pair, bank: Bank | None = None) -> Path:
    d = RESULTS_ROOT / pair.name
    return d if bank is None or bank.name == "factual" else d / bank.name


def states_dir(pair: Pair, bank: Bank | None = None) -> Path:
    return results_dir(pair, bank) / "states"


def adapters_dir(pair: Pair, bank: Bank | None = None) -> Path:
    return results_dir(pair, bank) / "adapters"


def adapter_path(pair: Pair, i: int, j: int, bank: Bank | None = None,
                 kind: str = ADAPTER_KIND, n_taps: int = N_TAPS,
                 train_method: str = TRAIN_METHOD) -> Path:
    return (adapters_dir(pair, bank)
            / f"adapter_i{i:02d}_j{j:02d}{variant_tag(kind, n_taps, train_method)}.npz")


def bench_path(pair: Pair, i: int, j: int, split: str, mode: str = "exit",
               bank: Bank | None = None, kind: str = ADAPTER_KIND,
               n_taps: int = N_TAPS, train_method: str = TRAIN_METHOD) -> Path:
    return (results_dir(pair, bank) / "bench"
            / f"bench_i{i:02d}_j{j:02d}_{mode}_{split}"
              f"{variant_tag(kind, n_taps, train_method)}.json")


def sweep_stem(split: str, kind: str = ADAPTER_KIND, n_taps: int = N_TAPS,
               train_method: str = TRAIN_METHOD) -> str:
    return f"sweep_{split}{variant_tag(kind, n_taps, train_method)}"


def capture_layers(pair: Pair) -> tuple[list[int], list[int]]:
    """Which layers state capture has to store: the sweep grid, every tap layer
    the grid implies up to CAPTURE_MAX_TAPS, plus every depth-matched j, so a
    hand-picked (i, j) or a change of --n-taps usually needs no re-capture.

    Small-model states are cheap (2048-d at float16, a few MB per layer per
    prompt), so capturing the taps speculatively is a much better trade than
    re-running capture when a variant needs one.
    """
    small = set(pair.grid_i)
    for i in pair.grid_i:
        for k in range(1, CAPTURE_MAX_TAPS + 1):
            small |= set(tap_layers(i, k))
    small = sorted(L for L in small if 1 <= L < pair.n_layers_small)
    large = sorted(set(pair.grid_j) | {pair.depth_matched_j(i) for i in pair.grid_i})
    large = [j for j in large if 1 <= j < pair.n_layers_large]
    return small, large


def validate_layers(pair: Pair, i: int, j: int) -> None:
    if not 1 <= i < pair.n_layers_small:
        raise SystemExit(
            f"--i {i} out of range for {pair.small_tag}: use 1..{pair.n_layers_small - 1} "
            f"(0 is the embedding table; {pair.n_layers_small} is the post-norm state, "
            f"not a residual stream)")
    if not 1 <= j < pair.n_layers_large:
        raise SystemExit(
            f"--j {j} out of range for {pair.large_tag}: use 1..{pair.n_layers_large - 1} "
            f"(0 is the embedding table; {pair.n_layers_large} is post-norm)")
