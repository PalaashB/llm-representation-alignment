"""Confidence intervals for accuracy, and the comparison rule that uses them.

Every accuracy number in both stitching packages is a proportion measured on a
few dozen to a few hundred prompts, and the grids they appear in are read by
eye for "which cell is best". That reading is only meaningful if the interval
around each cell is smaller than the differences being compared, which on a
35-prompt split it emphatically is not: one prompt is 2.9 points and the
standard error near 70% is about 8 points, so neighbouring cells in the
published tables differ by less than their own noise.

`wilson` rather than the textbook normal interval. The normal approximation
p +- z*sqrt(p(1-p)/n) is wrong in exactly the regime these banks live in: it
gives a zero-width interval at p=0 or p=1 (several grid cells collapse to 0%,
and the large model hits 100% on easy banks) and it can extend past [0, 1].
The Wilson score interval has neither pathology and is accurate at small n,
which is the whole reason it is here.

`bootstrap_diff` covers the paired case. Two paths scored on the *same* prompt
set are correlated — they get the same easy items right — so comparing two
independent Wilson intervals is conservative to the point of hiding real
effects. Resampling prompts and recomputing both accuracies keeps the pairing
and gives a direct interval on the difference.
"""

from __future__ import annotations

import math

import numpy as np

Z95 = 1.959963984540054      # two-sided normal quantile at 95%
DEFAULT_BOOTSTRAP = 10000


def wilson(n_correct: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion, as (lo, hi).

    Degenerate n=0 returns the whole unit interval rather than raising: an
    empty subset (a divergent set with no members, say) is a legitimate state
    for a caller to be in and "we know nothing" is the honest interval.
    """
    if n <= 0:
        return (0.0, 1.0)
    p = n_correct / n
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def wilson_from_rate(accuracy: float, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson interval when only the rate survived into a saved report.

    Rounds back to a whole number of correct answers, which is exact whenever
    the rate really did come from n prompts, and is what lets the backfill path
    put intervals on benches written before intervals existed.
    """
    if n <= 0 or accuracy != accuracy:      # NaN
        return (0.0, 1.0)
    return wilson(int(round(accuracy * n)), n, z)


def ci_width_pts(lo: float, hi: float) -> float:
    return (hi - lo) * 100


def intervals_overlap(a: tuple[float, float], b: tuple[float, float]) -> bool:
    """True when two intervals share any point, i.e. the difference is not
    resolved at this sample size."""
    return a[0] <= b[1] and b[0] <= a[1]


def separated(a_correct: int, a_n: int, b_correct: int, b_n: int,
              z: float = Z95) -> bool:
    """True when a's interval lies entirely above b's.

    Deliberately one-directional and deliberately strict: it answers "may I
    claim a beats b", and non-overlap of two Wilson intervals is a stronger
    requirement than a two-sample test at the same level. Both packages gate
    their recommendations on it, so erring toward silence is the right bias —
    a false "no viable point" costs a rerun, a false recommendation goes in a
    write-up.
    """
    a_lo, a_hi = wilson(a_correct, a_n, z)
    b_lo, b_hi = wilson(b_correct, b_n, z)
    return a_lo > b_hi


def bootstrap_diff(a_correct: list[bool] | np.ndarray, b_correct: list[bool] | np.ndarray,
                   n_boot: int = DEFAULT_BOOTSTRAP, seed: int = 0,
                   alpha: float = 0.05) -> dict:
    """Paired bootstrap over prompts for accuracy(a) - accuracy(b).

    Both arrays are per-prompt correctness on the *same* prompts in the same
    order; resampling prompt indices (not the two arrays independently) is what
    preserves the pairing. Returns the observed difference, its interval, and
    whether that interval excludes zero.
    """
    a = np.asarray(a_correct, dtype=float)
    b = np.asarray(b_correct, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"paired bootstrap needs equal lengths, got {a.shape} and {b.shape}")
    n = len(a)
    if n == 0:
        return {"diff": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "excludes_zero": False, "n": 0, "n_boot": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    diffs = a[idx].mean(1) - b[idx].mean(1)
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    return {"diff": float(a.mean() - b.mean()), "lo": float(lo), "hi": float(hi),
            "excludes_zero": bool(lo > 0 or hi < 0), "n": n, "n_boot": n_boot}


def fmt_pct_ci(accuracy: float, ci: tuple[float, float]) -> str:
    """`85.7% [74.6-92.7]` — the form every table in both packages prints."""
    if accuracy != accuracy:
        return "   nan"
    return f"{accuracy * 100:.1f}% [{ci[0] * 100:.1f}-{ci[1] * 100:.1f}]"


def min_resolvable_pts(n: int, p: float = 0.85, z: float = Z95) -> float:
    """Width in points of the Wilson interval at (n, p).

    Used by the bank-size gates to say concretely what a split can and cannot
    resolve, so "150 prompts per split" is a measured requirement rather than a
    round number someone liked.
    """
    lo, hi = wilson(int(round(p * n)), n, z)
    return ci_width_pts(lo, hi)
