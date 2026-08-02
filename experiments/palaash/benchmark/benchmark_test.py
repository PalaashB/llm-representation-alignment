"""Small end-to-end test run of benchmark.py: one model, 10 repeats.

Same code path as the real thing — this script does not reimplement any of the
timing or CSV logic, it imports `benchmark` and calls its `main()` with
`--models llama-1b --repeats 10`. So if this passes, the full run works too;
if `benchmark.py` changes, this follows automatically.

Two things differ from a real run, both deliberate:

  * output goes to `results_test/`, not `results/`, so a test never lands in
    the table you actually report;
  * the three CSVs are deleted first, so a pass proves *this* run wrote them
    rather than finding leftovers from an earlier one (`--keep` opts out, which
    is how you exercise the upsert-into-an-existing-row path).

    python benchmark_test.py                       # ~10 loads + 10 generations
    python benchmark_test.py --device cpu --dtype fp32   # extra flags pass through
    python benchmark_test.py --keep --gpu 3060     # add a 2nd row to the test CSVs

Exits non-zero if the CSVs are missing or llama-1b's cell came out empty, so
this is usable as a pre-flight check before queueing the real 100-repeat job.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark                                   # noqa: E402


MODEL = "llama-1b"
REPEATS = 10
TEST_DIR = Path(__file__).resolve().parent / "results_test"


def redirect_outputs() -> None:
    """Point benchmark's four module-level paths at results_test/.

    `main()` and `append_raw()` read these at call time, so rebinding them here
    is enough — no argument threading needed, and nothing in benchmark.py has
    to know that a test mode exists.
    """
    benchmark.RESULTS_DIR = TEST_DIR
    benchmark.LOAD_CSV = TEST_DIR / "model_load_time_seconds.csv"
    benchmark.INFER_CSV = TEST_DIR / "model_inference_time_ms.csv"
    benchmark.RAW_CSV = TEST_DIR / "raw_timings.csv"


def check() -> int:
    """Verify the run produced the CSVs it should have. Returns an exit code."""
    problems: list[str] = []

    for path in (benchmark.LOAD_CSV, benchmark.INFER_CSV):
        if not path.exists():
            problems.append(f"{path.name} was not created")
            continue

        # Same dtype pin as upsert_cell: without it a "2080" row label reads
        # back as int64 and the row/label comparisons below stop matching.
        # benchmark.main() has already printed both frames in full, so this
        # only reports the cell the test is actually about.
        df = pd.read_csv(path, index_col="gpu", dtype={"gpu": str})
        cells = df[MODEL].dropna()
        print(f"{path.name:34s} {MODEL} = "
              f"{', '.join(f'{g}: {v:.4f}' for g, v in cells.items()) or '(empty)'}")

        if list(df.columns) != list(benchmark.MODELS):
            problems.append(f"{path.name}: columns are {list(df.columns)}, "
                            f"expected {list(benchmark.MODELS)}")
        if df[MODEL].notna().sum() == 0:
            problems.append(f"{path.name}: no {MODEL} value was written")
        # One row per GPU. Two rows for one card is the mixed str/int index bug.
        if df.index.duplicated().any():
            problems.append(f"{path.name}: duplicate gpu rows {list(df.index)}")

    if not benchmark.RAW_CSV.exists():
        problems.append("raw_timings.csv was not created")
    else:
        raw = pd.read_csv(benchmark.RAW_CSV, dtype={"gpu": str})
        counts = raw.groupby("phase").size().to_dict()
        print(f"\nraw_timings.csv  {len(raw)} rows  {counts}")
        for phase in ("load", "inference"):
            # 10 repeats in, 10 timing rows out — a short count means the loop
            # died partway and the mean was taken over fewer runs than asked.
            if counts.get(phase, 0) < REPEATS:
                problems.append(f"raw_timings.csv: {counts.get(phase, 0)} "
                                f"{phase} rows, expected >= {REPEATS}")
        if "error" in counts:
            problems.append(f"raw_timings.csv: {counts['error']} error row(s) — "
                            f"the model failed to load or generate")

    print()
    if problems:
        for p in problems:
            print(f"FAIL  {p}")
        return 1
    print(f"PASS  {MODEL} x{REPEATS} wrote all three CSVs -> {TEST_DIR}")
    return 0


def main() -> None:
    argv = sys.argv[1:]
    keep = "--keep" in argv
    argv = [a for a in argv if a != "--keep"]

    redirect_outputs()
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    if not keep:
        for path in (benchmark.LOAD_CSV, benchmark.INFER_CSV, benchmark.RAW_CSV):
            path.unlink(missing_ok=True)

    print(f"test run: {MODEL}, {REPEATS} repeats -> {TEST_DIR}\n")

    # Fixed flags first so a user-supplied --repeats/--models later on the
    # command line wins (argparse keeps the last occurrence).
    sys.argv = ["benchmark.py", "--models", MODEL, "--repeats", str(REPEATS)] + argv
    benchmark.main()

    sys.exit(check())


if __name__ == "__main__":
    main()
