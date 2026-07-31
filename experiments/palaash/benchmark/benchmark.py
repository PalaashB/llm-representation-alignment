"""Benchmark model load time and inference time for the four Q1 models.

Run the same script once per GPU (change the GPU with your SLURM
`--gres`/`--constraint`); each run appends/overwrites *one row* — its GPU — in
two CSVs whose columns are the four models:

    results/model_load_time_seconds.csv     mean seconds to load a model
    results/model_inference_time_ms.csv     mean milliseconds per generation

    gpu,llama-1b,llama-3b,qwen-0.5b,qwen-3b
    2080,...,...,...,...
    3060,...,...,...,...
    A100,...,...,...,...

Every individual timing is also appended to `results/raw_timings.csv` (one row
per repeat) so std-devs / outliers can be inspected later.

    python benchmark.py                       # 100 loads + 100 generations per model
    python benchmark.py --load-repeats 10     # loading 100x is slow, see README
    python benchmark.py --models qwen-0.5b    # one model only
    python benchmark.py --gpu A100            # override GPU auto-detection

What is measured
    load       tokenizer + weights + .eval(), from a warm Hugging Face cache,
               with the previous copy freed from VRAM first. A warm-up load
               happens before the timed loop so no timed load pays the
               download / cold page-cache cost.
    inference  one greedy `generate` of exactly MAX_NEW_TOKENS tokens on a
               fixed chat-formatted prompt (min_new_tokens is pinned to
               max_new_tokens so every model does the same amount of work),
               after `--warmup` untimed generations. CUDA is synchronised on
               both sides of each timing.
"""

from __future__ import annotations

import argparse
import gc
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

# 100 loads would otherwise print 100 weight-loading progress bars into the log.
transformers.utils.logging.set_verbosity_error()
transformers.utils.logging.disable_progress_bar()

# The pipeline package lives one level up (experiments/palaash/q1).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from q1.config import MAX_NEW_TOKENS                      # noqa: E402
from q1.model_utils import build_prompt_ids               # noqa: E402


# ── What we benchmark ─────────────────────────────────────────────────────────
# Column order of both CSVs. Keys are the short names used on the command line.
MODELS = {
    "llama-1b":  "meta-llama/Llama-3.2-1B-Instruct",
    "llama-3b":  "meta-llama/Llama-3.2-3B-Instruct",
    "qwen-0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen-3b":   "Qwen/Qwen2.5-3B-Instruct",
}

# Row order of both CSVs (any GPU not listed is appended at the end).
GPU_ORDER = ["2080", "3060", "A100"]

# Fixed prompt for every model / GPU so the inference numbers are comparable.
BENCH_QUESTION = "What is the capital city of Bhutan?"

# Column order of raw_timings.csv — fixed, because rows are appended headerless.
RAW_COLUMNS = ["gpu", "gpu_name", "model", "model_id", "dtype", "timestamp",
               "phase", "iteration", "seconds", "new_tokens", "error"]

RESULTS_DIR = Path(__file__).resolve().parent / "results"
LOAD_CSV = RESULTS_DIR / "model_load_time_seconds.csv"
INFER_CSV = RESULTS_DIR / "model_inference_time_ms.csv"
RAW_CSV = RESULTS_DIR / "raw_timings.csv"


# ── Device identification ─────────────────────────────────────────────────────
def pick_device() -> str:
    """cuda > mps > cpu, matching q1.model_utils.pick_device."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def describe_device(device: str) -> tuple[str, str]:
    """Return (short label used as the CSV row, full device name).

    The label must describe the device we actually benchmark on, not whatever
    card happens to be in the box — `--device cpu` on a GPU node is a CPU row.
    For CUDA the label is matched out of the device name, e.g. "NVIDIA GeForce
    RTX 2080 Ti" -> "2080", "NVIDIA A100-SXM4-40GB" -> "A100". Anything
    unrecognised falls back to the full name (override with --gpu).
    """
    if device == "cuda" and torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        upper = name.upper()
        for label in ("2080", "3060", "A100", "A6000", "V100", "H100", "L40", "3090", "4090"):
            if label in upper:
                return label, name
        return name, name
    if device == "mps":
        return "mps", f"Apple {platform.machine()} (MPS)"
    return "cpu", platform.processor() or platform.machine() or "cpu"


def pick_dtype(requested: str, device: str) -> torch.dtype:
    """bfloat16 where the accelerator supports it, float16 otherwise.

    Turing cards (2080) have no native bf16, so forcing bf16 there would time
    an emulated path rather than what the model would really run at. The same
    argument rules out half precision on CPU, where fp16 kernels are emulated
    and pathologically slow — CPU runs default to fp32.
    """
    if requested == "bf16":
        if device == "cuda" and not torch.cuda.is_bf16_supported():
            print("  [warn] this card has no native bf16 — timing an emulated path")
        return torch.bfloat16
    if requested == "fp16":
        return torch.float16
    if requested == "fp32":
        return torch.float32
    if device == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if device == "cpu":
        return torch.float32
    return torch.float16


# ── Timing primitives ─────────────────────────────────────────────────────────
def _sync(device: str) -> None:
    """Block until queued work finishes — GPU kernels are launched async."""
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def load_once(model_id: str, device: str, dtype: torch.dtype):
    """Load tokenizer + model onto `device`; return (tokenizer, model, seconds)."""
    _sync(device)
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        device_map={"": device},
    )
    model.eval()
    _sync(device)
    return tok, model, time.perf_counter() - t0


def reclaim() -> None:
    """Collect unreferenced models and hand their VRAM back to the driver.

    This only reclaims what nothing points at any more, so every caller must
    drop its own reference (`model = None`) *before* calling — a `free(model)`
    helper cannot do that for you, since deleting the parameter inside the
    helper leaves the caller's name bound and the weights resident.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def time_loads(model_id: str, device: str, dtype: torch.dtype, repeats: int):
    """Time `repeats` loads, freeing between each. Returns (times, tok, model).

    The last loaded copy is kept alive and handed back so the inference phase
    does not pay for yet another load.
    """
    # Warm-up load: pulls the weights into the HF cache / OS page cache so the
    # timed loads all measure the same (warm) path.
    tok, model, warm = load_once(model_id, device, dtype)
    print(f"  warm-up load: {warm:8.3f} s")

    times: list[float] = []
    for i in range(repeats):
        # Release the previous copy *before* the next load allocates, so peak
        # VRAM is one model and not two — otherwise a 3B model that fits fine
        # OOMs here on an 8 GB card.
        model = None
        reclaim()
        tok, model, secs = load_once(model_id, device, dtype)
        times.append(secs)
        if (i + 1) % 10 == 0 or i == repeats - 1:
            print(f"  load {i + 1:4d}/{repeats}  last {secs:7.3f} s  "
                  f"mean {sum(times) / len(times):7.3f} s")
    return times, tok, model            # final copy stays loaded for inference


@torch.no_grad()
def time_inference(tok, model, device: str, repeats: int, warmup: int,
                   max_new_tokens: int) -> list[float]:
    """Time `repeats` greedy generations of exactly `max_new_tokens` tokens."""
    ids = build_prompt_ids(tok, BENCH_QUESTION, device)
    attn = torch.ones_like(ids)

    def one_generation():
        model.generate(
            ids,
            attention_mask=attn,
            max_new_tokens=max_new_tokens,
            min_new_tokens=max_new_tokens,   # same work for every model
            do_sample=False,                 # greedy -> deterministic
            pad_token_id=tok.eos_token_id,
        )

    for _ in range(warmup):                  # CUDA context, kernel autotuning, allocator
        one_generation()

    times: list[float] = []
    for i in range(repeats):
        _sync(device)
        t0 = time.perf_counter()
        one_generation()
        _sync(device)
        times.append(time.perf_counter() - t0)
        if (i + 1) % 20 == 0 or i == repeats - 1:
            mean_ms = 1000 * sum(times) / len(times)
            print(f"  infer {i + 1:4d}/{repeats}  mean {mean_ms:8.2f} ms  "
                  f"({max_new_tokens / (mean_ms / 1000):6.1f} tok/s)")
    return times


# ── CSV writing ───────────────────────────────────────────────────────────────
def upsert_cell(csv_path: Path, gpu: str, model_key: str, value: float) -> None:
    """Set one (gpu row, model column) cell, creating/extending the CSV."""
    if csv_path.exists():
        df = pd.read_csv(csv_path, index_col="gpu")
    else:
        df = pd.DataFrame(index=pd.Index([], name="gpu"))

    for col in MODELS:                       # keep all four columns, in order
        if col not in df.columns:
            df[col] = pd.NA
    df.loc[gpu, model_key] = value

    # A column created from pd.NA is object dtype, and to_csv's float_format is
    # only applied to float columns — without this the CSV gets full repr noise
    # like 0.32546864612959325 instead of 0.3255.
    df = df[list(MODELS)].apply(pd.to_numeric, errors="coerce")
    known = [g for g in GPU_ORDER if g in df.index]
    df = df.reindex(known + [g for g in df.index if g not in known])
    df.index.name = "gpu"
    df.to_csv(csv_path, float_format="%.4f")


def append_raw(rows: list[dict]) -> None:
    # Pin the column order: appending with mode="a" writes no header, so a
    # differently-ordered frame would silently shift values into wrong columns.
    df = pd.DataFrame(rows, columns=RAW_COLUMNS)
    df.to_csv(RAW_CSV, mode="a", header=not RAW_CSV.exists(), index=False)


# ── Entry point ───────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", choices=sorted(MODELS), default=sorted(MODELS),
                    help="subset of models to benchmark (default: all four)")
    ap.add_argument("--repeats", type=int, default=100,
                    help="timed repeats for both phases (default: 100)")
    ap.add_argument("--load-repeats", type=int, default=None,
                    help="override the number of timed loads (default: --repeats)")
    ap.add_argument("--infer-repeats", type=int, default=None,
                    help="override the number of timed generations (default: --repeats)")
    ap.add_argument("--warmup", type=int, default=3,
                    help="untimed generations before the inference timings (default: 3)")
    ap.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS,
                    help=f"tokens generated per timed inference (default: {MAX_NEW_TOKENS})")
    ap.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto",
                    help="weight dtype; auto = bf16 if the card supports it, "
                         "fp32 on cpu, else fp16")
    ap.add_argument("--gpu", default=None,
                    help="row label for this run (default: detected from the device)")
    ap.add_argument("--device", default=None, choices=["cuda", "cpu", "mps"],
                    help="device to benchmark on (default: cuda, else mps, else cpu)")
    args = ap.parse_args()

    load_repeats = args.load_repeats if args.load_repeats is not None else args.repeats
    infer_repeats = args.infer_repeats if args.infer_repeats is not None else args.repeats

    # Both phases average over their timings, so zero repeats is a divide-by-zero.
    if load_repeats < 1 or infer_repeats < 1:
        ap.error("repeats must be >= 1")
    if args.warmup < 0:
        ap.error("--warmup must be >= 0")
    if args.max_new_tokens < 1:
        ap.error("--max-new-tokens must be >= 1")

    device = args.device or pick_device()
    detected, device_name = describe_device(device)
    gpu = args.gpu or detected
    dtype = pick_dtype(args.dtype, device)
    stamp = datetime.now().isoformat(timespec="seconds")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"device      : {device} ({device_name})")
    print(f"row label   : {gpu}")
    print(f"dtype       : {str(dtype).replace('torch.', '')}")
    print(f"torch       : {torch.__version__}")
    print(f"repeats     : {load_repeats} loads / {infer_repeats} generations "
          f"of {args.max_new_tokens} tokens")
    print(f"models      : {', '.join(args.models)}\n")

    for key in args.models:
        model_id = MODELS[key]
        print(f"[{key}] {model_id}")
        common = dict(gpu=gpu, gpu_name=device_name, model=key, model_id=model_id,
                      dtype=str(dtype).replace("torch.", ""), timestamp=stamp)
        tok = model = None
        try:
            load_times, tok, model = time_loads(model_id, device, dtype, load_repeats)
            infer_times = time_inference(tok, model, device, infer_repeats,
                                         args.warmup, args.max_new_tokens)
        except Exception as exc:             # OOM / gated repo / missing weights
            print(f"  [skip] {key} on {gpu}: {type(exc).__name__}: {exc}\n")
            append_raw([{**common, "phase": "error", "iteration": 0,
                         "seconds": float("nan"), "new_tokens": 0,
                         "error": f"{type(exc).__name__}: {exc}"}])
            continue
        finally:
            # Runs on the `continue` too: an OOM part-way through must not leave
            # this model resident while the next one loads.
            tok = model = None
            reclaim()

        mean_load = sum(load_times) / len(load_times)
        mean_infer_ms = 1000 * sum(infer_times) / len(infer_times)
        upsert_cell(LOAD_CSV, gpu, key, mean_load)
        upsert_cell(INFER_CSV, gpu, key, mean_infer_ms)
        append_raw(
            [{**common, "phase": "load", "iteration": i, "seconds": t,
              "new_tokens": 0, "error": ""} for i, t in enumerate(load_times)]
            + [{**common, "phase": "inference", "iteration": i, "seconds": t,
                "new_tokens": args.max_new_tokens, "error": ""}
               for i, t in enumerate(infer_times)]
        )
        print(f"  -> mean load {mean_load:.3f} s | "
              f"mean inference {mean_infer_ms:.2f} ms\n")

    print(f"load times      -> {LOAD_CSV}")
    print(f"inference times -> {INFER_CSV}")
    print(f"raw timings     -> {RAW_CSV}")
    for path in (LOAD_CSV, INFER_CSV):
        if path.exists():
            print(f"\n{path.name}\n{pd.read_csv(path, index_col='gpu')}")


if __name__ == "__main__":
    main()
