# Load / inference benchmark

Times **model loading** and **model inference** for the four models used by the
Q1 pipeline, on whichever GPU the job lands on. One run = one GPU = one row.

```bash
cd experiments/palaash/benchmark
python benchmark.py                      # 100 timed loads + 100 timed generations, per model
sbatch run_benchmark.slurm               # same, under SLURM (set the GPU in the header)
```

## Output

| file | contents |
|------|----------|
| `results/model_load_time_seconds.csv` | mean seconds to load a model |
| `results/model_inference_time_ms.csv` | mean milliseconds per generation |
| `results/raw_timings.csv` | every individual timing (for std-devs / outliers) |

Both summary CSVs share the same shape — rows are GPUs, columns are the four
models — and each run **upserts only its own row**, so running on the 2080,
then the 3060, then the A100 accumulates the full table:

```csv
gpu,llama-1b,llama-3b,qwen-0.5b,qwen-3b
2080,...,...,...,...
3060,...,...,...,...
A100,...,...,...,...
```

The row label comes from `torch.cuda.get_device_name(0)` (e.g. "NVIDIA GeForce
RTX 2080 Ti" → `2080`); override it with `--gpu A100` if the detection misses.

## What the two numbers mean

* **Load** — `AutoTokenizer.from_pretrained` + `AutoModelForCausalLM.from_pretrained`
  (weights to GPU) + `.eval()`. One untimed warm-up load runs first so the timed
  loads never pay the download / cold page-cache cost, and the previous copy is
  freed (`del` + `gc.collect()` + `empty_cache()`) before each one.
* **Inference** — one greedy `generate` of exactly `MAX_NEW_TOKENS` (24) tokens on
  a fixed chat-formatted prompt, batch size 1, KV cache on. `min_new_tokens` is
  pinned to `max_new_tokens` so early EOS can't make one model look faster by
  generating fewer tokens. `--warmup 3` untimed generations run first, and CUDA
  is synchronised on both sides of every timing.

## Useful flags

```bash
python benchmark.py --load-repeats 10          # loads are the slow part, see below
python benchmark.py --models qwen-0.5b qwen-3b # subset
python benchmark.py --gpu A100                 # force the row label
python benchmark.py --dtype fp16               # force a dtype (default: auto)
python benchmark.py --max-new-tokens 64        # longer generations
python benchmark.py --device cpu --dtype fp32 --repeats 2   # local sanity check
```

## Two things worth knowing before you read the numbers

**100 timed loads is genuinely slow.** A warm load of the 3B models is roughly
5–15 s, so 100 loads ≈ 10–25 min *per model* — most of the job's wall clock,
for a quantity that varies far less than inference does. The default is 100 as
specified; `--load-repeats 10 --infer-repeats 100` gives essentially the same
load mean in a fraction of the time. Note also that all timed loads are **warm**
(weights already in the HF cache and OS page cache) — a first-ever load on a
node with a cold cache is much slower and is deliberately excluded.

**dtype differs across these cards, by design.** `--dtype auto` uses bfloat16
where the card supports it (A100, Ampere and newer) and float16 where it does
not (the 2080 is Turing — bf16 there is emulated and would time a path nobody
actually runs). The dtype used is recorded per-row in `raw_timings.csv`. If you
want a strictly like-for-like cross-GPU comparison, pass `--dtype fp16`
everywhere. Also expect the 3B models to be tight or OOM on an 8 GB 2080 — a
model that fails to load is logged as an `error` row in `raw_timings.csv`, left
blank in the summary CSVs, and the run continues with the next model.
