#!/bin/bash
#SBATCH --job-name=llm-bench
#SBATCH --output=logs/bench_%j.out
#SBATCH --error=logs/bench_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
# ── Change the GPU here, one submission per card ──────────────────────────────
#SBATCH --constraint=2080          # then: 3060, then: a100
#
#   sbatch run_benchmark.slurm                      # 100 loads + 100 generations
#   sbatch run_benchmark.slurm --load-repeats 10    # extra args pass through
#
# Each run writes exactly one row (its GPU) into results/*.csv, so submitting
# this three times with three --constraint values fills the table.

set -euo pipefail
mkdir -p logs

module load cuda/12.1 2>/dev/null || true
# source /path/to/your/venv/bin/activate      # <- your environment here

nvidia-smi

cd "$(dirname "$0")"
python benchmark.py "$@"
