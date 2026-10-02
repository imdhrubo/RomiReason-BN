#!/bin/bash
# Qwen template-correction rerun: reasoning protocol only.
#SBATCH --partition=Nebula_GPU
#SBATCH --time=2-00:00:00
#SBATCH --mem=80G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:L40S:2
#SBATCH --job-name=rrbn-qwen32-v14r
#SBATCH --output=slurm-rrbn-qwen32-v14r-%j.out
#SBATCH --error=slurm-rrbn-qwen32-v14r-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
module load python/3.12.3
VENV_PATH="${VENV_PATH:-$PWD/.venv-hpc}"; RR_CACHE_ROOT="${ROMIREASON_CACHE_ROOT:-/scratch/$USER/romireason-bn-cache}"
export HF_HOME="${HF_HOME:-$RR_CACHE_ROOT/hf}" XDG_CACHE_HOME="${XDG_CACHE_HOME:-$RR_CACHE_ROOT/xdg}" VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$RR_CACHE_ROOT/vllm}"
source "$VENV_PATH/bin/activate"
python scripts/hpc/run_vllm_jsonl.py --jobs "$RR_CACHE_ROOT/evaluation/qwen_template_correction_v1_4_reasoning_inference_inputs/qwen3-32b.jsonl" --shard-size 500 --all-shards --output-dir "$RR_CACHE_ROOT/results/qwen_template_correction_v1_4_reasoning/qwen3-32b/responses" --event-dir "$RR_CACHE_ROOT/results/qwen_template_correction_v1_4_reasoning/qwen3-32b/events" --tensor-parallel-size 2 --dtype bfloat16
