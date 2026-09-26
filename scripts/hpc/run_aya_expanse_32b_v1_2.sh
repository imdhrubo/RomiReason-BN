#!/bin/bash
# One persistent allocation: edit only these site-specific resource lines.
#SBATCH --partition=GPU
#SBATCH --time=3-00:00:00
#SBATCH --mem=80G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:A100:2
#SBATCH --job-name=rrbn-aya32-v12
#SBATCH --output=slurm-rrbn-aya32-v12-%j.out
#SBATCH --error=slurm-rrbn-aya32-v12-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
module load python/3.12.3
VENV_PATH="${VENV_PATH:-$PWD/.venv-hpc}"
RR_CACHE_ROOT="${ROMIREASON_CACHE_ROOT:-/scratch/$USER/romireason-bn-cache}"
export HF_HOME="${HF_HOME:-$RR_CACHE_ROOT/hf}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$RR_CACHE_ROOT/xdg}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$RR_CACHE_ROOT/vllm}"
RESULTS_ROOT="${ROMIREASON_RESULTS_ROOT:-$RR_CACHE_ROOT/results}"
mkdir -p "$HF_HOME" "$XDG_CACHE_HOME" "$VLLM_CACHE_ROOT" "$RESULTS_ROOT"
if [[ ! -f "$VENV_PATH/bin/activate" ]]; then
  echo "Missing HPC virtual environment: $VENV_PATH. Run scripts/hpc/setup_hpc_venv.sh first." >&2
  exit 1
fi
source "$VENV_PATH/bin/activate"

python scripts/hpc/run_vllm_jsonl.py \
  --jobs data/evaluation/primary_v1_2_inference_inputs/aya-expanse-32b.jsonl \
  --shard-size 500 --all-shards \
  --output-dir "$RESULTS_ROOT/primary_v1_2/aya-expanse-32b/responses" \
  --event-dir "$RESULTS_ROOT/primary_v1_2/aya-expanse-32b/events" \
  --tensor-parallel-size 2 --dtype bfloat16
