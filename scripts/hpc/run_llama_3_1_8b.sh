#!/bin/bash
# One persistent allocation: edit only these site-specific resource lines.
#SBATCH --partition=GPU
#SBATCH --time=3-00:00:00
#SBATCH --mem=80G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:A100:2
#SBATCH --job-name=rrbn-llama8b
#SBATCH --output=slurm-rrbn-llama8b-%j.out
#SBATCH --error=slurm-rrbn-llama8b-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
module load python/3.12.3
VENV_PATH="${VENV_PATH:-$PWD/.venv-hpc}"
export HF_HOME="${HF_HOME:-$PWD/.hf-cache}"
RESULTS_ROOT="${ROMIREASON_RESULTS_ROOT:-$PWD/results}"
if [[ ! -f "$VENV_PATH/bin/activate" ]]; then
  echo "Missing HPC virtual environment: $VENV_PATH. Run scripts/hpc/setup_hpc_venv.sh first." >&2
  exit 1
fi
source "$VENV_PATH/bin/activate"

python scripts/hpc/run_vllm_jsonl.py \
  --jobs data/evaluation/primary_v1_1_inference_inputs/llama-3-1-8b-instruct.jsonl \
  --shard-size 500 \
  --all-shards \
  --output-dir "$RESULTS_ROOT/primary_v1_1/llama-3-1-8b-instruct/responses" \
  --event-dir "$RESULTS_ROOT/primary_v1_1/llama-3-1-8b-instruct/events" \
  --tensor-parallel-size 2 \
  --dtype bfloat16
