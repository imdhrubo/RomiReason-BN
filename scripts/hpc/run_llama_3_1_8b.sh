#!/bin/bash
# Edit only these site-specific resource lines before submission.
#SBATCH --partition=CHANGE_ME
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --gpus=2
#SBATCH --array=0-234%1
#SBATCH --job-name=rrbn-llama8b
#SBATCH --output=slurm-rrbn-llama8b-%A_%a.out

set -euo pipefail
cd "$(dirname "$0")/../.."
VENV_PATH="${VENV_PATH:-$PWD/.venv-hpc}"
export HF_HOME="${HF_HOME:-$PWD/.hf-cache}"
if [[ ! -f "$VENV_PATH/bin/activate" ]]; then
  echo "Missing HPC virtual environment: $VENV_PATH. Run scripts/hpc/setup_hpc_venv.sh first." >&2
  exit 1
fi
source "$VENV_PATH/bin/activate"

python scripts/hpc/run_vllm_jsonl.py \
  --jobs data/evaluation/primary_v1_1_inference_inputs/llama-3-1-8b-instruct.jsonl \
  --shard-index "$SLURM_ARRAY_TASK_ID" \
  --shard-size 500 \
  --output "results/primary_v1_1/llama-3-1-8b-instruct/responses/shard-$(printf '%05d' "$SLURM_ARRAY_TASK_ID").jsonl" \
  --event-log "results/primary_v1_1/llama-3-1-8b-instruct/events/shard-$(printf '%05d' "$SLURM_ARRAY_TASK_ID").jsonl" \
  --tensor-parallel-size 2 \
  --dtype bfloat16
