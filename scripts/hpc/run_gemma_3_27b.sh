#!/bin/bash
# One persistent allocation: edit only these site-specific resource lines.
#SBATCH --partition=GPU
#SBATCH --time=3-00:00:00
#SBATCH --mem=80G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:A100:2
#SBATCH --job-name=rrbn-gemma27b
#SBATCH --output=slurm-rrbn-gemma27b-%j.out
#SBATCH --error=slurm-rrbn-gemma27b-%j.err

set -euo pipefail
cd "$(dirname "$0")/../.."
module load python/3.12.3
VENV_PATH="${VENV_PATH:-$PWD/.venv-hpc}"
export HF_HOME="${HF_HOME:-$PWD/.hf-cache}"
if [[ ! -f "$VENV_PATH/bin/activate" ]]; then
  echo "Missing HPC virtual environment: $VENV_PATH. Run scripts/hpc/setup_hpc_venv.sh first." >&2
  exit 1
fi
source "$VENV_PATH/bin/activate"

python scripts/hpc/run_vllm_jsonl.py \
  --jobs data/evaluation/primary_v1_1_inference_inputs/gemma-3-27b-it.jsonl \
  --shard-size 500 \
  --all-shards \
  --output-dir results/primary_v1_1/gemma-3-27b-it/responses \
  --event-dir results/primary_v1_1/gemma-3-27b-it/events \
  --tensor-parallel-size 2 \
  --dtype bfloat16
