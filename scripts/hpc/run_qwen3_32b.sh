#!/bin/bash
# One persistent allocation: edit only these site-specific resource lines.
#SBATCH --partition=GPU
#SBATCH --time=3-00:00:00
#SBATCH --mem=80G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:A100:2
#SBATCH --job-name=rrbn-qwen32b
#SBATCH --output=slurm-rrbn-qwen32b-%j.out
#SBATCH --error=slurm-rrbn-qwen32b-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
module load python/3.12.3
VENV_PATH="${VENV_PATH:-$PWD/.venv-hpc}"
export HF_HOME="${HF_HOME:-$PWD/.hf-cache}"
if [[ ! -f "$VENV_PATH/bin/activate" ]]; then
  echo "Missing HPC virtual environment: $VENV_PATH. Run scripts/hpc/setup_hpc_venv.sh first." >&2
  exit 1
fi
source "$VENV_PATH/bin/activate"

python scripts/hpc/run_vllm_jsonl.py \
  --jobs data/evaluation/primary_v1_1_inference_inputs/qwen3-32b.jsonl \
  --shard-size 500 \
  --all-shards \
  --output-dir results/primary_v1_1/qwen3-32b/responses \
  --event-dir results/primary_v1_1/qwen3-32b/events \
  --tensor-parallel-size 2 \
  --dtype bfloat16
