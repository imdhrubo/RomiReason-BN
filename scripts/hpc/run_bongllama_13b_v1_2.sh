#!/bin/bash
# Secondary Bengali-specialization size extension; frozen v1.2 protocol, 19,543 items x 6 forms.
#SBATCH --partition=Nebula_GPU
#SBATCH --time=2-00:00:00
#SBATCH --mem=80G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:L40S:1
#SBATCH --job-name=rrbn-bong13-v12
#SBATCH --output=slurm-rrbn-bong13-v12-%j.out
#SBATCH --error=slurm-rrbn-bong13-v12-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
module load python/3.12.3
VENV_PATH="${VENV_PATH:-$PWD/.venv-hpc}"
RR_CACHE_ROOT="${ROMIREASON_CACHE_ROOT:-/scratch/$USER/romireason-bn-cache}"
export HF_HOME="${HF_HOME:-$RR_CACHE_ROOT/hf}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$RR_CACHE_ROOT/xdg}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$RR_CACHE_ROOT/vllm/bongllama13-v1_2}"
RESULTS_ROOT="${ROMIREASON_RESULTS_ROOT:-$RR_CACHE_ROOT/results}"
EVALUATION_ROOT="${ROMIREASON_EVALUATION_ROOT:-$RR_CACHE_ROOT/evaluation}"
BONG13_MERGED_MODEL_DIR="${BONG13_MERGED_MODEL_DIR:-$RR_CACHE_ROOT/models/bongllama-13b-instruct-v0-1-merged}"
mkdir -p "$HF_HOME" "$XDG_CACHE_HOME" "$VLLM_CACHE_ROOT" "$RESULTS_ROOT" "$EVALUATION_ROOT"
if [[ ! -f "$VENV_PATH/bin/activate" ]]; then
  echo "Missing HPC virtual environment: $VENV_PATH. Run scripts/hpc/setup_hpc_venv.sh first." >&2
  exit 1
fi
source "$VENV_PATH/bin/activate"
SHARD_START="${SHARD_START:-${SLURM_ARRAY_TASK_ID:-0}}"
SHARD_STRIDE="${SHARD_STRIDE:-1}"
if [[ ! -f "$BONG13_MERGED_MODEL_DIR/model.safetensors.index.json" ]]; then
  echo "Missing merged BongLLaMA-13B checkpoint: $BONG13_MERGED_MODEL_DIR. Submit scripts/hpc/prepare_bongllama_13b_merged.sh first." >&2
  exit 1
fi

python scripts/hpc/run_vllm_jsonl.py \
  --jobs "$EVALUATION_ROOT/bengali_specialization_size_extension_v1_2_inference_inputs/bongllama-13b-instruct-v0-1.jsonl" \
  --shard-size 500 --all-shards \
  --output-dir "$RESULTS_ROOT/bengali_specialization_size_extension_v1_2/bongllama-13b-instruct-v0-1/responses" \
  --event-dir "$RESULTS_ROOT/bengali_specialization_size_extension_v1_2/bongllama-13b-instruct-v0-1/events" \
  --tensor-parallel-size 1 --dtype bfloat16 --model-path "$BONG13_MERGED_MODEL_DIR" --shard-start "$SHARD_START" --shard-stride "$SHARD_STRIDE"
