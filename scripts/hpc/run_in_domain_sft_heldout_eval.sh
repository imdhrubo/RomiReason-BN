#!/usr/bin/env bash
# Run one SFT adapter over all incomplete held-out evaluation shards.
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
module load python/3.12.3

: "${JOBS:?Set JOBS to the answer-key-free inference JSONL}"
: "${RESULTS_DIR:?Set RESULTS_DIR}"
: "${EVENTS_DIR:?Set EVENTS_DIR}"
: "${LORA_PATH:?Set LORA_PATH to the run best_adapter directory}"
: "${LORA_NAME:?Set LORA_NAME to a stable adapter label}"
: "${TENSOR_PARALLEL_SIZE:?Set TENSOR_PARALLEL_SIZE}"
runner="${RUNNER:-vllm}"

venv_path="${VENV_PATH:-$PWD/.venv-hpc}"
if [[ ! -f "$venv_path/bin/activate" ]]; then
  echo "Missing evaluation virtual environment: $venv_path" >&2
  exit 2
fi
source "$venv_path/bin/activate"

cache_root="${ROMIREASON_CACHE_ROOT:-/scratch/$USER/romireason-bn-cache}"
export HF_HOME="${HF_HOME:-$cache_root/hf}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$cache_root/xdg}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$cache_root/vllm}"

case "$runner" in
  vllm) runner_script="scripts/hpc/run_vllm_jsonl.py" ;;
  transformers_peft) runner_script="scripts/hpc/run_transformers_peft_jsonl.py" ;;
  *) echo "Unknown RUNNER: $runner" >&2; exit 2 ;;
esac

runner_args=(
  --jobs "$JOBS" \
  --shard-size "${SHARD_SIZE:-500}" \
  --all-shards \
  --output-dir "$RESULTS_DIR" \
  --event-dir "$EVENTS_DIR" \
  --lora-path "$LORA_PATH" \
  --lora-name "$LORA_NAME" \
  --dtype "${DTYPE:-bfloat16}"
)
if [[ "$runner" == vllm ]]; then
  runner_args+=(--tensor-parallel-size "$TENSOR_PARALLEL_SIZE")
else
  runner_args+=(--batch-size "${TRANSFORMERS_BATCH_SIZE:-1}")
fi
python "$runner_script" "${runner_args[@]}"

touch "$RESULTS_DIR/.complete"
