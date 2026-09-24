#!/bin/bash
# Set these SBATCH values for your site and model. Use a separate submission per model.
#SBATCH --job-name=romireason
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --gpus=2
#SBATCH --array=0-234%1

set -euo pipefail

: "${JOB_FILE:?Set JOB_FILE to one model inference JSONL}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR for raw response shards}"
: "${EVENT_LOG_DIR:?Set EVENT_LOG_DIR for shard event logs}"
: "${TENSOR_PARALLEL_SIZE:?Set TENSOR_PARALLEL_SIZE to allocated GPU count}"
printf -v SHARD_NAME '%05d' "$SLURM_ARRAY_TASK_ID"

python scripts/hpc/run_vllm_jsonl.py \
  --jobs "$JOB_FILE" \
  --shard-index "$SLURM_ARRAY_TASK_ID" \
  --shard-size 500 \
  --output "$OUTPUT_DIR/shard-${SHARD_NAME}.jsonl" \
  --event-log "$EVENT_LOG_DIR/shard-${SHARD_NAME}.jsonl" \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --dtype bfloat16
