#!/usr/bin/env bash
# Submit the 36 frozen in-domain SFT held-out evaluation runs.
set -euo pipefail

project_root="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$project_root"
cache_root="${ROMIREASON_CACHE_ROOT:-/scratch/$USER/romireason-bn-cache}"
jobs_root="${SFT_HELDOUT_JOBS_ROOT:-$cache_root/evaluation/in_domain_sft_v1_test}"
results_root="${SFT_HELDOUT_RESULTS_ROOT:-$cache_root/results/in_domain_sft_v1_test}"
adapters_root="${SFT_OUTPUT_ROOT:-/scratch/$USER/romireason-bn-sft}"
partition="${SFT_EVAL_PARTITION:-GPU}"
gpu_type="${SFT_EVAL_GPU_TYPE:-A100}"
skip_runs=",${SKIP_RUNS:-},"

models=(Qwen3-8B Qwen3-32B aya-expanse-8b aya-expanse-32b)
conditions=(native_only romanized_only mixed)
seeds=(20271011 20271012 20271013)

for model in "${models[@]}"; do
  if [[ "$model" == Qwen3-32B || "$model" == aya-expanse-32b ]]; then
    gpu_count=2
    cpus=16
  else
    gpu_count=1
    cpus=8
  fi
  for condition in "${conditions[@]}"; do
    for seed in "${seeds[@]}"; do
      run_key="${model}:${condition}:${seed}"
      if [[ "$skip_runs" == *",${run_key},"* ]]; then
        echo "Skipping explicitly excluded run: $run_key"
        continue
      fi
      jobs="$jobs_root/$model/$condition/seed-$seed/inference.jsonl"
      adapter="$adapters_root/$model/$condition/seed-$seed/best_adapter"
      results="$results_root/$model/$condition/seed-$seed/responses"
      events="$results_root/$model/$condition/seed-$seed/events"
      if [[ ! -f "$jobs" || ! -f "$adapter/adapter_config.json" ]]; then
        echo "Missing jobs or adapter for $run_key" >&2
        exit 2
      fi
      if [[ "${SKIP_COMPLETED:-1}" == 1 && -f "$results/.complete" ]]; then
        echo "Skipping completed run: $run_key"
        continue
      fi
      sbatch \
        --partition="$partition" \
        --gres="gpu:${gpu_type}:${gpu_count}" \
        --cpus-per-task="$cpus" \
        --mem=80G \
        --time=2-00:00:00 \
        --job-name="rrbn-sft-eval-${model}-${condition}-${seed}" \
        --output="slurm-%x-%j.out" \
        --error="slurm-%x-%j.err" \
        --export="ALL,JOBS=$jobs,RESULTS_DIR=$results,EVENTS_DIR=$events,LORA_PATH=$adapter,LORA_NAME=${model}-${condition}-${seed},TENSOR_PARALLEL_SIZE=$gpu_count" \
        scripts/hpc/run_in_domain_sft_heldout_eval.sh
    done
  done
done
