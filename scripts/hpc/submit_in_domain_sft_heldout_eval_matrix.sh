#!/usr/bin/env bash
# Submit the frozen multi-protocol in-domain SFT held-out evaluation campaign.
set -euo pipefail

project_root="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$project_root"
cache_root="${ROMIREASON_CACHE_ROOT:-/scratch/$USER/romireason-bn-cache}"
jobs_root="${SFT_HELDOUT_JOBS_ROOT:-$cache_root/evaluation/in_domain_sft_post_evaluation_v1_test}"
results_root="${SFT_HELDOUT_RESULTS_ROOT:-$cache_root/results/in_domain_sft_post_evaluation_v1_test}"
adapters_root="${SFT_OUTPUT_ROOT:-/scratch/$USER/romireason-bn-sft}"
partition="${SFT_EVAL_PARTITION:-GPU}"
gpu_type="${SFT_EVAL_GPU_TYPE:-A100}"
skip_runs=",${SKIP_RUNS:-},"

evaluations=(
  "v1_2_reasoning_aya:aya-expanse-8b,aya-expanse-32b"
  "v1_3_answer_only_aya:aya-expanse-8b,aya-expanse-32b"
  "v1_2_reasoning_qwen_thinking_enabled:Qwen3-8B,Qwen3-32B"
  "v1_4_corrected_answer_only_qwen:Qwen3-8B,Qwen3-32B"
)
conditions=(native_only romanized_only mixed)
seeds=(20271011 20271012 20271013)

for evaluation_spec in "${evaluations[@]}"; do
  IFS=':' read -r evaluation models_csv <<< "$evaluation_spec"
  IFS=',' read -r -a models <<< "$models_csv"
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
      run_key="${evaluation}:${model}:${condition}:${seed}"
      if [[ "$skip_runs" == *",${run_key},"* ]]; then
        echo "Skipping explicitly excluded run: $run_key"
        continue
      fi
      jobs="$jobs_root/$evaluation/$model/$condition/seed-$seed/inference.jsonl"
      adapter="$adapters_root/$model/$condition/seed-$seed/best_adapter"
      results="$results_root/$evaluation/$model/$condition/seed-$seed/responses"
      events="$results_root/$evaluation/$model/$condition/seed-$seed/events"
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
        --job-name="rrbn-sft-eval-${evaluation}-${model}-${condition}-${seed}" \
        --output="slurm-%x-%j.out" \
        --error="slurm-%x-%j.err" \
        --export="ALL,JOBS=$jobs,RESULTS_DIR=$results,EVENTS_DIR=$events,LORA_PATH=$adapter,LORA_NAME=${evaluation}-${model}-${condition}-${seed},TENSOR_PARALLEL_SIZE=$gpu_count" \
        scripts/hpc/run_in_domain_sft_heldout_eval.sh
    done
  done
  done
done
