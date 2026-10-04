#!/usr/bin/env bash
# Submit the frozen in-domain SFT matrix.  This script only submits jobs; it
# never changes the data split or model revisions.
set -euo pipefail

project_root="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$project_root"

models=(
  "Qwen3-8B|Qwen/Qwen3-8B|b968826d9c46dd6066d109eabc6255188de91218"
  "Qwen3-32B|Qwen/Qwen3-32B|9216db5781bf21249d130ec9da846c4624c16137"
  "Aya-8B|CohereLabs/aya-expanse-8b|5062468bf9bc0c6035fd64e06274333ec127d980"
  "Aya-32B|CohereLabs/aya-expanse-32b|b306ea27e360683b50c005d7fcbad6a242317910"
)
conditions=(native_only romanized_only mixed)
seeds=(20271011 20271012 20271013)

for entry in "${models[@]}"; do
  IFS='|' read -r label model revision <<< "$entry"
  for condition in "${conditions[@]}"; do
    for seed in "${seeds[@]}"; do
      sbatch --job-name="rrbn-sft-${label}-${condition}-${seed}" \
        --export="ALL,MODEL=${model},REVISION=${revision},CONDITION=${condition},SEED=${seed}" \
        scripts/hpc/run_in_domain_sft_qlora.sh
    done
  done
done
