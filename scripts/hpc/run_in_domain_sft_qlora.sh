#!/usr/bin/env bash
#SBATCH --job-name=rrbn-sft
#SBATCH --partition=GPU
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
module load python/3.12.3
project_root="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$project_root"
source .venv-sft/bin/activate

: "${MODEL:?Set MODEL, e.g. Qwen/Qwen3-8B}"
: "${REVISION:?Set REVISION to the pinned model commit}"
: "${CONDITION:?Set CONDITION to native_only, romanized_only, or mixed}"
: "${SEED:?Set SEED to one frozen training seed}"

sft_root="${SFT_ROOT:-$project_root/data/finetuning/romireason_in_domain_v1}"
output_root="${SFT_OUTPUT_ROOT:-/scratch/$USER/romireason-bn-sft}"
cache_root="${SFT_CACHE_ROOT:-/scratch/$USER/romireason-bn-sft-cache}"
export HF_HOME="${HF_HOME:-$cache_root/hf}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$cache_root/xdg}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$cache_root/triton}"
mkdir -p "$output_root" "$HF_HOME" "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR"
case "$CONDITION" in
  native_only|romanized_only|mixed) ;;
  *) echo "Unknown CONDITION: $CONDITION" >&2; exit 2 ;;
esac

python scripts/hpc/train_qlora_sft.py \
  --model "$MODEL" --revision "$REVISION" --seed "$SEED" \
  --train "$sft_root/${CONDITION}_train.jsonl" \
  --development "$sft_root/${CONDITION}_development.jsonl" \
  --output "$output_root/${MODEL##*/}/${CONDITION}/seed-${SEED}" \
  --max-steps "${MAX_STEPS:-1000}" \
  --max-length "${MAX_LENGTH:-2048}" \
  --learning-rate "${LEARNING_RATE:-2e-4}" \
  --gradient-accumulation "${GRADIENT_ACCUMULATION:-16}"
