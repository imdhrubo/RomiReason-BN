#!/bin/bash
# CPU-only, one-time faithful merge of the published BongLLaMA-13B LoRA checkpoint.
#SBATCH --partition=Orion
#SBATCH --time=1-00:00:00
#SBATCH --mem=80G
#SBATCH --cpus-per-task=16
#SBATCH --job-name=rrbn-merge-bong13
#SBATCH --output=slurm-rrbn-merge-bong13-%j.out
#SBATCH --error=slurm-rrbn-merge-bong13-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
module load python/3.12.3
VENV_PATH="${VENV_PATH:-$PWD/.venv-hpc}"
RR_CACHE_ROOT="${ROMIREASON_CACHE_ROOT:-/scratch/$USER/romireason-bn-cache}"
export HF_HOME="${HF_HOME:-$RR_CACHE_ROOT/hf}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$RR_CACHE_ROOT/xdg}"
SOURCE_DIR="$RR_CACHE_ROOT/models/bongllama-13b-source"
MERGED_DIR="$RR_CACHE_ROOT/models/bongllama-13b-instruct-v0-1-merged"
source "$VENV_PATH/bin/activate"

python -c "from huggingface_hub import snapshot_download; snapshot_download('BanglaLLM/bangla-llama-13b-instruct-v0.1', revision='a5986f8599cd749ab4e8a2afb068b881493dac28', local_dir='$SOURCE_DIR')"
python scripts/hpc/merge_bongllama_lora_checkpoint.py --source "$SOURCE_DIR" --output "$MERGED_DIR"
