#!/usr/bin/env bash
set -euo pipefail
module load python/3.12.3
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
venv_dir="${SFT_VENV:-/scratch/$USER/romireason-bn-sft-venv}"
cache_dir="${SFT_PIP_CACHE:-/scratch/$USER/romireason-bn-sft-cache/pip}"
mkdir -p "$cache_dir"
python -m venv "$venv_dir"
source "$venv_dir/bin/activate"
export PIP_CACHE_DIR="$cache_dir"
python -m pip install --upgrade pip
python -m pip install "torch>=2.6,<2.9" "transformers>=4.56,<5" "datasets>=3,<5" \
  "accelerate>=1,<2" "peft>=0.17,<1" "bitsandbytes>=0.45,<1" sentencepiece
python -m pip install -e "$project_root"
echo "SFT_ENV_READY: $(python --version) at $venv_dir"
