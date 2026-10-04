#!/usr/bin/env bash
set -euo pipefail
module load python/3.12.3
python -m venv .venv-sft
source .venv-sft/bin/activate
python -m pip install --upgrade pip
python -m pip install "torch>=2.6,<2.9" "transformers>=4.56,<5" "datasets>=3,<5" \
  "accelerate>=1,<2" "peft>=0.17,<1" "bitsandbytes>=0.45,<1" sentencepiece
python -m pip install -e .
echo "SFT_ENV_READY: $(python --version)"
