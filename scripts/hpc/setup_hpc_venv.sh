#!/bin/bash
# Run once on the HPC login node after transferring the project.
set -euo pipefail

cd "$(dirname "$0")/../.."

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_PATH="${VENV_PATH:-$PWD/.venv-hpc}"
export HF_HOME="${HF_HOME:-$PWD/.hf-cache}"

"$PYTHON_BIN" -m venv "$VENV_PATH"
source "$VENV_PATH/bin/activate"
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install vllm transformers huggingface_hub
python -m pip freeze > "$VENV_PATH/romireason-hpc-freeze.txt"

echo "HPC environment ready: $VENV_PATH"
echo "Set HF_HOME=$HF_HOME if you want the model cache outside the project directory."
echo "Then activate it and run: hf auth login"
