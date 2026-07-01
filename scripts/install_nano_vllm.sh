#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY=ref/nano-vllm/.venv/bin/python
U="uv pip install --python $PY"
FA_WHEEL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"

uv venv --clear --python 3.12 ref/nano-vllm/.venv
$U torch --index-url https://download.pytorch.org/whl/cu124
$U transformers triton xxhash safetensors sentencepiece huggingface_hub einops
$U psutil ninja packaging wheel setuptools
# ponytail: FA 2.8 wheels break on torch 2.6+cu124 (ABI); use 2.7.4.post1 wheel
$U "$FA_WHEEL" --no-deps
$U -e ref/nano-vllm --no-deps
echo "ok: $($PY -c 'import torch, flash_attn, nanovllm; print(torch.__version__, flash_attn.__version__)')"
