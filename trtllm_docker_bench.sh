#!/usr/bin/env bash
# Run trtllm_mistral.py inside the official TensorRT-LLM container.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
IMAGE="${TRTLLM_IMAGE:-nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc19}"

if ! docker info >/dev/null 2>&1; then
  echo "docker is not running" >&2
  exit 1
fi

if ! docker run --rm --gpus all "${IMAGE}" python3 -c "import tensorrt_llm" >/dev/null 2>&1; then
  echo "TensorRT-LLM container has no GPU (install nvidia-container-toolkit, restart docker)" >&2
  echo "  https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html" >&2
  exit 1
fi

exec docker run --rm --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
  -v "${ROOT}:${ROOT}" -w "${ROOT}" \
  "${IMAGE}" \
  python3 trtllm_mistral.py "$@"
