#!/usr/bin/env bash
# Back-compat wrapper — docker logic lives in mistral_bench.py.
set -euo pipefail
cd "$(dirname "$0")"
exec .venv/bin/python mistral_bench.py --engines tensorrt-llm --trtllm-docker "$@"
