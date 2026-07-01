#!/usr/bin/env bash
# Sequential 10-run benches — one GPU job at a time.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=(.venv/bin/python mistral_bench.py --count 120 --runs 10)
log() { echo "=== $(date -Is) $* ==="; }

log "beam=3 tinygrad"
"${PY[@]}" --engines tinygrad --beam 3 | tee ref/tg_kernels/bench_beam3_10run.log

log "beam=8 tinygrad"
"${PY[@]}" --engines tinygrad --beam 8 | tee ref/tg_kernels/bench_beam8_10run.log

log "vllm sglang"
"${PY[@]}" --engines vllm,sglang | tee ref/tg_kernels/bench_serving_10run.log

log "trtllm docker"
"${PY[@]}" --engines tensorrt-llm --trtllm-docker | tee ref/tg_kernels/bench_trtllm_10run.log

log "done"
