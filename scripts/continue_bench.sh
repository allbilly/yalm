#!/usr/bin/env bash
# Resilient bench runner — survives agent timeout. Usage:
#   nohup bash scripts/continue_bench.sh >> ref/bench_run.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
NVPY=ref/nano-vllm/.venv/bin/python
UPIP="uv pip install --python ref/nano-vllm/.venv/bin/python"
log() { echo "=== $(date -Is) $* ==="; }

log "continue_bench start"

# --- 1. tinygrad (no nano-vllm deps needed) ---
if ! grep -q 'tok/s' ref/tg_kernels/bench_fused_120.log 2>/dev/null; then
  log "tinygrad fused BEAM=8 count=120"
  FUSE_QKV=1 FUSE_GLU=1 NO_CB=1 BEAM=8 DEV=CUDA \
    .venv/bin/python tinygrad_mistral.py --count 120 \
    | tee ref/tg_kernels/bench_fused_120.log || log "tinygrad fused FAILED"
else
  log "skip tinygrad fused (already in log)"
fi

if ! grep -q 'tok/s' ref/tg_kernels/bench_unfused_120.log 2>/dev/null; then
  log "tinygrad unfused count=120"
  .venv/bin/python tinygrad_mistral.py --count 120 --no-fuse \
    | tee ref/tg_kernels/bench_unfused_120.log || log "tinygrad unfused FAILED"
else
  log "skip tinygrad unfused (already in log)"
fi

# --- 2. nano-vllm env: torch cu124 (matches host CUDA 12.6, not cu130) ---
log "nano-vllm: torch cu124"
$UPIP torch --index-url https://download.pytorch.org/whl/cu124 --reinstall

if ! "$NVPY" -c "import flash_attn" 2>/dev/null; then
  log "nano-vllm: flash-attn build (10–30 min)"
  $UPIP psutil ninja packaging wheel setuptools
  $UPIP flash-attn --no-build-isolation || log "flash-attn FAILED (see install.log)"
fi

if ! "$NVPY" -c "import nanovllm" 2>/dev/null; then
  log "nano-vllm: editable install"
  $UPIP -e ref/nano-vllm --no-deps || $UPIP -e ref/nano-vllm || log "nano-vllm install FAILED"
fi

# --- 3. nano-vllm bench ---
if "$NVPY" -c "import flash_attn, nanovllm" 2>/dev/null; then
  log "nano-vllm mistral count=120"
  "$NVPY" nano_vllm_mistral.py --count 120 | tee ref/nano-vllm/bench_mistral_120.log \
    || log "nano-vllm bench FAILED"
else
  log "SKIP nano-vllm bench (flash_attn or nanovllm missing)"
fi

log "continue_bench done"
