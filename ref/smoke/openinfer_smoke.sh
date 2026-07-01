#!/usr/bin/env bash
# openinfer Rust+CUDA inference engine smoke.
#
# Real smoke is two-phase:
#   (A) cargo check on a *Rust-only* crate (openinfer-engine) — proves the
#       type/contract layer compiles cleanly with the pinned nightly toolchain.
#       This is the "does the code we can read without a GPU still make sense"
#       test.
#   (B) cargo check on the default server (openinfer-server) which pulls in
#       the CUDA-built openinfer-kernels. This is expected to fail on this
#       host because the pinned flashinfer 3rdparty commit (d768c14) expects
#       a newer CCCL (libcu++ `cuda/cmath` top-level header) than the host's
#       CUDA 12.6.20 ships. We report the failure honestly.
#
# Neither path actually launches the model server. To do that you would also
# need a Qwen3-4B weights directory and a CUDA-13.x toolchain on the host.
#
# Mistral: not supported. The workspace has crates for qwen3, qwen35,
# deepseek-v2/v4, glm52, kimi-k2 — no mistral crate.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
OI="$HERE/../openinfer"
export PATH=$HOME/.cargo/bin:/home/a/go/bin:/home/a/yalm/.venv/bin:$PATH
export PKG_CONFIG_PATH=$OI/../smoke/.deps/extracted/usr/lib/x86_64-linux-gnu/pkgconfig
export OPENSSL_DIR=$OI/../smoke/.deps/extracted/usr
export OPENSSL_LIB_DIR=$OI/../smoke/.deps/extracted/usr/lib/x86_64-linux-gnu
export OPENSSL_INCLUDE_DIR=$OI/../smoke/.deps/extracted/usr/include
export PROTOC=$OI/../smoke/.deps/extracted/usr/bin/protoc
export LD_LIBRARY_PATH=$OI/../smoke/.deps/extracted/usr/lib/x86_64-linux-gnu
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH

# 0. nightly toolchain present
rustc_ver=$(rustc --version 2>&1 | head -1)
echo "rustc: $rustc_ver"

cd "$OI"

# (A) Rust-only crate check
echo
echo "--- (A) cargo check -p openinfer-engine (Rust-only) ---"
if cargo check -p openinfer-engine 2> /tmp/openinfer_engine.log; then
  echo "(A) PASS: openinfer-engine rmeta produced"
  A_RESULT="PASS"
else
  echo "(A) FAIL"
  tail -20 /tmp/openinfer_engine.log
  A_RESULT="FAIL"
fi

# (B) Default server (pulls CUDA kernels)
echo
echo "--- (B) cargo check -p openinfer-server (default, includes CUDA build) ---"
if cargo check -p openinfer-server 2> /tmp/openinfer_server.log; then
  echo "(B) PASS"
  B_RESULT="PASS"
else
  # Look for the specific CCCL/flashinfer header issue
  cccl_err=$(grep -E "cuda/cmath|flashinfer.*maximum|cuda::maximum|libcu\+\+" /tmp/openinfer_server.log | head -3)
  if [[ -n "$cccl_err" ]]; then
    echo "(B) EXPECTED FAIL: pinned flashinfer 3rdparty needs newer CCCL than host's CUDA 12.6.20"
    echo "  ---- first CCCL errors ----"
    echo "$cccl_err" | sed 's/^/  /'
    echo "  ----------------------------"
    B_RESULT="EXPECTED FAIL (CCCL version)"
  else
    echo "(B) FAIL (unrelated; first lines of log below)"
    head -30 /tmp/openinfer_server.log
    B_RESULT="FAIL"
  fi
fi

# Mistral status
if find "$OI" -name "*mistral*" -not -path "*/target/*" 2>/dev/null | grep -q .; then
  echo
  echo "Mistral: NO model crate, but found incidental mentions:"
  find "$OI" -name "*mistral*" -not -path "*/target/*" 2>/dev/null | head -3 | sed 's/^/  /'
else
  echo
  echo "Mistral: NO (workspace has qwen3, qwen35, deepseek-v2/v4, glm52, kimi-k2 — no mistral crate)"
fi

echo
echo "Result: engine_rust=$A_RESULT server_full=$B_RESULT"
[[ "$A_RESULT" == "PASS" ]] || exit 1
echo "PASS: openinfer smoke (Rust type layer compiles; CUDA build needs newer CCCL)"
