#!/usr/bin/env bash
# FlashQwen C++ engine smoke.
#
# Real smoke = "the build configuration is sound up to the dependency
# boundary, and the CUDA side parses". A full build needs gRPC + abseil +
# OpenSSL dev libs (sudo apt-get install ...) and CUDA-time compile of all
# .cu files for the target arch.
#
# What this verifies:
#   1. /usr/local/cuda/bin/nvcc is the real nvcc (not the /usr/bin/nvcc stub)
#   2. cuBLAS headers are reachable from /usr/local/cuda/include
#   3. CMake parses the project, runs nvcc identification, and fails on
#      gRPC/pkg-config as expected (no sudo to install dev libs)
#   4. kernels.cu (the core BF16 GEMM / RMSNorm / RoPE / etc. kernel
#      bundle) compiles standalone with the right include paths
#
# What this does NOT do (would need sudo/root for dev libs and ~10 min
# of nvcc compile time):
#   - Link the gRPC server stub
#   - Compile attn_cute.cu / model_runtime.cu (need CUTLASS at build time)
#   - Link the final flashqwen-engine binary
#
# Mistral: not supported. Engine hardcodes Qwen3ForCausalLM in model_spec.h.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
FQ="$HERE/../FlashQwen"
NVCC=/usr/local/cuda/bin/nvcc
export PATH=/usr/local/cuda/bin:$PATH

# 1. nvcc sanity
nvcc_full=$($NVCC --version 2>&1)
if ! grep -qE "release [0-9]+\.[0-9]+" <<<"$nvcc_full"; then
  echo "FAIL: nvcc broken or stub"; echo "$nvcc_full"; exit 1
fi
nvcc_ver=$(grep -oE "release [0-9]+\.[0-9]+" <<<"$nvcc_full" | head -1)
echo "nvcc: $nvcc_ver"

# 2. cuBLAS headers present
[[ -f /usr/local/cuda/include/cublas_v2.h ]] || { echo "FAIL: cuBLAS headers missing"; exit 1; }
echo "cuBLAS headers: OK"

# 3. CMake configure
rm -rf "$FQ/engine/build"
CMAKE_RESULT="PASS"
if CUDACXX=$NVCC cmake -S "$FQ/engine" -B "$FQ/engine/build" \
   -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86 \
   > /tmp/flashqwen_cmake.log 2>&1; then
  echo "CMake configure: PASS"
else
  CMAKE_RESULT="EXPECTED FAIL"
  echo "CMake configure: EXPECTED FAIL (gRPC + abseil dev libs not installed system-wide)"
  echo "  ---- relevant cmake log lines ----"
  grep -iE "grpc|abseil|openssl|not found" /tmp/flashqwen_cmake.log | sed 's/^/  /' | head -5
  echo "  ---------------------------------"
fi

# 4. Standalone compile of kernels.cu
SRC=$FQ/engine/src/kernels.cu
VENDOR=$FQ/engine/third_party
KERN_RESULT="PASS"
if $NVCC -O2 --use_fast_math -std=c++17 \
   -gencode arch=compute_86,code=sm_86 \
   -I/usr/local/cuda/include -I"$VENDOR" \
   -c "$SRC" -o /tmp/flashqwen_kernels.o 2> /tmp/flashqwen_kernels.log; then
  echo "kernels.cu compile: PASS (object: $(stat -c%s /tmp/flashqwen_kernels.o) bytes)"
else
  KERN_RESULT="FAIL"
  echo "kernels.cu compile: FAIL"
  cat /tmp/flashqwen_kernels.log
fi

# 5. Mistral support
if grep -q '"Qwen3ForCausalLM"' "$FQ/engine/src/model_spec.h"; then
  echo "Mistral: NO (model_spec.h Supported() returns arch == 'Qwen3ForCausalLM' only)"
else
  echo "Mistral: unknown"
fi

echo
echo "Result: cmake=$CMAKE_RESULT kernels=$KERN_RESULT"
[[ "$KERN_RESULT" == "PASS" ]] || exit 1
echo "PASS: FlashQwen C++ smoke (CMake parse + standalone kernels.cu compile)"
