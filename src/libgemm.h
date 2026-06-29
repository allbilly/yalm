#pragma once

#include <cuda_runtime.h>

enum class LibGemmBackend {
  CUBLAS,
  CUBLASLT,
  CUDNN,
  TILE, // CUDA tile-style matvec (not NVIDIA cuTILE Python DSL)
};

void libgemm_init(cudaStream_t stream, LibGemmBackend backend);
void libgemm_shutdown();

// y = alpha * W @ x + beta * y; W is (d, n) row-major fp16, x (n,) fp32, y (d,) fp32.
void libgemm_matvec(
  const void* w, const float* x, float* y,
  int n, int d, float alpha, float beta, cudaStream_t stream
);

// Used by cuDNN / cuBLASLt when their matvec path is unavailable.
void libgemm_matvec_cublas_only(
  const void* w, const float* x, float* y,
  int n, int d, float alpha, float beta, cudaStream_t stream
);
