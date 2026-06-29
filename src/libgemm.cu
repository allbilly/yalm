#include "libgemm.h"

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <unordered_map>

#if defined(YALM_CUDNN)
void libgemm_cudnn_init(cudaStream_t stream);
void libgemm_cudnn_shutdown();
void libgemm_matvec_cudnn(
  const void* w, const float* x, float* y,
  int n, int d, float alpha, float beta, cudaStream_t stream
);
#endif

#define CUBLAS_CHECK(x)                                                                                \
  do {                                                                                               \
    cublasStatus_t err = (x);                                                                        \
    if (err != CUBLAS_STATUS_SUCCESS) {                                                              \
      fprintf(stderr, "cuBLAS error in %s at %s:%d: %d\n", __FUNCTION__, __FILE__, __LINE__, err); \
      abort();                                                                                       \
    }                                                                                                \
  } while (0)

#define CUBLASLT_CHECK(x)                                                                                \
  do {                                                                                                   \
    cublasStatus_t err = (x);                                                                            \
    if (err != CUBLAS_STATUS_SUCCESS) {                                                                  \
      fprintf(stderr, "cuBLASLt error in %s at %s:%d: %d\n", __FUNCTION__, __FILE__, __LINE__, err); \
      abort();                                                                                           \
    }                                                                                                    \
  } while (0)

static cublasHandle_t g_cublas = nullptr;
static cublasLtHandle_t g_cublaslt = nullptr;
static LibGemmBackend g_backend = LibGemmBackend::CUBLAS;
static half* g_x_f16 = nullptr;
static size_t g_x_f16_cap = 0;

__global__ void f32_to_f16_kernel(const float* in, half* out, int n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) out[i] = __float2half(in[i]);
}

static half* x_f16_scratch(int n) {
  size_t bytes = static_cast<size_t>(n) * sizeof(half);
  if (bytes > g_x_f16_cap) {
    if (g_x_f16) cudaFree(g_x_f16);
    if (cudaMalloc(&g_x_f16, bytes) != cudaSuccess) abort();
    g_x_f16_cap = bytes;
  }
  return g_x_f16;
}

static void f32_to_f16(const float* in, half* out, int n, cudaStream_t stream) {
  int blocks = (n + 255) / 256;
  f32_to_f16_kernel<<<blocks, 256, 0, stream>>>(in, out, n);
}

struct LtPlan {
  cublasLtMatmulDesc_t op = nullptr;
  cublasLtMatrixLayout_t a = nullptr;
  cublasLtMatrixLayout_t b = nullptr;
  cublasLtMatrixLayout_t c = nullptr;
  cublasLtMatmulPreference_t pref = nullptr;
  cublasLtMatmulAlgo_t algo{};
  bool has_algo = false;
  size_t workspace = 0;
  void* workspace_dev = nullptr;
  int n = 0;
  int d = 0;

  LtPlan() = default;
  LtPlan(LtPlan&& o) noexcept
    : op(o.op), a(o.a), b(o.b), c(o.c), pref(o.pref), algo(o.algo),
      has_algo(o.has_algo), workspace(o.workspace), workspace_dev(o.workspace_dev),
      n(o.n), d(o.d) {
    o.op = nullptr;
    o.a = nullptr;
    o.b = nullptr;
    o.c = nullptr;
    o.pref = nullptr;
    o.workspace_dev = nullptr;
  }
  LtPlan& operator=(LtPlan&&) = delete;
  LtPlan(const LtPlan&) = delete;
  LtPlan& operator=(const LtPlan&) = delete;

  ~LtPlan() {
    if (workspace_dev) cudaFree(workspace_dev);
    if (pref) cublasLtMatmulPreferenceDestroy(pref);
    if (c) cublasLtMatrixLayoutDestroy(c);
    if (b) cublasLtMatrixLayoutDestroy(b);
    if (a) cublasLtMatrixLayoutDestroy(a);
    if (op) cublasLtMatmulDescDestroy(op);
  }
};

static std::unordered_map<long long, LtPlan> g_lt_plans;

static long long lt_key(int n, int d) {
  return (static_cast<long long>(n) << 32) | static_cast<unsigned>(d);
}

static LtPlan& lt_plan(int n, int d) {
  long long key = lt_key(n, d);
  auto it = g_lt_plans.find(key);
  if (it != g_lt_plans.end()) return it->second;

  LtPlan p;
  p.n = n;
  p.d = d;
  CUBLASLT_CHECK(cublasLtMatmulDescCreate(&p.op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  cublasOperation_t op_a = CUBLAS_OP_T;
  CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSA, &op_a, sizeof(op_a)));
  CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&p.a, CUDA_R_16F, n, d, n));
  CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&p.b, CUDA_R_16F, n, 1, n));
  CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&p.c, CUDA_R_32F, d, 1, d));
  CUBLASLT_CHECK(cublasLtMatmulPreferenceCreate(&p.pref));
  size_t ws_limit = 32 * 1024 * 1024;
  CUBLASLT_CHECK(cublasLtMatmulPreferenceSetAttribute(
    p.pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_limit, sizeof(ws_limit)));
  int returned = 0;
  cublasLtMatmulHeuristicResult_t result{};
  CUBLASLT_CHECK(cublasLtMatmulAlgoGetHeuristic(
    g_cublaslt, p.op, p.a, p.b, p.c, p.c, p.pref, 1, &result, &returned));
  if (returned == 0) {
    fprintf(stderr, "cuBLASLt: no heuristic for matvec n=%d d=%d\n", n, d);
    abort();
  }
  p.algo = result.algo;
  p.has_algo = true;
  p.workspace = result.workspaceSize;
  if (p.workspace) {
    if (cudaMalloc(&p.workspace_dev, p.workspace) != cudaSuccess) abort();
  }
  g_lt_plans.emplace(key, std::move(p));
  return g_lt_plans.find(key)->second;
}

void libgemm_init(cudaStream_t stream, LibGemmBackend backend) {
  g_backend = backend;
  if (!g_cublas) {
    CUBLAS_CHECK(cublasCreate(&g_cublas));
    CUBLAS_CHECK(cublasSetMathMode(g_cublas, CUBLAS_TF32_TENSOR_OP_MATH));
  }
  if (backend == LibGemmBackend::CUBLASLT && !g_cublaslt) {
    CUBLASLT_CHECK(cublasLtCreate(&g_cublaslt));
  }
#if defined(YALM_CUDNN)
  if (backend == LibGemmBackend::CUDNN) {
    libgemm_cudnn_init(stream);
  }
#endif
  CUBLAS_CHECK(cublasSetStream(g_cublas, stream));
}

void libgemm_shutdown() {
  g_lt_plans.clear();
#if defined(YALM_CUDNN)
  libgemm_cudnn_shutdown();
#endif
  if (g_x_f16) {
    cudaFree(g_x_f16);
    g_x_f16 = nullptr;
    g_x_f16_cap = 0;
  }
  if (g_cublaslt) {
    cublasLtDestroy(g_cublaslt);
    g_cublaslt = nullptr;
  }
  if (g_cublas) {
    cublasDestroy(g_cublas);
    g_cublas = nullptr;
  }
}

static void matvec_cublas(
  const void* w, const float* x, float* y,
  int n, int d, float alpha, float beta, cudaStream_t stream
) {
  CUBLAS_CHECK(cublasSetStream(g_cublas, stream));
  half* x_f16 = x_f16_scratch(n);
  f32_to_f16(x, x_f16, n, stream);
  CUBLAS_CHECK(cublasGemmEx(
    g_cublas, CUBLAS_OP_T, CUBLAS_OP_N,
    d, 1, n,
    &alpha,
    w, CUDA_R_16F, n,
    x_f16, CUDA_R_16F, n,
    &beta,
    y, CUDA_R_32F, d,
    CUBLAS_COMPUTE_32F,
    CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

void libgemm_matvec_cublas_only(
  const void* w, const float* x, float* y,
  int n, int d, float alpha, float beta, cudaStream_t stream
) {
  matvec_cublas(w, x, y, n, d, alpha, beta, stream);
}

#define FULL_MASK 0xffffffffu

__global__ void tile_matvec_kernel(
  const half* w, const float* x, float* y,
  int n, int d, float alpha, float beta
) {
  int row = blockIdx.x;
  if (row >= d) return;
  int lane = threadIdx.x & 31;
  float sum = 0.f;
  const half* row_w = w + (size_t)row * n;
  for (int j = lane; j < n; j += 32) {
    sum += __half2float(row_w[j]) * x[j];
  }
  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    sum += __shfl_down_sync(FULL_MASK, sum, offset);
  }
  if (lane == 0) {
    y[row] = alpha * sum + beta * y[row];
  }
}

static void matvec_tile(
  const void* w, const float* x, float* y,
  int n, int d, float alpha, float beta, cudaStream_t stream
) {
  tile_matvec_kernel<<<(unsigned)d, 32, 0, stream>>>(
    static_cast<const half*>(w), x, y, n, d, alpha, beta);
}

static void matvec_cublaslt(
  const void* w, const float* x, float* y,
  int n, int d, float alpha, float beta, cudaStream_t stream
) {
  half* x_f16 = x_f16_scratch(n);
  f32_to_f16(x, x_f16, n, stream);
  LtPlan& p = lt_plan(n, d);
  cublasLtPointerMode_t mode = CUBLASLT_POINTER_MODE_HOST;
  CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
    p.op, CUBLASLT_MATMUL_DESC_POINTER_MODE, &mode, sizeof(mode)));
  cublasStatus_t st = cublasLtMatmul(
    g_cublaslt, p.op,
    &alpha,
    w, p.a,
    x_f16, p.b,
    &beta,
    y, p.c,
    y, p.c,
    &p.algo,
    p.workspace_dev, p.workspace,
    stream);
  // ponytail: cuBLASLt matvec (n=1) often NOT_SUPPORTED on Ampere; fall back to GemmEx.
  if (st != CUBLAS_STATUS_SUCCESS) {
    matvec_cublas(w, x, y, n, d, alpha, beta, stream);
  }
}

void libgemm_matvec(
  const void* w, const float* x, float* y,
  int n, int d, float alpha, float beta, cudaStream_t stream
) {
  switch (g_backend) {
    case LibGemmBackend::CUBLASLT:
      matvec_cublaslt(w, x, y, n, d, alpha, beta, stream);
      break;
#if defined(YALM_CUDNN)
    case LibGemmBackend::CUDNN:
      libgemm_matvec_cudnn(w, x, y, n, d, alpha, beta, stream);
      break;
#endif
    case LibGemmBackend::TILE:
      matvec_tile(w, x, y, n, d, alpha, beta, stream);
      break;
    default:
      matvec_cublas(w, x, y, n, d, alpha, beta, stream);
      break;
  }
}
