#include "libgemm.h"

#if defined(YALM_CUDNN)

#include <cudnn.h>
#include <cudnn_backend.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <unordered_map>
#include <vector>

#define CUDNN_CHECK(x)                                                                                \
  do {                                                                                              \
    cudnnStatus_t err = (x);                                                                        \
    if (err != CUDNN_STATUS_SUCCESS) {                                                              \
      fprintf(stderr, "cuDNN error in %s at %s:%d: %d\n", __FUNCTION__, __FILE__, __LINE__, err); \
      abort();                                                                                      \
    }                                                                                               \
  } while (0)

static cudnnHandle_t g_cudnn = nullptr;
static float* g_y_tmp = nullptr;
static size_t g_y_tmp_cap = 0;
static half* g_x_f16 = nullptr;
static size_t g_x_f16_cap = 0;

__global__ void cudnn_f32_to_f16_kernel(const float* in, half* out, int n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) out[i] = __float2half(in[i]);
}

__global__ void accum_y_kernel(float* y, const float* tmp, float beta, int d) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= d) return;
  y[i] = tmp[i] + beta * y[i];
}

static void accum_y(float* y, const float* tmp, float beta, int d, cudaStream_t stream) {
  accum_y_kernel<<<(d + 255) / 256, 256, 0, stream>>>(y, tmp, beta, d);
}

static float* y_tmp_scratch(int d) {
  size_t bytes = static_cast<size_t>(d) * sizeof(float);
  if (bytes > g_y_tmp_cap) {
    if (g_y_tmp) cudaFree(g_y_tmp);
    if (cudaMalloc(&g_y_tmp, bytes) != cudaSuccess) abort();
    g_y_tmp_cap = bytes;
  }
  return g_y_tmp;
}

static void destroy_desc(cudnnBackendDescriptor_t d) {
  if (d) cudnnBackendDestroyDescriptor(d);
}

struct CudnnPlan {
  cudnnBackendDescriptor_t plan = nullptr;
  void* workspace = nullptr;
  size_t workspace_size = 0;
  int64_t a_uid = 0;
  int64_t b_uid = 0;
  int64_t c_uid = 0;
  int n = 0;
  int d = 0;

  CudnnPlan() = default;
  CudnnPlan(CudnnPlan&& o) noexcept
    : plan(o.plan), workspace(o.workspace), workspace_size(o.workspace_size),
      a_uid(o.a_uid), b_uid(o.b_uid), c_uid(o.c_uid), n(o.n), d(o.d) {
    o.plan = nullptr;
    o.workspace = nullptr;
  }
  CudnnPlan& operator=(CudnnPlan&&) = delete;
  CudnnPlan(const CudnnPlan&) = delete;
  CudnnPlan& operator=(const CudnnPlan&) = delete;

  ~CudnnPlan() {
    if (workspace) cudaFree(workspace);
    destroy_desc(plan);
  }
};

static std::unordered_map<long long, CudnnPlan> g_cudnn_plans;

static long long cudnn_key(int n, int d) {
  return (static_cast<long long>(n) << 32) | static_cast<unsigned>(d);
}

static cudnnBackendDescriptor_t make_tensor(
  int64_t uid, cudnnDataType_t dtype, int nd, const int64_t* dim, const int64_t* stride
) {
  cudnnBackendDescriptor_t t;
  CUDNN_CHECK(cudnnBackendCreateDescriptor(CUDNN_BACKEND_TENSOR_DESCRIPTOR, &t));
  int64_t align = 16;
  CUDNN_CHECK(cudnnBackendSetAttribute(
    t, CUDNN_ATTR_TENSOR_UNIQUE_ID, CUDNN_TYPE_INT64, 1, &uid));
  CUDNN_CHECK(cudnnBackendSetAttribute(
    t, CUDNN_ATTR_TENSOR_DATA_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &dtype));
  CUDNN_CHECK(cudnnBackendSetAttribute(
    t, CUDNN_ATTR_TENSOR_DIMENSIONS, CUDNN_TYPE_INT64, nd, dim));
  CUDNN_CHECK(cudnnBackendSetAttribute(
    t, CUDNN_ATTR_TENSOR_STRIDES, CUDNN_TYPE_INT64, nd, stride));
  CUDNN_CHECK(cudnnBackendSetAttribute(
    t, CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, CUDNN_TYPE_INT64, 1, &align));
  CUDNN_CHECK(cudnnBackendFinalize(t));
  return t;
}

static bool build_cudnn_plan(int n, int d, CudnnPlan& out) {
  const int64_t a_uid = 1, b_uid = 2, c_uid = 3;
  int64_t a_dim[3] = {1, d, n};
  int64_t a_stride[3] = {static_cast<int64_t>(d) * n, n, 1};
  int64_t b_dim[3] = {1, n, 1};
  int64_t b_stride[3] = {n, 1, 1};
  int64_t c_dim[3] = {1, d, 1};
  int64_t c_stride[3] = {d, 1, 1};

  cudnnBackendDescriptor_t a = make_tensor(a_uid, CUDNN_DATA_HALF, 3, a_dim, a_stride);
  cudnnBackendDescriptor_t b = make_tensor(b_uid, CUDNN_DATA_HALF, 3, b_dim, b_stride);
  cudnnBackendDescriptor_t c = make_tensor(c_uid, CUDNN_DATA_FLOAT, 3, c_dim, c_stride);

  cudnnBackendDescriptor_t matmul_desc;
  CUDNN_CHECK(cudnnBackendCreateDescriptor(CUDNN_BACKEND_MATMUL_DESCRIPTOR, &matmul_desc));
  cudnnDataType_t comp = CUDNN_DATA_FLOAT;
  CUDNN_CHECK(cudnnBackendSetAttribute(
    matmul_desc, CUDNN_ATTR_MATMUL_COMP_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &comp));
  CUDNN_CHECK(cudnnBackendFinalize(matmul_desc));

  cudnnBackendDescriptor_t op;
  CUDNN_CHECK(cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR, &op));
  CUDNN_CHECK(cudnnBackendSetAttribute(
    op, CUDNN_ATTR_OPERATION_MATMUL_ADESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &a));
  CUDNN_CHECK(cudnnBackendSetAttribute(
    op, CUDNN_ATTR_OPERATION_MATMUL_BDESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &b));
  CUDNN_CHECK(cudnnBackendSetAttribute(
    op, CUDNN_ATTR_OPERATION_MATMUL_CDESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &c));
  CUDNN_CHECK(cudnnBackendSetAttribute(
    op, CUDNN_ATTR_OPERATION_MATMUL_DESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &matmul_desc));
  CUDNN_CHECK(cudnnBackendFinalize(op));

  cudnnBackendDescriptor_t op_graph;
  CUDNN_CHECK(cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR, &op_graph));
  CUDNN_CHECK(cudnnBackendSetAttribute(
    op_graph, CUDNN_ATTR_OPERATIONGRAPH_OPS, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &op));
  CUDNN_CHECK(cudnnBackendSetAttribute(
    op_graph, CUDNN_ATTR_OPERATIONGRAPH_HANDLE, CUDNN_TYPE_HANDLE, 1, &g_cudnn));
  CUDNN_CHECK(cudnnBackendFinalize(op_graph));

  cudnnBackendDescriptor_t heur;
  CUDNN_CHECK(cudnnBackendCreateDescriptor(CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR, &heur));
  cudnnBackendHeurMode_t mode = CUDNN_HEUR_MODE_A;
  CUDNN_CHECK(cudnnBackendSetAttribute(
    heur, CUDNN_ATTR_ENGINEHEUR_MODE, CUDNN_TYPE_HEUR_MODE, 1, &mode));
  CUDNN_CHECK(cudnnBackendSetAttribute(
    heur, CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &op_graph));
  CUDNN_CHECK(cudnnBackendFinalize(heur));

  int64_t max_results = 1;
  int64_t returned = 0;
  std::vector<cudnnBackendDescriptor_t> results(1);
  CUDNN_CHECK(cudnnBackendGetAttribute(
    heur, CUDNN_ATTR_ENGINEHEUR_RESULTS, CUDNN_TYPE_BACKEND_DESCRIPTOR, max_results, &returned, results.data()));
  destroy_desc(heur);
  destroy_desc(op_graph);
  destroy_desc(op);
  destroy_desc(matmul_desc);
  destroy_desc(a);
  destroy_desc(b);
  destroy_desc(c);

  if (returned == 0) return false;

  cudnnBackendDescriptor_t eng_cfg = results[0];
  cudnnBackendDescriptor_t plan;
  CUDNN_CHECK(cudnnBackendCreateDescriptor(CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR, &plan));
  CUDNN_CHECK(cudnnBackendSetAttribute(
    plan, CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &eng_cfg));
  cudnnStatus_t fin = cudnnBackendFinalize(plan);
  destroy_desc(eng_cfg);
  if (fin != CUDNN_STATUS_SUCCESS) {
    destroy_desc(plan);
    return false;
  }

  int64_t ws = 0;
  CUDNN_CHECK(cudnnBackendGetAttribute(
    plan, CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE, CUDNN_TYPE_INT64, 1, &returned, &ws));
  out.plan = plan;
  out.workspace_size = static_cast<size_t>(ws);
  out.a_uid = a_uid;
  out.b_uid = b_uid;
  out.c_uid = c_uid;
  out.n = n;
  out.d = d;
  if (ws) {
    if (cudaMalloc(&out.workspace, ws) != cudaSuccess) {
      destroy_desc(out.plan);
      out.plan = nullptr;
      return false;
    }
  }
  return true;
}

static CudnnPlan& cudnn_plan(int n, int d) {
  long long key = cudnn_key(n, d);
  auto it = g_cudnn_plans.find(key);
  if (it != g_cudnn_plans.end()) return it->second;

  CudnnPlan p;
  if (!build_cudnn_plan(n, d, p)) {
    fprintf(stderr, "cuDNN: no matvec plan for n=%d d=%d; using cuBLAS fallback\n", n, d);
    g_cudnn_plans.emplace(key, CudnnPlan{});
    return g_cudnn_plans.find(key)->second;
  }
  g_cudnn_plans.emplace(key, std::move(p));
  return g_cudnn_plans.find(key)->second;
}

void libgemm_cudnn_init(cudaStream_t stream) {
  if (!g_cudnn) {
    CUDNN_CHECK(cudnnCreate(&g_cudnn));
  }
  CUDNN_CHECK(cudnnSetStream(g_cudnn, stream));
}

void libgemm_cudnn_shutdown() {
  g_cudnn_plans.clear();
  if (g_x_f16) {
    cudaFree(g_x_f16);
    g_x_f16 = nullptr;
    g_x_f16_cap = 0;
  }
  if (g_y_tmp) {
    cudaFree(g_y_tmp);
    g_y_tmp = nullptr;
    g_y_tmp_cap = 0;
  }
  if (g_cudnn) {
    cudnnDestroy(g_cudnn);
    g_cudnn = nullptr;
  }
}

static half* local_x_f16_scratch(int n) {
  size_t bytes = static_cast<size_t>(n) * sizeof(half);
  if (bytes > g_x_f16_cap) {
    if (g_x_f16) cudaFree(g_x_f16);
    if (cudaMalloc(&g_x_f16, bytes) != cudaSuccess) abort();
    g_x_f16_cap = bytes;
  }
  return g_x_f16;
}

static void local_f32_to_f16(const float* in, half* out, int n, cudaStream_t stream) {
  cudnn_f32_to_f16_kernel<<<(n + 255) / 256, 256, 0, stream>>>(in, out, n);
}

void libgemm_matvec_cudnn(
  const void* w, const float* x, float* y,
  int n, int d, float alpha, float beta, cudaStream_t stream
) {
  (void)alpha;
  CudnnPlan& p = cudnn_plan(n, d);
  if (!p.plan) {
    libgemm_matvec_cublas_only(w, x, y, n, d, alpha, beta, stream);
    return;
  }

  half* x_f16 = local_x_f16_scratch(n);
  local_f32_to_f16(x, x_f16, n, stream);
  float* out = (beta == 0.f) ? y : y_tmp_scratch(d);

  int64_t uids[3] = {p.a_uid, p.b_uid, p.c_uid};
  void* ptrs[3] = {const_cast<void*>(w), x_f16, out};

  cudnnBackendDescriptor_t variant;
  CUDNN_CHECK(cudnnBackendCreateDescriptor(CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR, &variant));
  CUDNN_CHECK(cudnnBackendSetAttribute(
    variant, CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS, CUDNN_TYPE_INT64, 3, uids));
  CUDNN_CHECK(cudnnBackendSetAttribute(
    variant, CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS, CUDNN_TYPE_VOID_PTR, 3, ptrs));
  if (p.workspace) {
    CUDNN_CHECK(cudnnBackendSetAttribute(
      variant, CUDNN_ATTR_VARIANT_PACK_WORKSPACE, CUDNN_TYPE_VOID_PTR, 1, &p.workspace));
  }
  CUDNN_CHECK(cudnnBackendFinalize(variant));

  cudnnStatus_t st = cudnnBackendExecute(g_cudnn, p.plan, variant);
  destroy_desc(variant);
  if (st != CUDNN_STATUS_SUCCESS) {
    libgemm_matvec_cublas_only(w, x, y, n, d, alpha, beta, stream);
    return;
  }
  if (beta != 0.f) {
    accum_y(y, out, beta, d, stream);
  }
}

#endif
