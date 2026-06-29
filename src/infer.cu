#include "model.h"
#include "libgemm.h"

#include <cuda_fp16.h>
#include "fmt/format.h"

#include <cfloat>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define FULL_MASK 0xffffffff

#define CUDA_CHECK(x)                                                                                    \
  do {                                                                                                 \
    cudaError_t err = x;                                                                             \
    if (err != cudaSuccess) {                                                                        \
      fprintf(stderr, "CUDA error in %s at %s:%d: %s (%s=%d)\n", __FUNCTION__, __FILE__, __LINE__, \
              cudaGetErrorString(err), cudaGetErrorName(err), err);                                \
      abort();                                                                                     \
    }                                                                                                \
  } while (0)

#define CUDA_CHECK2(x, msg)                                                                                    \
  do {                                                                                                 \
    cudaError_t err = x;                                                                             \
    if (err != cudaSuccess) {                                                                        \
      fprintf(stderr, "[%s] CUDA error in %s at %s:%d: %s (%s=%d)\n", msg.c_str(), __FUNCTION__, __FILE__, __LINE__, \
              cudaGetErrorString(err), cudaGetErrorName(err), err);                                \
      abort();                                                                                     \
    }                                                                                                \
  } while (0)

static void* cuda_devicecopy(void* host, size_t size) {
  void* device = NULL;
  CUDA_CHECK(cudaMalloc(&device, size));
  CUDA_CHECK(cudaMemcpyAsync(device, host, size, cudaMemcpyHostToDevice));
  return device;
}

static void* cuda_hostcopy(void* device, size_t size, std::string debug = "") {
  void* host = NULL;
  CUDA_CHECK2(cudaMallocHost(&host, size), debug);
  CUDA_CHECK2(cudaMemcpy(host, device, size, cudaMemcpyDeviceToHost), debug);
  return host;
}

[[maybe_unused]] static void* cuda_devicealloc(size_t size) {
  void* ptr = NULL;
  CUDA_CHECK(cudaMalloc(&ptr, size));
  return ptr;
}

[[maybe_unused]] static void* cuda_hostalloc(size_t size) {
  void* ptr = NULL;
  CUDA_CHECK(cudaHostAlloc(&ptr, size, 0));
  return ptr;
}

extern "C" void* upload_cuda(void* host, size_t size) {
  return cuda_devicecopy(host, size);
}

extern "C" void* download_cuda(void* device, size_t size, std::string debug) {
  return cuda_hostcopy(device, size, debug);
}

extern "C" void register_cuda_host(void* host, size_t size) {
  CUDA_CHECK(cudaHostRegister(host, size, cudaHostRegisterDefault));
}

extern "C" void free_cuda(void* device) {
  CUDA_CHECK(cudaFree(device));
}

extern "C" void unregister_cuda_host(void* host) {
  CUDA_CHECK(cudaHostUnregister(host));
}

static int warp_size = 0;
static int max_threads_per_block = 0;

extern "C" void set_cuda_device(int device) {
  CUDA_CHECK(cudaSetDevice(device));
  CUDA_CHECK(cudaDeviceGetAttribute(&warp_size, cudaDevAttrWarpSize, device));
  CUDA_CHECK(cudaDeviceGetAttribute(&max_threads_per_block, cudaDevAttrMaxThreadsPerBlock, device));
}

extern "C" void init_cuda_stream(cudaStream_t* stream) {
  CUDA_CHECK(cudaStreamCreate(stream));
}

#if DEBUG_MODEL
#include "fmt/format.h"
static std::map<std::string, DebugTensor> _debug_map;
std::map<std::string, DebugTensor>& debug_map_cuda() {
  return _debug_map;
}
template <typename T>
static std::vector<T> copy_debug_tensor(T* device, size_t numel) {
  T* host = (T*)cuda_hostcopy(device, numel * sizeof(T));
  std::vector<T> fv(host, host + numel);
  return fv;
}
template <typename T>
static void save_debug_tensor(const std::string& name, T* x, size_t size) {
  _debug_map[name] = DebugTensor(copy_debug_tensor<T>(x, size));
}
#endif

__device__ inline float blocktranspose(float v, float def) {
  // Performs block-and-warp transpose operation:
  //   For a block containing K warps where lane 0 contains val_k,
  //   this function returns:
  //   - For warp 0, lane K: val_k
  //   - For all other warps and lanes: def
  int lane = threadIdx.x % warpSize;
  int warp = threadIdx.x / warpSize;

  // Will hold results of all warps.
  // Capacity 32 since there can be at most 32 warps in a block.
  __shared__ float sm[32];
  if (lane == 0) sm[warp] = v;
  __syncthreads();

  return lane < blockDim.x / warpSize ? sm[lane] : def;
}
__device__ inline float2 blocktranspose2(float2 v, float2 def) {
  // Block-and-warp transpose for two floats per warp.
  //   For a block containing K warps where lane 0 contains (val_k.x, val_k.y),
  //   this returns:
  //   - For warp 0, lane K: (val_k.x, val_k.y)
  //   - For all other warps/lanes: def
  int lane = threadIdx.x % warpSize;
  int warp = threadIdx.x / warpSize;
  __shared__ float sm_x[32];
  __shared__ float sm_y[32];
  if (lane == 0) { sm_x[warp] = v.x; sm_y[warp] = v.y; }
  __syncthreads();
  int W = blockDim.x / warpSize;
  if (lane < W) {
    return make_float2(sm_x[lane], sm_y[lane]);
  } else {
    return def;
  }
}


__device__ 
inline float warp_reduce_sum(float val) {
  for (int offset = warpSize / 2; offset > 0; offset /= 2)
    val += __shfl_down_sync(FULL_MASK, val, offset);

  return val;
}

__device__ 
inline float warp_all_reduce_max(float val) {
  // Max reduction across a warp.
  // All threads will contain the max of all threads in the warp.
  for (int mask = warpSize/2; mask > 0; mask /= 2) {
    val = max(val, __shfl_xor_sync(FULL_MASK, val, mask));
  }
  return val;
}

__device__ 
inline float block_all_reduce_max(float val) {
  // Max reduction across a 1-D block implemented as double warp max reduction.
  // All threads will contain the max of all threads in the block.
  
  // Will hold results of all warps.
  // Capacity 32 since there can be at most 32 warps in a block.
  __shared__ float shared[32];
  const int wid  = threadIdx.x / warpSize;
  const int lane = threadIdx.x % warpSize;

  val = warp_all_reduce_max(val);

  if (blockDim.x < warpSize) return val;
  if (lane == 0) shared[wid] = val;

  __syncthreads();

  if ( wid == 0 ) {
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : -FLT_MAX;
  }
  val = warp_all_reduce_max(val);
  if (lane == 0) shared[wid] = val;
  
  __syncthreads();
  
  return shared[0];
}

__device__ 
inline float warp_all_reduce_sum(float val) {
  // Sum reduction across a warp.
  // All threads will contain the sum of all threads in the warp.
  for (int mask = warpSize/2; mask > 0; mask /= 2) {
    val += __shfl_xor_sync(FULL_MASK, val, mask);
  }
  return val;
}

__device__ 
inline float block_all_reduce_sum(float val) {
  // Sum reduction across a 1-D block implemented as double warp sum reduction.
  // All threads will contain the sum of all threads in the block.
  
  // Will hold results of all warps.
  // Capacity 32 since there can be at most 32 warps in a block.
  __shared__ float shared[32];
  const int wid  = threadIdx.x / warpSize;
  const int lane = threadIdx.x % warpSize;

  val = warp_all_reduce_sum(val);

  if (blockDim.x < warpSize) return val;
  if (lane == 0) shared[wid] = val;

  __syncthreads();

  if ( wid == 0 ) {
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0.0;
  }
  val = warp_all_reduce_sum(val);
  if (lane == 0) shared[wid] = val;
  
  __syncthreads();
  
  return shared[0];
}

__device__
inline float matmul_row(const float* row, const float* x, int offset, int dim) {
  // Vectorized: each lane reads 1 float2 (8 B) per iter; 32 lanes coalesce to 256 B.
  // lane k reads (x[2k], x[2k+1]) — adjacent pairs, sector-coalesced.
  int n2 = dim / 2;
  const float2* x2 = reinterpret_cast<const float2*>(x);
  float sum = 0.0f;
  for (int j = offset; j < n2; j += warpSize) {
    float2 xv = x2[j];
    float2 wv = reinterpret_cast<const float2*>(row)[j];
    sum += wv.x * xv.x;
    sum += wv.y * xv.y;
  }
  if ((dim & 1) && offset == 0) {
    sum += row[dim - 1] * x[dim - 1];
  }
  return warp_reduce_sum(sum);
}

__device__
inline float matmul_row(const half* row, const float* x, int offset, int dim) {
  // Vectorized: each lane reads 1 half2 (4 B) + 1 float2 (8 B) per iter.
  // 32 lanes coalesce to 128 B (row) + 256 B (x) per sector — half the loop
  // iterations and full memory transaction width vs the scalar version.
  int n2 = dim / 2;
  const half2* row2 = reinterpret_cast<const half2*>(row);
  const float2* x2 = reinterpret_cast<const float2*>(x);
  float sum = 0.0f;
  for (int j = offset; j < n2; j += warpSize) {
    half2 w = row2[j];
    float2 xv = x2[j];
    sum += __half2float(w.x) * xv.x;
    sum += __half2float(w.y) * xv.y;
  }
  if ((dim & 1) && offset == 0) {
    sum += __half2float(row[dim - 1]) * x[dim - 1];
  }
  return warp_reduce_sum(sum);
}

__device__
inline float2 matmul_row2(const half* row1, const half* row2, const float* x, int offset, int dim) {
  int n2 = dim / 2;
  const half2* row1_2 = reinterpret_cast<const half2*>(row1);
  const half2* row2_2 = reinterpret_cast<const half2*>(row2);
  const float2* x2 = reinterpret_cast<const float2*>(x);
  float sum1 = 0.f;
  float sum3 = 0.f;
  for (int j = offset; j < n2; j += warpSize) {
    float2 xv = x2[j];
    half2 w1 = row1_2[j];
    half2 w3 = row2_2[j];
    sum1 += __half2float(w1.x) * xv.x + __half2float(w1.y) * xv.y;
    sum3 += __half2float(w3.x) * xv.x + __half2float(w3.y) * xv.y;
  }
  if ((dim & 1) && offset == 0) {
    float xt = x[dim - 1];
    sum1 += __half2float(row1[dim - 1]) * xt;
    sum3 += __half2float(row2[dim - 1]) * xt;
  }
  sum1 = warp_reduce_sum(sum1);
  sum3 = warp_reduce_sum(sum3);
  return make_float2(sum1, sum3);
}

__device__
inline float2 matmul_row2(const float* row1, const float* row2, const float* x, int offset, int dim) {
  int n2 = dim / 2;
  const float2* row1_2 = reinterpret_cast<const float2*>(row1);
  const float2* row2_2 = reinterpret_cast<const float2*>(row2);
  const float2* x2 = reinterpret_cast<const float2*>(x);
  float sum1 = 0.f;
  float sum3 = 0.f;
  for (int j = offset; j < n2; j += warpSize) {
    float2 xv = x2[j];
    float2 w1 = row1_2[j];
    float2 w3 = row2_2[j];
    sum1 += w1.x * xv.x + w1.y * xv.y;
    sum3 += w3.x * xv.x + w3.y * xv.y;
  }
  if ((dim & 1) && offset == 0) {
    float xt = x[dim - 1];
    sum1 += row1[dim - 1] * xt;
    sum3 += row2[dim - 1] * xt;
  }
  sum1 = warp_reduce_sum(sum1);
  sum3 = warp_reduce_sum(sum3);
  return make_float2(sum1, sum3);
}

template <typename T>
__global__
void matmul(const T* A, const float* x, int n, int d, float* out) {
  // A (d,n) @ x (n,) -> out (d,)
  // PRECOND: Block is 1-D.
  int i = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
  if (i >= d) return;
  // Since block is 1-dimensional, thread ID is same as threadIdx.x,
  // and warp partitions thread IDs
  int offset = threadIdx.x % warpSize;
  float rowSum = matmul_row(&A[n * i], x, offset, n);
  if (offset == 0) {
    out[i] = rowSum;
  }
}

template <typename T>
__global__
void matmul_wide(const T* A, const float* x, int n, int d, float* out) {
  // A (d,n) @ x (n,) -> out (d,)
  // PRECOND: Block is 1-D and contains WPB warps.
  int i = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
  if (i >= d) return;
  // Warp j computes sum for row at <blockIdx.x*WPB + j>
  // Lane 0 of each warp will hold result
  int k = threadIdx.x % warpSize;
  float rowSum = matmul_row(&A[n * i], x, k, n);
  // Transpose values so lane k in warp 0 contains row at <blockIdx.x*WPB + k>
  // For WPB=32, this allows us to coalesce 32 float32 writes into a single 128-byte store
  rowSum = blocktranspose(rowSum, 1.0);
  if (threadIdx.x < blockDim.x / warpSize) {
    int block_start_i = blockIdx.x * blockDim.x / warpSize;
    out[block_start_i + k] = rowSum;
  }
}

template <typename T>
__global__
void fused_matmul_add_residuals(const T* A, const float* x, int n, int d, float* out) {
  int i = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
  if (i >= d) return;
  int k = threadIdx.x % warpSize;
  float rowSum = matmul_row(&A[(size_t)n * i], x, k, n);
  rowSum = blocktranspose(rowSum, 1.0);
  if (threadIdx.x < blockDim.x / warpSize) {
    int block_start_i = blockIdx.x * blockDim.x / warpSize;
    out[block_start_i + k] += rowSum;
  }
}

// w2 path: one warp/row — hb is 14336-wide; 32-warps/block re-read it 32× (L2 can't help enough).
template <typename T>
__global__
void fused_matmul_add_residuals_row(const T* A, const float* x, int n, int d, float* out) {
  int i = blockIdx.x;
  if (i >= d) return;
  int k = threadIdx.x % warpSize;
  float rowSum = matmul_row(&A[(size_t)n * i], x, k, n);
  if (k == 0) {
    out[i] += rowSum;
  }
}

__global__
void qkv_clip_kernel(
  float* q, float* k, float* v,
  int q_dim, int kv_dim, float clip
) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int total = q_dim + 2 * kv_dim;
  if (i >= total) return;
  float* p = i < q_dim ? &q[i] : (i < q_dim + kv_dim ? &k[i - q_dim] : &v[i - q_dim - kv_dim]);
  float val = *p;
  *p = val < -clip ? -clip : (val > clip ? clip : val);
}

template <typename T>
__global__
void fused_qkv_matmul_clip(
  const T* wq,      // (q_dim, dim)
  const T* wk,      // (kv_dim, dim)
  const T* wv,      // (kv_dim, dim)
  const float* x,   // (dim,)
  int dim,          // input dimension
  int q_dim,        // n_heads * head_dim
  int kv_dim,       // n_kv_heads * head_dim
  float qkv_clip,   // clipping value
  float* q_out,     // (q_dim,)
  float* k_out,     // (kv_dim,)
  float* v_out      // (kv_dim,)
) {
  int warp_in_block = threadIdx.x / warpSize;
  int warp_id = blockIdx.x * (blockDim.x / warpSize) + warp_in_block;
  int total_rows = q_dim + 2 * kv_dim;
  if (warp_id >= total_rows) return;

  // Determine which matrix (Q, K, or V) we're computing and the start row.
  const T* w;
  float* out_base;
  int out_row;
  int rows_left;
  if (warp_id < q_dim) {
    w = wq + warp_id * dim;
    out_base = q_out;
    out_row = warp_id;
    rows_left = q_dim - warp_id;
  } else if (warp_id < q_dim + kv_dim) {
    w = wk + (warp_id - q_dim) * dim;
    out_base = k_out;
    out_row = warp_id - q_dim;
    rows_left = q_dim + kv_dim - warp_id;
  } else {
    w = wv + (warp_id - q_dim - kv_dim) * dim;
    out_base = v_out;
    out_row = warp_id - q_dim - kv_dim;
    rows_left = q_dim + 2 * kv_dim - warp_id;
  }
  int k = threadIdx.x % warpSize;
  float row_sum = matmul_row(w, x, k, dim);
  // matmul_row returns the warp-reduced sum on every lane of the warp.
  // Apply clipping on lane 0 and broadcast back to all lanes via shfl.
  float clamped = (k == 0)
    ? (row_sum < -qkv_clip ? -qkv_clip : (row_sum > qkv_clip ? qkv_clip : row_sum))
    : 0.0f;
  clamped = __shfl_sync(FULL_MASK, clamped, 0);
  float t = blocktranspose(clamped, 0.0f);
  // Warp 0 lane k now holds the value from warp (block_start_warp + k).
  // Only write while the source row stays within the same output matrix.
  if (warp_in_block == 0) {
    int idx = k;
    if (idx < rows_left && idx < blockDim.x / warpSize) {
      out_base[out_row + idx] = t;
    }
  }
}

// Write RoPE-rotated K pair into calm-style 16-tiled cache.
__device__ void rope_k_cache_pair(
  const float* k, int pair_idx, int head_dim, int pos, float theta, int rotary_dim,
  half* kb, int g, int max_seq_len, int kv_pos
) {
  int j_head = pair_idx % head_dim;
  if (j_head >= head_dim - 1) return;
  float freq = j_head >= rotary_dim ? 0.f : 1.0f / powf(theta, (float)j_head / (float)rotary_dim);
  float val = pos * freq;
  float fcr, fci;
  sincosf(val, &fci, &fcr);
  float2 v01 = make_float2(k[pair_idx], k[pair_idx + 1]);
  half r0 = __float2half(v01.x * fcr - v01.y * fci);
  half r1 = __float2half(v01.x * fci + v01.y * fcr);
  half* kbase = kb + g * head_dim * max_seq_len;
  int off0 = kv_pos * 16 + max_seq_len * (j_head / 16) * 16 + (j_head % 16);
  int off1 = kv_pos * 16 + max_seq_len * ((j_head + 1) / 16) * 16 + ((j_head + 1) % 16);
  kbase[off0] = r0;
  kbase[off1] = r1;
}

__device__ float attn_dot_score(
  const half* kh, const float* query, int head_dim, int max_seq_len, int t
) {
  float score = 0.f;
  int hd16 = head_dim & ~15;
  for (int j = 0; j < hd16; j += 16) {
    const half* kp = kh + j * max_seq_len + t * 16;
    #pragma unroll
    for (int i = 0; i < 16; i += 2) {
      half2 k2 = *reinterpret_cast<const half2*>(&kp[i]);
      score += __half2float(k2.x) * query[j + i];
      score += __half2float(k2.y) * query[j + i + 1];
    }
  }
  for (int j = hd16; j < head_dim; ++j) {
    score += query[j] * __half2float(kh[(j / 16) * (16 * max_seq_len) + t * 16 + (j % 16)]);
  }
  return score;
}

__global__
void attn_dot(
  const half* kb,  // (n_kv_heads, head_dim, max_seq_len) 16-tiled
  const float* q,   // (n_heads, head_dim)
  int head_dim,
  int kv_len,
  int max_seq_len,
  int n_heads,
  int n_kv_heads,
  float* out        // (n_heads, kv_len)
) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  int h = blockIdx.y * blockDim.y + threadIdx.y;
  if (t >= kv_len || h >= n_heads) return;

  int group_size = n_heads / n_kv_heads;
  int g = h / group_size;
  const float* query = q + h * head_dim;
  const half* kh = kb + g * head_dim * max_seq_len;
  float score = attn_dot_score(kh, query, head_dim, max_seq_len, t);
  out[h * max_seq_len + t] = score / sqrtf((float)head_dim);
}

__global__
void attn_softmax(
  const float* att, 
  int seq_len, 
  int max_seq_len, 
  int n_heads, 
  float* out
) {
  int offset = threadIdx.x;
  int h = blockIdx.x;
  int block_size = blockDim.x;
  if (h >= n_heads) return;
  
  const float* atth = att + max_seq_len * h;
  float* outh = out + max_seq_len * h;
  
  float score_max = -FLT_MAX;
  for (int t = offset; t < seq_len; t += block_size) {
    if (atth[t] > score_max) {
      score_max = atth[t];
    }
  }
  score_max = block_all_reduce_max(score_max);
  float score_sum = 0.0f;
  for (int t = offset; t < seq_len; t += block_size) {
    outh[t] = expf(atth[t] - score_max);
    score_sum += outh[t];
  }
  score_sum = block_all_reduce_sum(score_sum);
  for (int t = offset; t < seq_len; t += block_size) {
    outh[t] /= score_sum;
  }
}

// Dot product of one V row (contiguous in seq_len) with attention weights; warp-parallel on t.
__device__ inline float att_mix_row(const half* vrow, const float* atth, int seq_len, int lane) {
  float sum = 0.f;
  int seq4 = seq_len & ~3;
  for (int t = lane * 4; t < seq4; t += warpSize * 4) {
    half2 v01 = *reinterpret_cast<const half2*>(&vrow[t]);
    half2 v23 = *reinterpret_cast<const half2*>(&vrow[t + 2]);
    float4 aa = *reinterpret_cast<const float4*>(&atth[t]);
    sum += __half2float(v01.x) * aa.x;
    sum += __half2float(v01.y) * aa.y;
    sum += __half2float(v23.x) * aa.z;
    sum += __half2float(v23.y) * aa.w;
  }
  for (int t = seq4 + lane; t < seq_len; t += warpSize) {
    sum += __half2float(vrow[t]) * atth[t];
  }
  return warp_all_reduce_sum(sum);
}

__global__
void att_mix(
  const half* vb,  // (n_kv_heads, head_dim, max_seq_len)
  const float* att, // (n_heads, kv_len)
  int head_dim, 
  int n_heads, 
  int n_kv_heads,
  int seq_len, 
  int max_seq_len, 
  float* out // (n_heads, head_dim)
) {
  // One block per head; warps iterate head_dim rows, lanes reduce over seq (calm-style on transposed V).
  int h = blockIdx.x;
  int group_size = n_heads / n_kv_heads;
  int g = h / group_size;
  int lane = threadIdx.x;

  const float* atth = att + max_seq_len * h;
  const half* vh = vb + g * head_dim * max_seq_len;
  float* outh = out + head_dim * h;

  for (int i = threadIdx.y; i < head_dim; i += blockDim.y) {
    float sum = att_mix_row(vh + i * max_seq_len, atth, seq_len, lane);
    if (lane == 0) {
      outh[i] = sum;
    }
  }
}

__device__ void softmax_scores(float* scores, int seq_len, int block_size) {
  int offset = threadIdx.x;
  float score_max = -FLT_MAX;
  for (int t = offset; t < seq_len; t += block_size) {
    score_max = max(score_max, scores[t]);
  }
  score_max = block_all_reduce_max(score_max);
  float score_sum = 0.f;
  for (int t = offset; t < seq_len; t += block_size) {
    scores[t] = expf(scores[t] - score_max);
    score_sum += scores[t];
  }
  score_sum = block_all_reduce_sum(score_sum);
  for (int t = offset; t < seq_len; t += block_size) {
    scores[t] /= score_sum;
  }
}

// Fused attn_dot + softmax + att_mix; scores live in shared mem (no global att traffic).
__global__
void attn_fused(
  const half* kb,
  const half* vb,
  const float* q,
  int head_dim,
  int kv_len,
  int max_seq_len,
  int n_heads,
  int n_kv_heads,
  float* xout
) {
  extern __shared__ float sm_att[];
  int h = blockIdx.x;
  if (h >= n_heads) return;

  int group_size = n_heads / n_kv_heads;
  int g = h / group_size;
  const float* query = q + h * head_dim;
  const half* kh = kb + g * head_dim * max_seq_len;
  const half* vh = vb + g * head_dim * max_seq_len;
  float* outh = xout + head_dim * h;

  int block_size = blockDim.x;
  for (int t = threadIdx.x; t < kv_len; t += block_size) {
    sm_att[t] = attn_dot_score(kh, query, head_dim, max_seq_len, t) / sqrtf((float)head_dim);
  }
  __syncthreads();

  softmax_scores(sm_att, kv_len, block_size);
  __syncthreads();

  int lane = threadIdx.x % warpSize;
  int warp_id = threadIdx.x / warpSize;
  int n_warps = blockDim.x / warpSize;
  for (int i = warp_id; i < head_dim; i += n_warps) {
    float sum = att_mix_row(vh + i * max_seq_len, sm_att, kv_len, lane);
    if (lane == 0) {
      outh[i] = sum;
    }
  }
}

__global__
void rmsnorm(const float* x, const float* weight, int size, float eps, float* out) {
  // PRECOND: only one 1-D block is launched
  float rms = 0.0;
  int offset = threadIdx.x;
  for (int i = offset; i < size; i += blockDim.x) {
    rms += x[i] * x[i];
  }
  rms = block_all_reduce_sum(rms);
  rms = sqrtf(rms / size + eps);
  float scale = 1.0 / rms;
  for (int i = offset; i < size; i += blockDim.x) {
    out[i] = x[i] * scale * weight[i];
  }
}

__device__
inline void rope(
  const float* x, int pair_idx, int head_dim, int pos, float theta, int rotary_dim, float* out
) {
  int j_head = pair_idx % head_dim;
  if (j_head < head_dim - 1) {  // Ensure we have a pair of elements
    float freq = j_head >= rotary_dim ? 0.f : 1.0f / powf(theta, (float)j_head / (float)rotary_dim);
    float val = pos * freq;
    float fcr = cosf(val);
    float fci = sinf(val);
    
    float2 v01 = *((float2*)&x[pair_idx]);
    float2 result = make_float2(
      v01.x * fcr - v01.y * fci,
      v01.x * fci + v01.y * fcr
    );
    *((float2*)&out[pair_idx]) = result;
  }
}

__device__
inline void rope(
  const float* x, int pair_idx, int head_dim, int pos, float theta, int rotary_dim, half* out
) {
  int j_head = pair_idx % head_dim;
  if (j_head < head_dim - 1) {  // Ensure we have a pair of elements
    float freq = j_head >= rotary_dim ? 0.f : 1.0f / powf(theta, (float)j_head / (float)rotary_dim);
    float val = pos * freq;
    float fcr = cosf(val);
    float fci = sinf(val);
    
    float2 v01 = *((float2*)&x[pair_idx]);
    half2 result = __floats2half2_rn(
      v01.x * fcr - v01.y * fci,
      v01.x * fci + v01.y * fcr
    );
    *((half2*)&out[pair_idx]) = result;
  }
}

__device__
inline void rope(
  const half* x, int pair_idx, int head_dim, int pos, float theta, int rotary_dim, half* out
) {
  int j_head = pair_idx % head_dim;
  if (j_head < head_dim - 1) {  // Ensure we have a pair of elements
    float freq = j_head >= rotary_dim ? 0.f : 1.0f / powf(theta, (float)j_head / (float)rotary_dim);
    float val = pos * freq;
    float fcr = cosf(val);
    float fci = sinf(val);
    
    float2 v01 = __half22float2(*((half2*)&x[pair_idx]));
    half2 result = __floats2half2_rn(
      v01.x * fcr - v01.y * fci,
      v01.x * fci + v01.y * fcr
    );
    *((half2*)&out[pair_idx]) = result;
  }
}

template <ActivationType A> __device__ inline float act(float x);
template<> __device__ inline float act<ActivationType::SILU>(float x) {
  return x / (1.0f + expf(-x));
}
template<> __device__ inline float act<ActivationType::GELU>(float x) {
  float x3 = x * x * x;
  return 0.5f * x * (1.0f + tanhf(0.797885f * (x + 0.044715f * x3)));
}

template <ActivationType A>
__global__
void ffn_glu_from_w1w3(
  const float* w1_out, const float* w3_out, int hidden_dim, float* hb
) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= hidden_dim) return;
  hb[i] = act<A>(w1_out[i]) * w3_out[i];
}

template <typename T, ActivationType A>
__global__
void fused_ffn_w1_w3_glu_act(
  const T* w1,        // (hidden_dim, dim)
  const T* w3,        // (hidden_dim, dim)
  const float* x,     // (dim,)
  int dim,
  int hidden_dim,
  float* out         // (hidden_dim,)
) {
  int warp_in_block = threadIdx.x / warpSize;
  int i = blockIdx.x * (blockDim.x / warpSize) + warp_in_block;
  if (i >= hidden_dim) return;
  int k = threadIdx.x % warpSize;
  float2 sums = matmul_row2(&w1[dim * i], &w3[dim * i], x, k, dim);
  float2 packed = blocktranspose2(sums, make_float2(0.f, 0.f));
  if (warp_in_block == 0 && k < blockDim.x / warpSize) {
    int row_idx = blockIdx.x * (blockDim.x / warpSize) + k;
    if (row_idx < hidden_dim) {
      out[row_idx] = act<A>(packed.x) * packed.y;
    }
  }
}
__global__
void copy_embedding_float(
  const float* token_embedding_table, int dim, int token, float* out
) {
  // PRECOND: grid and blocks are 1-D
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= dim) return;
  
  const float* v = token_embedding_table + dim * token;
  out[i] = v[i];
}

__global__
void copy_embedding_half(
  const half* token_embedding_table, int dim, int token, float* out
) {
  // PRECOND: grid and blocks are 1-D
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= dim) return;
  
  const half* v = token_embedding_table + dim * token;
  out[i] = __half2float(v[i]);
}

__global__
void fused_rope_and_cache_update(
  const float* q,         // (n_heads * head_dim,)
  const float* k,         // (n_kv_heads * head_dim,)
  const float* v,         // (n_kv_heads * head_dim,)
  int head_dim,          
  int n_heads,
  int n_kv_heads,
  int pos,               // current position
  int kv_pos,           // position in KV cache
  float theta,          // RoPE theta parameter
  int rotary_dim,       // how many dimensions to rotate
  int max_seq_len,      // V cache seq dimension
  float* q_out,         // (n_heads * head_dim,)
  half* kb,            // (n_kv_heads, head_dim, max_seq_len) 16-tiled
  half* vb            // (n_kv_heads, head_dim, max_seq_len)
) {
  // Each thread handles two consecutive elements (for RoPE complex rotation)
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int pair_idx = tid * 2;
  
  // Handle Q matrix RoPE
  if (pair_idx < n_heads * head_dim) {
    rope(
      q, pair_idx, head_dim, pos, 
      theta, rotary_dim, q_out
    );
  }
  
  // Handle K matrix RoPE and cache update (transposed like V)
  if (pair_idx < n_kv_heads * head_dim) {
    int g = pair_idx / head_dim;
    rope_k_cache_pair(k, pair_idx, head_dim, pos, theta, rotary_dim, kb, g, max_seq_len, kv_pos);
  }
  
  // Handle V cache update (no RoPE needed); calm-style (n_kv_heads, head_dim, seq) layout
  if (pair_idx < n_kv_heads * head_dim) {
    int g = pair_idx / head_dim;
    int j = pair_idx % head_dim;
    int base = g * head_dim * max_seq_len + j * max_seq_len + kv_pos;
    vb[base] = __float2half(v[pair_idx]);
    if (pair_idx + 1 < n_kv_heads * head_dim) {
      vb[base + max_seq_len] = __float2half(v[pair_idx + 1]);
    }
  }
}

__global__
void rotate_sink_tokens(
  half* kb, 
  int kv_sink,
  int kv_dim,
  int head_dim,
  int max_seq_len,
  float theta,
  int rotary_dim
) {
  if (kv_sink == 0) return;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int pair_idx = tid * 2;
  if (pair_idx >= kv_dim) return;

  int g = pair_idx / head_dim;
  int j = pair_idx % head_dim;
  if (j >= head_dim - 1) return;

  half* kbase = kb + g * head_dim * max_seq_len;
  for (int r = 0; r < kv_sink; r++) {
    int off0 = r * 16 + max_seq_len * (j / 16) * 16 + (j % 16);
    int off1 = r * 16 + max_seq_len * ((j + 1) / 16) * 16 + ((j + 1) % 16);
    float v0 = __half2float(kbase[off0]);
    float v1 = __half2float(kbase[off1]);
    float freq = j >= rotary_dim ? 0.f : 1.0f / powf(theta, (float)j / (float)rotary_dim);
    float fcr, fci;
    sincosf(freq, &fci, &fcr);
    kbase[off0] = __float2half(v0 * fcr - v1 * fci);
    kbase[off1] = __float2half(v0 * fci + v1 * fcr);
  }
}

template <typename T>
void Block::_block_cuda(
  InferenceState& s, int pos, int kv_sink, int kv_pos, int kv_len
) const {
#define STATIC_KERNEL(x) if (!s.graph().is_created) x;
  const Config& c = *_config;
  
  // attention pre-norm
  switch (c.norm_type) {
    case LayerNormType::RMSNorm: {
      STATIC_KERNEL((rmsnorm<<<1, max_threads_per_block, 0, s.stream()>>>(
        s.x(), rms_att_weight(), c.dim, c.norm_eps, s.xb()
      )));
      break;
    }
  }
  
  int q_dim = c.n_heads * c.head_dim;
  int kv_dim = c.n_kv_heads * c.head_dim;

  {
    // qkv matmuls for this position (coalesced writes via blocktranspose)
    // some models require clipping qkv values
    int total_rows = q_dim + 2 * kv_dim;  // Total rows across Q, K, V
    STATIC_KERNEL((fused_qkv_matmul_clip<<<
      (total_rows + 31) / 32, warp_size * 32, 0, s.stream()
    >>>(
      wq<T>(),
      wk<T>(),
      wv<T>(),
      s.xb(),
      c.dim,
      q_dim,
      kv_dim,
      c.qkv_clip,
      s.q(),
      s.k(),
      s.v()
    )));
  }
  
  // Update Q, K with RoPE relative positional encoding: 
  // complex-valued rotate q and k in each head
  // Also copy K, V to KV cache
  half* kb = (half*)key_cache();
  half* vb = (half*)value_cache();
  {
    // Calculate number of thread blocks needed
    // We need enough threads to handle the largest of:
    // - n_heads * head_dim (for Q)
    // - n_kv_heads * head_dim (for K and V)
    int max_dim = max(c.n_heads * c.head_dim, c.n_kv_heads * c.head_dim);
    int threads_needed = (max_dim + 1) / 2;  // Each thread handles 2 elements
    int num_blocks = (threads_needed + max_threads_per_block - 1) / max_threads_per_block;

    cudaKernelNodeParams params;
    params.blockDim = {static_cast<unsigned int>(max_threads_per_block), 1, 1};
    params.gridDim = {static_cast<unsigned int>(num_blocks), 1, 1};
    params.sharedMemBytes = 0;
    params.func = reinterpret_cast<void*>(fused_rope_and_cache_update);
    float* q = s.q();
    float* k = s.k();
    float* v = s.v();
    void* kernelParams[] = {
      &q,
      &k,
      &v,
      (void*)&c.head_dim,
      (void*)&c.n_heads,
      (void*)&c.n_kv_heads,
      &pos,
      &kv_pos,
      (void*)&c.rope_theta,
      (void*)&c.rotary_dim,
      (void*)&c.max_seq_len,
      &q,           // Q can be updated in-place
      &kb,
      &vb
    };
    params.kernelParams = kernelParams;
    params.extra = nullptr;
    s.graph().add_or_update_kernel_node(fmt::format("{}:fused_rope_and_cache_update", _layer_i), params, s.stream());
  }
  {
    // Sink tokens remain untouched while the rest of the KV cache is incrementally 
    // replaced in ring order, but sink i must always be positioned (max_seq_len - i)
    // away from current timestep. Hence, each forward pass, rotate any sink tokens 
    // forward by 1. See https://arxiv.org/abs/2309.17453 for more.
    int threads_needed = (kv_dim + 1) / 2;  // Each thread handles 2 elements
    int num_blocks = (threads_needed + max_threads_per_block - 1) / max_threads_per_block;
    cudaKernelNodeParams params;
    params.blockDim = {static_cast<unsigned int>(max_threads_per_block), 1, 1};
    params.gridDim = {static_cast<unsigned int>(num_blocks), 1, 1};
    params.sharedMemBytes = 0;
    params.func = reinterpret_cast<void*>(rotate_sink_tokens);
    void* kernelParams[] = {
      &kb,
      &kv_sink,
      &kv_dim,
      (void*)&c.head_dim,
      (void*)&c.max_seq_len,
      (void*)&c.rope_theta,
      (void*)&c.rotary_dim
    };
    params.kernelParams = kernelParams;
    params.extra = nullptr;
    s.graph().add_or_update_kernel_node(fmt::format("{}:rotate_sink_tokens", _layer_i), params, s.stream());
  }
  
  // multihead attention: fused dot + softmax + mix (shared-mem scores)
  {
    cudaKernelNodeParams params;
    params.blockDim = {static_cast<unsigned int>(max_threads_per_block), 1, 1};
    params.gridDim = {static_cast<unsigned int>(c.n_heads), 1, 1};
    params.sharedMemBytes = static_cast<unsigned int>(c.max_seq_len * sizeof(float));
    params.func = reinterpret_cast<void*>(attn_fused);
    float* q = s.q();
    float* xb2 = s.xb2();
    void* kernelParams[] = {
      &kb,
      &vb,
      &q,
      (void*)&c.head_dim,
      &kv_len,
      (void*)&c.max_seq_len,
      (void*)&c.n_heads,
      (void*)&c.n_kv_heads,
      &xb2
    };
    params.kernelParams = kernelParams;
    params.extra = nullptr;
    s.graph().add_or_update_kernel_node(fmt::format("{}:attn_fused", _layer_i), params, s.stream());
  }

  // final matmul projection and residual back:
  // x <- wo(...) + x
  STATIC_KERNEL((fused_matmul_add_residuals<<<c.dim/32, warp_size*32, 0, s.stream()>>>(
    wo<T>(), s.xb2(), q_dim, c.dim, s.x()
  )));
  
  // ffn pre-norm
  switch (c.norm_type) {
    case LayerNormType::RMSNorm: {
      STATIC_KERNEL((rmsnorm<<<1, max_threads_per_block, 0, s.stream()>>>(
        s.x(), rms_ffn_weight(), c.dim, c.norm_eps, s.xb()
      )));
      break;
    }
  }

  if (c.n_experts > 0) {
    assert(false && "Mixture of experts not yet supported for CUDA");
  }
  
  // mix self.w2(F.silu(self.w1(x)) * self.w3(x))
  // Note this is a feedforward with a GLU, not a simple MLP.
  // Block has 32 warps; writes are coalesced via blocktranspose2.
  switch (c.act) {
    case ActivationType::GELU: {
      STATIC_KERNEL((fused_ffn_w1_w3_glu_act<T, ActivationType::GELU><<<
        (c.hidden_dim + 31) / 32, warp_size * 32, 0, s.stream()
      >>>(
        w1<T>(), w3<T>(), s.xb(), c.dim, c.hidden_dim, s.hb()
      )));
      break;
    }
    case ActivationType::SILU: {
      STATIC_KERNEL((fused_ffn_w1_w3_glu_act<T, ActivationType::SILU><<<
        (c.hidden_dim + 31) / 32, warp_size * 32, 0, s.stream()
      >>>(
        w1<T>(), w3<T>(), s.xb(), c.dim, c.hidden_dim, s.hb()
      )));
      break;
    }
  }
  
  // add residual back: x <- w2(...) + x
  STATIC_KERNEL((fused_matmul_add_residuals_row<T><<<c.dim, warp_size, 0, s.stream()>>>(
    w2<T>(), s.hb(), c.hidden_dim, c.dim, s.x()
  )));
#undef STATIC_KERNEL
}

template <typename T>
void Block::_block_cuda_lib(
  InferenceState& s, int pos, int kv_sink, int kv_pos, int kv_len
) const {
  const Config& c = *_config;
  cudaStream_t stream = s.stream();

  switch (c.norm_type) {
    case LayerNormType::RMSNorm: {
      rmsnorm<<<1, max_threads_per_block, 0, stream>>>(
        s.x(), rms_att_weight(), c.dim, c.norm_eps, s.xb());
      break;
    }
  }

  int q_dim = c.n_heads * c.head_dim;
  int kv_dim = c.n_kv_heads * c.head_dim;

  libgemm_matvec(wq<T>(), s.xb(), s.q(), c.dim, q_dim, 1.f, 0.f, stream);
  libgemm_matvec(wk<T>(), s.xb(), s.k(), c.dim, kv_dim, 1.f, 0.f, stream);
  libgemm_matvec(wv<T>(), s.xb(), s.v(), c.dim, kv_dim, 1.f, 0.f, stream);
  qkv_clip_kernel<<<
    (q_dim + 2 * kv_dim + max_threads_per_block - 1) / max_threads_per_block,
    max_threads_per_block, 0, stream
  >>>(s.q(), s.k(), s.v(), q_dim, kv_dim, c.qkv_clip);

  half* kb = (half*)key_cache();
  half* vb = (half*)value_cache();
  {
    int max_dim = max(c.n_heads * c.head_dim, c.n_kv_heads * c.head_dim);
    int threads_needed = (max_dim + 1) / 2;
    int num_blocks = (threads_needed + max_threads_per_block - 1) / max_threads_per_block;
    fused_rope_and_cache_update<<<num_blocks, max_threads_per_block, 0, stream>>>(
      s.q(), s.k(), s.v(), c.head_dim, c.n_heads, c.n_kv_heads,
      pos, kv_pos, c.rope_theta, c.rotary_dim, c.max_seq_len,
      s.q(), kb, vb);
  }
  {
    int threads_needed = (kv_dim + 1) / 2;
    int num_blocks = (threads_needed + max_threads_per_block - 1) / max_threads_per_block;
    rotate_sink_tokens<<<num_blocks, max_threads_per_block, 0, stream>>>(
      kb, kv_sink, kv_dim, c.head_dim, c.max_seq_len, c.rope_theta, c.rotary_dim);
  }
  attn_fused<<<c.n_heads, max_threads_per_block, c.max_seq_len * sizeof(float), stream>>>(
    kb, vb, s.q(), c.head_dim, kv_len, c.max_seq_len, c.n_heads, c.n_kv_heads, s.xb2());

  libgemm_matvec(wo<T>(), s.xb2(), s.x(), q_dim, c.dim, 1.f, 1.f, stream);

  switch (c.norm_type) {
    case LayerNormType::RMSNorm: {
      rmsnorm<<<1, max_threads_per_block, 0, stream>>>(
        s.x(), rms_ffn_weight(), c.dim, c.norm_eps, s.xb());
      break;
    }
  }

  if (c.n_experts > 0) {
    assert(false && "Mixture of experts not yet supported for CUDA");
  }

  libgemm_matvec(w1<T>(), s.xb(), s.hb(), c.dim, c.hidden_dim, 1.f, 0.f, stream);
  libgemm_matvec(w3<T>(), s.xb(), s.hb2(), c.dim, c.hidden_dim, 1.f, 0.f, stream);
  switch (c.act) {
    case ActivationType::GELU: {
      ffn_glu_from_w1w3<ActivationType::GELU><<<
        (c.hidden_dim + max_threads_per_block - 1) / max_threads_per_block,
        max_threads_per_block, 0, stream
      >>>(s.hb(), s.hb2(), c.hidden_dim, s.hb());
      break;
    }
    case ActivationType::SILU: {
      ffn_glu_from_w1w3<ActivationType::SILU><<<
        (c.hidden_dim + max_threads_per_block - 1) / max_threads_per_block,
        max_threads_per_block, 0, stream
      >>>(s.hb(), s.hb2(), c.hidden_dim, s.hb());
      break;
    }
  }
  libgemm_matvec(w2<T>(), s.hb(), s.x(), c.hidden_dim, c.dim, 1.f, 1.f, stream);
}

void mha_cuda(
  float* xout,  // (n_heads, head_dim)
  float* att,   // (n_heads, max_seq_len)
  f16_t* kb,    // (n_kv_heads, head_dim, max_seq_len) 16-tiled
  f16_t* vb,    // (n_kv_heads, head_dim, max_seq_len)
  float* q,     // (n_heads, head_dim)
  int head_dim, int kv_len, int max_seq_len, int n_heads, int n_kv_heads
) {
  int max_threads_per_block = 1024;
  // all cuda uploads leak forever...
  register_cuda_host(xout, n_heads * head_dim * sizeof(float));
  kb = static_cast<f16_t*>(upload_cuda(kb, max_seq_len * n_kv_heads * head_dim * sizeof(f16_t)));
  vb = static_cast<f16_t*>(upload_cuda(vb, max_seq_len * n_kv_heads * head_dim * sizeof(f16_t)));
  q = static_cast<float*>(upload_cuda(q, n_heads * head_dim * sizeof(float)));
  attn_fused<<<n_heads, max_threads_per_block, max_seq_len * sizeof(float), cudaStreamLegacy>>>(
      (half*)kb, (half*)vb, q,
      head_dim, kv_len, max_seq_len, n_heads, n_kv_heads, xout
  );
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaGetLastError());
  unregister_cuda_host(xout);
}

template <typename T>
void matmul_cuda(float* xout, float* x, T* w, int n, int d) {
  int warp_size = 32;
  // A (d,n) @ x (n,) -> out (d,)

  // all cuda uploads leak forever...
  register_cuda_host(xout, d * sizeof(float));
  x = static_cast<float*>(upload_cuda(x, n * sizeof(float)));
  w = static_cast<T*>(upload_cuda(w, n * d * sizeof(T)));
  matmul<<<d, warp_size, 0, cudaStreamLegacy>>>(w, x, n, d, xout);
  CUDA_CHECK(cudaDeviceSynchronize()); // After this, xout contains output
  CUDA_CHECK(cudaGetLastError()); // check for kernel launch errors
  unregister_cuda_host(xout);
}

template void matmul_cuda<float>(float*, float*, float*, int, int);
template void matmul_cuda<half>(float*, float*, half*, int, int);
template<> void matmul_cuda<f16_t>(float* xout, float* x, f16_t* w, int n, int d) {
  matmul_cuda<half>(xout, x, (half*)w, n, d);
}

template <typename T>
void ffn_cuda(
  float* xout, float* x, 
  T* w1, T* w2, T* w3, 
  int hidden_dim, int dim,
  ActivationType act
) {
  int warp_size = 32;
  // all cuda uploads leak forever...
  register_cuda_host(xout, dim * sizeof(float));
  x = static_cast<float*>(upload_cuda(x, dim * sizeof(float)));
  w1 = static_cast<T*>(upload_cuda(w1, hidden_dim * dim * sizeof(T)));
  w2 = static_cast<T*>(upload_cuda(w2, dim * hidden_dim * sizeof(T)));
  w3 = static_cast<T*>(upload_cuda(w3, hidden_dim * dim * sizeof(T)));
  float* hb = new float[hidden_dim];
  float* hb2 = new float[hidden_dim];
  hb = static_cast<float*>(upload_cuda(hb, hidden_dim * sizeof(float)));
  hb2 = static_cast<float*>(upload_cuda(hb2, hidden_dim * sizeof(float)));
  // hb, hb2 leak forever on cpu too...

  // mix self.w2(F.silu(self.w1(x)) * self.w3(x))
  // Note this is a feedforward with a GLU, not a simple MLP.
  int blocks = (hidden_dim + 31) / 32;
  switch (act) {
    case ActivationType::GELU: {
      fused_ffn_w1_w3_glu_act<T, ActivationType::GELU><<<
        blocks, warp_size * 32, 0, cudaStreamLegacy
      >>>(
        w1, w3, x, dim, hidden_dim, hb
      );
      break;
    }
    case ActivationType::SILU: {
      fused_ffn_w1_w3_glu_act<T, ActivationType::SILU><<<
        blocks, warp_size * 32, 0, cudaStreamLegacy
      >>>(
        w1, w3, x, dim, hidden_dim, hb
      );
      break;
    }
  }
  
  fused_matmul_add_residuals_row<T><<< dim, warp_size, 0, cudaStreamLegacy>>>(
    w2, hb, hidden_dim, dim, xout
  );
  CUDA_CHECK(cudaStreamSynchronize(cudaStreamLegacy)); // After this, xout contains output
  CUDA_CHECK(cudaGetLastError()); // check for kernel launch errors
  unregister_cuda_host(xout);
}

template void ffn_cuda<float>(float*, float*, float*, float*, float*, int, int, ActivationType);
template void ffn_cuda<half>(float*, float*, half*, half*, half*, int, int, ActivationType);
template <> void ffn_cuda<f16_t>(
  float* xout, float* x, 
  f16_t* w1, f16_t* w2, f16_t* w3, 
  int hidden_dim, int dim,
  ActivationType act
) {
  ffn_cuda<half>(
    xout, x, 
    (half*)w1, (half*)w2, (half*)w3, 
    hidden_dim, dim, act
  );
}

template void Block::_block_cuda<float>(InferenceState&, int, int, int, int) const;
template void Block::_block_cuda<half>(InferenceState&, int, int, int, int) const;
template<> void Block::_block_cuda<f16_t>(InferenceState& s, int pos, int kv_sink, int kv_pos, int kv_len) const {
  _block_cuda<half>(s, pos, kv_sink, kv_pos, kv_len);
}
template void Block::_block_cuda_lib<half>(InferenceState&, int, int, int, int) const;
template<> void Block::_block_cuda_lib<f16_t>(InferenceState& s, int pos, int kv_sink, int kv_pos, int kv_len) const {
  _block_cuda_lib<half>(s, pos, kv_sink, kv_pos, kv_len);
}

void Model::_forward_cuda(InferenceState& s, int token, int pos, InferenceMode mode) {
  const Config& c = *config;
  s.set_mode(mode);
  CudaGraph& g = s.graph();

  // Dispatch all the kernels that comprise the work being done on the GPU 
  // for the forward pass of the model. These calls will be recorded in the 
  // InferenceState stream to form a CUDA graph, which we can save and call again,
  // which is more efficient to execute on the device than the equivalent series
  // of kernel dispatches.
  g.wrap([&]() {
    _forward_cuda_build_graph(s, token, pos, mode);
  }, s.stream());

  g.launch(s.stream());
  
  if (mode == InferenceMode::OUTPUT_LOGITS) {
    CUDA_CHECK(cudaStreamSynchronize(s.stream())); // After this, s.logits contains logits of output token
  }
  CUDA_CHECK(cudaGetLastError()); // check for kernel launch errors
}

void Model::_forward_cuda_build_graph(InferenceState& s, int token, int pos, InferenceMode mode) {
#define STATIC_KERNEL(x) if (!s.graph().is_created) x;
  const Config& c = *config;

  {
    cudaKernelNodeParams params;
    params.blockDim = {static_cast<unsigned int>(max_threads_per_block), 1, 1};
    params.gridDim = {static_cast<unsigned int>((c.dim + max_threads_per_block - 1)/max_threads_per_block), 1, 1};
    params.sharedMemBytes = 0;
    params.extra = nullptr;
    switch (c.weight_dtype) {
      case DType::F32: {
        params.func = reinterpret_cast<void*>(copy_embedding_float);
        break;
      }
      case DType::F16: {
        params.func = reinterpret_cast<void*>(copy_embedding_half);
        break;
      }
      default: {
        assert(false && "unsupported weight dtype for CUDA");
      }
    }
    float* x = s.x();
    void* kernelParams[] = {
      &token_embedding_table,
      (void*)&c.dim,
      &token,
      &x
    };
    params.kernelParams = kernelParams;

    s.graph().add_or_update_kernel_node("copy_embedding", params, s.stream());
  }
  
  // When decoding past the context length, keep the first few tokens in the KV cache
  // untouched as "attention sinks" while replacing the rest in ring order.
  // See StreamingLLM (https://arxiv.org/pdf/2309.17453) for more.
  int kv_sink = pos >= c.max_seq_len ? KV_SINKS : 0;
  int kv_pos = kv_sink + (pos - kv_sink) % (c.max_seq_len - kv_sink);
  int kv_len = pos >= c.max_seq_len ? c.max_seq_len : pos + 1;
  
  // forward all layers in order
  for (auto b : blocks) {
    b->block(s, pos, kv_sink, kv_pos, kv_len);
  }

  if (mode == InferenceMode::HYDRATE_KV_CACHE) {
    // only hydrate the KV cache and don't compute output logits
    CUDA_CHECK(cudaGetLastError()); // check for kernel launch errors
    return;
  }
  
  // final layer norm
  switch (c.norm_type) {
    case LayerNormType::RMSNorm: {
      STATIC_KERNEL((rmsnorm<<<1, max_threads_per_block, 0, s.stream()>>>(
        s.x(), rms_final_weight, c.dim, c.norm_eps, s.x()
      )));
      break;
    }
  }
  
  // classifier into logits
  switch (c.weight_dtype) {
    case DType::F32: {
      STATIC_KERNEL((matmul_wide<<<c.vocab_size/32, warp_size*32, 0, s.stream()>>>(
        static_cast<float*>(wcls), s.x(), c.dim, c.vocab_size, s.logits()
      )));
      break;
    }
    case DType::F16: {
      STATIC_KERNEL((matmul_wide<<<c.vocab_size/32, warp_size*32, 0, s.stream()>>>(
        static_cast<half*>(wcls), s.x(), c.dim, c.vocab_size, s.logits()
      )));
      break;
    }
    default: {
      assert(false && "unsupported weight dtype for CUDA");
    }
  }
#undef STATIC_KERNEL
}

void Model::_forward_cuda_lib(InferenceState& s, int token, int pos, InferenceMode mode) {
  const Config& c = *config;
  s.set_mode(mode);
  cudaStream_t stream = s.stream();

  switch (c.weight_dtype) {
    case DType::F16: {
      copy_embedding_half<<<
        (c.dim + max_threads_per_block - 1) / max_threads_per_block,
        max_threads_per_block, 0, stream
      >>>(static_cast<half*>(token_embedding_table), c.dim, token, s.x());
      break;
    }
    default: {
      assert(false && "library gemm backend supports fp16 weights only");
    }
  }

  int kv_sink = pos >= c.max_seq_len ? KV_SINKS : 0;
  int kv_pos = kv_sink + (pos - kv_sink) % (c.max_seq_len - kv_sink);
  int kv_len = pos >= c.max_seq_len ? c.max_seq_len : pos + 1;

  for (auto b : blocks) {
    b->block(s, pos, kv_sink, kv_pos, kv_len);
  }

  if (mode == InferenceMode::HYDRATE_KV_CACHE) {
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return;
  }

  switch (c.norm_type) {
    case LayerNormType::RMSNorm: {
      rmsnorm<<<1, max_threads_per_block, 0, stream>>>(
        s.x(), rms_final_weight, c.dim, c.norm_eps, s.x());
      break;
    }
  }

  libgemm_matvec(wcls, s.x(), s.logits(), c.dim, c.vocab_size, 1.f, 0.f, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaGetLastError());
}

void CudaGraph::wrap(std::function<void()> func, cudaStream_t s) {
  if (!is_created) {
    CUDA_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
    func();
    CUDA_CHECK(cudaStreamEndCapture(s, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&instance, graph, nullptr, nullptr, 0));
    is_created = true;
  } else {
    func();
  }
}

void CudaGraph::launch(cudaStream_t s) {
  CUDA_CHECK(cudaGraphLaunch(instance, s));
}

void CudaGraph::add_or_update_kernel_node(std::string key, cudaKernelNodeParams params, cudaStream_t stream) {
  if (!is_created) {
    // Get the currently capturing graph (since `graph` starts out null when recording)
    cudaStreamCaptureStatus capture_status;
    cudaGraph_t current_graph;
    const cudaGraphNode_t *deps;
    size_t dep_count;
    CUDA_CHECK(cudaStreamGetCaptureInfo_v2(stream, &capture_status, nullptr, &current_graph, &deps, &dep_count));

    // Now add a new node
    cudaGraphNode_t new_node;
    CUDA_CHECK(cudaGraphAddKernelNode(&new_node, current_graph, deps, dep_count, &params));
    nodes[key] = new_node;
    // Update the stream dependency
    CUDA_CHECK(cudaStreamUpdateCaptureDependencies(stream, &new_node, 1, 1));
  } else {
    auto it = nodes.find(key);
    if (it != nodes.end()) {
      CUDA_CHECK(cudaGraphExecKernelNodeSetParams(instance, it->second, &params));
    } else {
      assert(false && "adding new graph nodes after capture currently not supported");
    }
  }
}