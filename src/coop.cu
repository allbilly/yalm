// calm-style cooperative layer kernel backend (adapted from ref/calm/src/infer.cu)
#include "coop.h"
#include "model.h"

#include <assert.h>
#include <cooperative_groups.h>
#include <cuda_fp16.h>
#include <math.h>
#include <stdio.h>

#include "coop_helpers.cuh"

#define CUDA_CHECK(x)                                                                                    \
  do {                                                                                                   \
    cudaError_t err = (x);                                                                               \
    if (err != cudaSuccess) {                                                                            \
      fprintf(stderr, "CUDA error in %s at %s:%d: %s (%s=%d)\n", __FUNCTION__, __FILE__, __LINE__,       \
              cudaGetErrorString(err), cudaGetErrorName(err), err);                                      \
      abort();                                                                                           \
    }                                                                                                    \
  } while (0)

#define MAX_LAYERS 128

struct CoopLayer {
  float* rms_att_weight;
  half* wq;
  half* wk;
  half* wv;
  half* wo;
  float* rms_ffn_weight;
  half* w1;
  half* w2;
  half* w3;
  half* key_cache;
  half* value_cache;
};

static __constant__ CoopLayer cooplayers[MAX_LAYERS];
static cudaStream_t coop_stream = nullptr;
static int coopsms = 0;
static int coop_output_par = 1;

struct CoopArgs {
  float* x;
  float* hb;
  float* q;
  float* att;
  int n_layers;
  int dim;
  int hidden_dim;
  int head_dim;
  int n_heads;
  int n_kv_heads;
  int seq_len;
  int rotary_dim;
  int kv_len;
  int kv_pos;
  int pos;
  float norm_eps;
  float theta_log2;
  float qkv_clip;
  int act_silu;
};

__device__ inline float gelu(float x) {
  return 0.5f * x * (1.0f + tanhf(0.797885f * (x + 0.044715f * x * x * x)));
}

__device__ inline float silu(float x) {
  return x / (1.0f + expf(-x));
}

__device__ inline float4 attn_load4(half* p) {
  ablock<__half2_raw, 2> h = *(ablock<__half2_raw, 2>*)p;
  float2 h0 = __half22float2(h.v[0]);
  float2 h1 = __half22float2(h.v[1]);
  return make_float4(h0.x, h0.y, h1.x, h1.y);
}

__device__ inline float attn_score(half* kht, float* qh, int head_dim, int seq_len, int t, int off) {
  float score = 0.0f;
  for (int j = 0; j < head_dim; j += 16) {
    float4 kk = attn_load4(&kht[j * seq_len + t * 16 + off]);
    float4 qq = *(float4*)&qh[j + off];
    score += kk.x * qq.x + kk.y * qq.y + kk.z * qq.z + kk.w * qq.w;
  }
  return score;
}

__device__ inline float attn_warpdot(half* val, float* atth, int kv_len) {
  int kv_len4 = kv_len & ~3;
  int lane = threadIdx.x % warpSize;
  float res = 0.0f;
  float sum = 0.0f;
  for (int t = lane * 4; t < kv_len4; t += warpSize * 4) {
    float4 vv = attn_load4(&val[t]);
    float4 aa = *(float4*)&atth[t];
    res += vv.x * aa.x + vv.y * aa.y + vv.z * aa.z + vv.w * aa.w;
    sum += aa.x + aa.y + aa.z + aa.w;
  }
  if (kv_len4 + lane < kv_len) {
    float a = atth[kv_len4 + lane];
    res += a * __half2float(val[kv_len4 + lane]);
    sum += a;
  }
  res = warpreduce_sum(res);
  sum = warpreduce_sum(sum);
  return res / sum;
}

__device__ static void softmax(float* xout, float* x, int size) {
  int i = threadIdx.x;
  float max_val = -FLT_MAX;
  for (int j = i; j < size; j += blockDim.x) {
    max_val = max(max_val, x[j]);
  }
  max_val = blockreduce_max(max_val);
  for (int j = i; j < size; j += blockDim.x) {
    xout[j] = expf(x[j] - max_val);
  }
}

__device__ static float rmsnorm(float* o, float* x, float* weight, int size, float eps) {
  int i = threadIdx.x;
  int blockSize = blockDim.x;
  float ss = 0.0f;
  for (int j = i * 2; j < size; j += blockSize * 2) {
    float2 xx = *(float2*)&x[j];
    float2 ww = *(float2*)&weight[j];
    ss += xx.x * xx.x + xx.y * xx.y;
    *(ablock<float, 2>*)&o[j] = {xx.x * ww.x, xx.y * ww.y};
  }
  ss = blockreduce_sum(ss);
  return rsqrtf(ss / size + eps);
}

__device__ static void syncgrid() {
  volatile unsigned int* barrier = &cooperative_groups::details::get_grid_workspace()->barrier;
  if (threadIdx.x == 0) {
    unsigned int nb = 1;
    if (blockIdx.x == 0) {
      nb = 0x80000000u - (gridDim.x - 1);
    }
    unsigned int old_arrive;
    asm volatile("atom.add.release.gpu.u32 %0,[%1],%2;" : "=r"(old_arrive) : _CG_ASM_PTR_CONSTRAINT(barrier), "r"(nb) : "memory");
    unsigned int current_arrive;
    do {
      asm volatile("ld.acquire.gpu.u32 %0,[%1];" : "=r"(current_arrive) : _CG_ASM_PTR_CONSTRAINT(barrier) : "memory");
    } while (((old_arrive ^ current_arrive) & 0x80000000u) == 0);
  }
  __syncthreads();
}

__global__ static void kernel_embed(float* o, half* weight, int token, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  o[i] = __half2float(weight[token * n + i]);
}

__global__ static void kernel_rotate_sink(
  half* key_cache, int kvd, int head_dim, int kv_sink, float theta_log2, int seq_len, int rotary_dim
) {
  int i = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
  if (i >= kv_sink * kvd) return;
  int j_head = i % head_dim;
  float freq = j_head >= rotary_dim ? 0.f : exp2f(-theta_log2 * (float)j_head / (float)rotary_dim);
  float fcr, fci;
  sincosf(freq, &fci, &fcr);
  int t = i / kvd;
  int k = i % kvd;
  int o = t * 16 + seq_len * (k / 16) * 16 + (k % 16);
  float v0 = __half2float(key_cache[o + 0]);
  float v1 = __half2float(key_cache[o + 1]);
  key_cache[o + 0] = __float2half(v0 * fcr - v1 * fci);
  key_cache[o + 1] = __float2half(v0 * fci + v1 * fcr);
}

__global__ __launch_bounds__(1024, 1) static void kernel_forward(const __grid_constant__ CoopArgs args) {
  extern __shared__ char smem[];
  __shared__ float rmsscale;
  float* xs = (float*)smem;

  int dim = args.dim;
  int hidden_dim = args.hidden_dim;
  int head_dim = args.head_dim;
  int kv_mul = args.n_heads / args.n_kv_heads;
  int q_dim = args.head_dim * args.n_heads;
  int kv_dim = args.head_dim * args.n_kv_heads;

  const int IK = 4;
  int io = blockIdx.x * IK + (threadIdx.x / warpSize % IK) + gridDim.x * IK * (threadIdx.x / warpSize / IK);
  int ib = (gridDim.x * blockDim.x) / warpSize;

  static __device__ int badsoftmax = 0;

  for (int l = 0; l < args.n_layers; ++l) {
    const CoopLayer* L = &cooplayers[l];
    half* keyb = L->key_cache;
    half* valb = L->value_cache;

    if (blockIdx.x == 0 && threadIdx.x < warpSize) {
      badsoftmax = 0;
    }

    rmsscale = rmsnorm(xs, args.x, L->rms_att_weight, dim, args.norm_eps);

    for (int j = io * 2; j < q_dim + kv_dim * 2; j += ib * 2) {
      half* w = j < q_dim ? L->wq : (j < q_dim + kv_dim ? L->wk : L->wv);
      int k = j < q_dim ? j : (j < q_dim + kv_dim ? j - q_dim : j - q_dim - kv_dim);

      float v0 = matmul_warppar(xs, w, k + 0, dim) * rmsscale;
      float v1 = matmul_warppar(xs, w, k + 1, dim) * rmsscale;
      v0 = min(max(v0, -args.qkv_clip), args.qkv_clip);
      v1 = min(max(v1, -args.qkv_clip), args.qkv_clip);

      if (threadIdx.x % warpSize == 0) {
        int j_head = j % head_dim;
        float freq = j_head >= args.rotary_dim ? 0.f
          : exp2f(-args.theta_log2 * (float)j_head / (float)args.rotary_dim);
        float fcr, fci;
        sincosf(args.pos * freq, &fci, &fcr);

        if (j < q_dim) {
          args.q[k + 0] = v0 * fcr - v1 * fci;
          args.q[k + 1] = v0 * fci + v1 * fcr;
        } else if (j < q_dim + kv_dim) {
          int off = args.kv_pos * 16 + args.seq_len * (k / 16) * 16 + (k % 16);
          keyb[off + 0] = __float2half(v0 * fcr - v1 * fci);
          keyb[off + 1] = __float2half(v0 * fci + v1 * fcr);
        } else {
          valb[args.kv_pos + args.seq_len * (k + 0)] = __float2half(v0);
          valb[args.kv_pos + args.seq_len * (k + 1)] = __float2half(v1);
        }
      }
    }

    __syncthreads();
    syncgrid();

    int kv_lent = (args.kv_len + 7) / 8;
    for (int j = io; j < kv_lent * args.n_heads; j += ib) {
      int h = j % args.n_heads;
      int kvh = h / kv_mul;
      int t = (j / args.n_heads) * 8 + (threadIdx.x % warpSize) / 4;
      unsigned active = __ballot_sync(0xffffffff, t < args.kv_len);
      if (t < args.kv_len) {
        float* qh = args.q + h * head_dim;
        half* kh = keyb + kvh * head_dim * args.seq_len;
        float* atth = args.att + h * args.seq_len * 2;
        float score = attn_score(kh, qh, head_dim, args.seq_len, t, 4 * (threadIdx.x % 4));
        score += __shfl_xor_sync(active, score, 2);
        score += __shfl_xor_sync(active, score, 1);
        score /= sqrtf((float)head_dim);
        atth[t] = expf(score);
        atth[t + args.seq_len] = score;
        if (fabsf(score) > 40.f) {
          badsoftmax = 1;
        }
      }
    }

    syncgrid();

    if (badsoftmax) {
      if (blockIdx.x < args.n_heads) {
        int h = blockIdx.x;
        float* atth = args.att + h * args.seq_len * 2;
        softmax(atth, atth + args.seq_len, args.kv_len);
      }
      syncgrid();
    }

    for (int j = io; j < q_dim; j += ib) {
      int h = j / head_dim;
      int kvh = h / kv_mul;
      int j_head = j % head_dim;
      float* atth = args.att + h * args.seq_len * 2;
      half* vh = valb + kvh * head_dim * args.seq_len;
      half* val = vh + j_head * args.seq_len;
      float res = attn_warpdot(val, atth, args.kv_len);
      if (threadIdx.x % warpSize == 0) {
        args.q[j] = res;
      }
    }

    syncgrid();

    for (int j = io; j < dim; j += ib) {
      float val = matmul_warppar(args.q, L->wo, j, q_dim);
      if (threadIdx.x % warpSize == 0) {
        args.x[j] += val;
      }
    }

    __syncthreads();
    syncgrid();

    rmsscale = rmsnorm(xs, args.x, L->rms_ffn_weight, dim, args.norm_eps);

    for (int j = io; j < hidden_dim; j += ib) {
      float v1 = matmul_warppar(xs, L->w1, j, dim) * rmsscale;
      float v3 = matmul_warppar(xs, L->w3, j, dim) * rmsscale;
      float val = (args.act_silu ? silu(v1) : gelu(v1)) * v3;
      if (threadIdx.x % warpSize == 0) {
        args.hb[j] = val;
      }
    }

    syncgrid();

    for (int j = io; j < dim; j += ib) {
      float val = matmul_warppar(args.hb, L->w2, j, hidden_dim);
      if (threadIdx.x % warpSize == 0) {
        args.x[j] += val;
      }
    }

    __syncthreads();
    syncgrid();
  }
}

__global__ static void kernel_output(
  float* xout, float* x, half* w, float* rms_weight, int n, int d, float norm_eps
) {
  extern __shared__ char smem[];
  float* xs = (float*)smem;
  float rmsscale = rmsnorm(xs, x, rms_weight, n, norm_eps);
  int io = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
  int ib = (gridDim.x * blockDim.x) / warpSize;
  for (int j = io; j < d; j += ib) {
    float val = matmul_warppar(xs, w, j, n) * rmsscale;
    val = blocktranspose(val, 0.f);
    if (threadIdx.x < blockDim.x / warpSize) {
      xout[j + threadIdx.x] = val;
    }
  }
}

void yalm_coop_prepare(Model& model) {
  const Config& c = *model.config;
  assert(c.n_layers <= MAX_LAYERS);
  assert(c.n_experts == 0);
  assert(c.weight_dtype == DType::F16);

  cudaDeviceProp devprop = {};
  CUDA_CHECK(cudaGetDeviceProperties(&devprop, 0));
  assert(devprop.cooperativeLaunch);
  coopsms = devprop.multiProcessorCount;

  if (!coop_stream) {
    CUDA_CHECK(cudaStreamCreate(&coop_stream));
  }

  CoopLayer layers[MAX_LAYERS] = {};
  for (int l = 0; l < c.n_layers; ++l) {
    const Block& b = *model.blocks[l];
    layers[l].rms_att_weight = b.rms_att_weight();
    layers[l].rms_ffn_weight = b.rms_ffn_weight();
    layers[l].wq = b.wq<half>();
    layers[l].wk = b.wk<half>();
    layers[l].wv = b.wv<half>();
    layers[l].wo = b.wo<half>();
    layers[l].w1 = b.w1<half>();
    layers[l].w2 = b.w2<half>();
    layers[l].w3 = b.w3<half>();
    layers[l].key_cache = (half*)b.key_cache();
    layers[l].value_cache = (half*)b.value_cache();
  }
  CUDA_CHECK(cudaMemcpyToSymbol(cooplayers, layers, sizeof(layers)));

  int output_blk = 32 * 32;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &coop_output_par, kernel_output, output_blk, c.dim * sizeof(float)));
}

void yalm_coop_forward(Model& model, InferenceState& s, int token, int pos, InferenceMode mode) {
  const Config& c = *model.config;
  int kv_dim = c.head_dim * c.n_kv_heads;
  int kv_sink = pos >= c.max_seq_len ? KV_SINKS : 0;
  int kv_pos = kv_sink + (pos - kv_sink) % (c.max_seq_len - kv_sink);
  int kv_len = pos >= c.max_seq_len ? c.max_seq_len : pos + 1;
  float theta_log2 = log2f(c.rope_theta);

  kernel_embed<<<(c.dim + 31) / 32, 32, 0, coop_stream>>>(
    s.x(), (half*)model.token_embedding_table, token, c.dim);

  if (kv_sink > 0) {
    for (int l = 0; l < c.n_layers; ++l) {
      kernel_rotate_sink<<<(kv_sink * kv_dim + 63) / 64, 32, 0, coop_stream>>>(
        (half*)model.blocks[l]->key_cache(), kv_dim, c.head_dim, kv_sink, theta_log2, c.max_seq_len, c.rotary_dim);
    }
  }

  CoopArgs args = {
    s.x(), s.hb(), s.q(), s.att(),
    c.n_layers, c.dim, c.hidden_dim, c.head_dim, c.n_heads, c.n_kv_heads,
    c.max_seq_len, c.rotary_dim, kv_len, kv_pos, pos,
    c.norm_eps, theta_log2, c.qkv_clip,
    c.act == ActivationType::SILU ? 1 : 0,
  };
  void* argsp = &args;
  CUDA_CHECK(cudaLaunchCooperativeKernel(
    (void*)kernel_forward, coopsms, 1024, &argsp, c.dim * sizeof(float), coop_stream));

  if (mode == InferenceMode::OUTPUT_LOGITS) {
    int output_blk = 32 * 32;
    kernel_output<<<coopsms * coop_output_par, output_blk, c.dim * sizeof(float), coop_stream>>>(
      s.logits(), s.x(), (half*)model.wcls, model.rms_final_weight, c.dim, c.vocab_size, c.norm_eps);
  }

  CUDA_CHECK(cudaStreamSynchronize(coop_stream));
  CUDA_CHECK(cudaGetLastError());
}
