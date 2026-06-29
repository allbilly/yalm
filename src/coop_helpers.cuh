#pragma once

#include <cuda_fp16.h>
#include <float.h>

template <typename T, int N>
union __align__(sizeof(T) * N) ablock {
  T v[N];
};

__device__ inline float warpreduce_sum(float v) {
#pragma unroll
  for (int mask = warpSize / 2; mask > 0; mask >>= 1) {
    v += __shfl_xor_sync(0xffffffff, v, mask);
  }
  return v;
}

__device__ inline float warpreduce_max(float v) {
#pragma unroll
  for (int mask = warpSize / 2; mask > 0; mask >>= 1) {
    v = max(v, __shfl_xor_sync(0xffffffff, v, mask));
  }
  return v;
}

__device__ inline int warpreduce_maxi(int v) {
#pragma unroll
  for (int mask = warpSize / 2; mask > 0; mask >>= 1) {
    v = max(v, __shfl_xor_sync(0xffffffff, v, mask));
  }
  return v;
}

__device__ inline float blocktranspose(float v, float def) {
  int lane = threadIdx.x % warpSize;
  int warp = threadIdx.x / warpSize;
  __shared__ float sm[32];
  sm[warp] = v;
  __syncthreads();
  return lane < blockDim.x / warpSize ? sm[lane] : def;
}

__device__ inline float blockreduce_sum(float v) {
  v = warpreduce_sum(v);
  v = blocktranspose(v, 0.f);
  v = warpreduce_sum(v);
  return v;
}

__device__ inline float blockreduce_max(float v) {
  v = warpreduce_max(v);
  v = blocktranspose(v, -FLT_MAX);
  v = warpreduce_max(v);
  return v;
}

__device__ inline float matmul_warppar(float* x, half* w, int i, int n) {
  int lane = threadIdx.x % warpSize;
  float val = 0.0f;
  for (int j = lane * 2; j < n; j += warpSize * 2) {
    float2 ww = __half22float2(*(half2*)&w[i * n + j]);
    float2 xx = *(float2*)&x[j];
    val += ww.x * xx.x;
    val += ww.y * xx.y;
  }
  return warpreduce_sum(val);
}
