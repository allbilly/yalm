float2 sum01 = make_float2(0.0, 0.0);
constexpr int UNROLL = 16;
half2 v01_0; float att_0; 
half2 v01_1; float att_1; 
half2 v01_2; float att_2; 
/* ... SNIP ... */
half2 v01_15; float att_15;
int t = warp_id;
for (int ctr = 0; ctr < seq_len / t_stride - UNROLL + 1; t += t_stride, ctr++) {
  int ctr_mod = ctr % UNROLL;
  if (ctr_mod == 0) {
    // prefetch every UNROLL iterations
    #define PREFETCH(j) \
      v01_##j = *((half2*)&vh[kv_stride * (t + j*t_stride) + i]); \
      att_##j = atth[t + j*t_stride];
    PREFETCH(0)
    PREFETCH(1)
    PREFETCH(2)
    /* ... SNIP ... */
    PREFETCH(15)
    #undef PREFETCH
  }
  // pull one value out of prefetch batch
  float2 v01;
  float att_t;
  switch (ctr_mod) {
    #define CASE(j) \
      case j: v01 = __half22float2(v01_##j); att_t = att_##j; break;
    CASE(0)
    CASE(1)
    CASE(2)
    /* ... SNIP ... */
    CASE(15)
    #undef CASE
  }
  // Sadly CUDA does not have float2 SIMD ops
  sum01.x += v01.x * att_t;
  sum01.y += v01.y * att_t;
}
// Handle any loop remainder that can't be unrolled
for (; t < seq_len; t += t_stride) {
  float2 v01 = __half22float2(*((half2*)&vh[kv_stride * t + i]));
  float att_t = atth[t];
  sum01.x += v01.x * att_t;
  sum01.y += v01.y * att_t;
}
// atomicAdd both `sum01` lanes when we're done