https://andrewkchan.dev/posts/yalm.html

─────────────────────────────────────────────────────────────────────────────────────────────┐
│                    LLM INFERENCE KERNELS: DETAILED PROFILER ANALYSIS                       │
│                              Based on YALM Blog (Mistral-7B)                                │
│                                     RTX 4090 Architecture                                   │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│ KERNEL 1: NAIVE MATMUL (1 thread per output element)                                        │
│ Source: blog.md Section 3.2                                                                 │
│ Performance: 2.9 tok/s (slower than CPU!)                                                   │
├─────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────────────────┐   │
│  │  PROBLEM: Under-utilization & Thread Waste                                           │   │
│  │                                                                                      │   │
│  │  RTX 4090 Capacity: 16,384 concurrent threads                                        │   │
│  │  Mistral-7B dim: 4096                                                                │   │
│  │                                                                                      │   │
│  │  ┌───────────────────────────────────────────────────────────────────────────────┐  │   │
│  │  │  Launch Configuration: <<< (d+1023)/1024, 1024 >>>                              │  │   │
│  │  │                                                                                 │  │   │
│  │  │  Block 0: [T0-T1023] ──────┐                                                   │  │   │
│  │  │  Block 1: [T0-T1023] ──────┤──► Only 4096 threads active!                     │  │   │
│  │  │  Block 2: [T0-T1023] ──────┤      (4096/16384 = 25% utilization)               │  │   │
│  │  │  Block 3: [T0-T1023] ──────┘                                                   │  │   │
│  │  │                                                                                 │  │   │
│  │  │  ⚠️  12,288 threads IDLE! 75% of CUDA cores wasted!                             │  │   │
│  │  └───────────────────────────────────────────────────────────────────────────────┘  │   │
│  │                                                                                      │   │
│  │  ┌───────────────────────────────────────────────────────────────────────────────┐  │   │
│  │  │  MEMORY ACCESS PATTERN                                                          │  │   │
│  │  │                                                                                 │  │   │
│  │  │  Thread i:                                                                      │  │   │
│  │  │    for (j = 0; j < n; j++) {                                                    │  │   │
│  │  │      sum += A[i*n + j] * x[j];                                                  │  │   │
│  │  │    }                                                                            │  │   │
│  │  │    out[i] = sum;   ◄── Each thread: 1 store (NON-COALESCED!)                    │  │   │
│  │  └───────────────────────────────────────────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                              │
│  🔴 BOTTLENECK: Poor Thread Utilization (25% occupancy)                                    │
│     • Only 4096 threads vs 16384 capacity                                                   │
│     • 1 thread per output = too granular                                                    │
│     • Write coalescing issues                                                               │
│                                                                                              │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
OPTIMIZATION: 1 warp per row with warp-stride loop
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│ KERNEL 2: WARP REDUCTION MATMUL                                                             │
│ Source: blog.md Section 3.2                                                                 │
│ Performance: 51.7 tok/s    17.8× speedup!                                                   │
├─────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────────────────┐   │
│  │  SOLUTION: Cooperative Warp Computing                                                │   │
│  │                                                                                      │   │
│  │  Configuration: 1 BLOCK per row, 1 WARP (32 threads) per block                       │   │
│  │                                                                                      │   │
│  │  ┌───────────────────────────────────────────────────────────────────────────────┐  │   │
│  │  │  WARP-STRIDE LOOP                                                               │  │   │
│  │  │                                                                                 │  │   │
│  │  │  Block 0 (Row 0):                                                               │  │   │
│  │  │    ┌─────┬─────┬─────┬─────┬─────┐                                             │  │   │
│  │  │    │ T0  │ T1  │ T2  │ ... │ T31 │ ◄── 1 warp (32 threads)                      │  │   │
│  │  │    └──┬──┴──┬──┴──┬──┴─────┴──┬──┘                                             │  │   │
│  │  │       │     │     │           │                                                │  │   │
│  │  │       ▼     ▼     ▼           ▼                                                │  │   │
│  │  │    A[0][0] A[0][1] A[0][2] ... A[0][31]  ──┐                                   │  │   │
│  │  │    A[0][32] A[0][33] ...        A[0][63]   ├──► stride = 32                    │  │   │
│  │  │    ...                                      │   each thread handles            │  │   │
│  │  │    A[0][n-32] ... A[0][n-1]    ◄────────────┘   n/32 elements                  │  │   │
│  │  │                                                                                 │  │   │
│  │  │  ┌─────────────────────────────────────┐                                       │  │   │
│  │  │  │  WARP REDUCTION (__shfl_down_sync)  │                                       │  │   │
│  │  │  │                                     │                                       │  │   │
│  │  │  │  Step 1: offset=16                  │                                       │  │   │
│  │  │  │    T0+=T16, T1+=T17, ... T15+=T31   │                                       │  │   │
│  │  │  │  Step 2: offset=8                   │                                       │  │   │
│  │  │  │    T0+=T8, T1+=T9, ... T7+=T15      │                                       │  │   │
│  │  │  │  Step 3: offset=4                   │                                       │  │   │
│  │  │  │    T0+=T4, T1+=T5, T2+=T6, T3+=T7   │                                       │  │   │
│  │  │  │  Step 4: offset=2                   │                                       │  │   │
│  │  │  │    T0+=T2, T1+=T3                   │                                       │  │   │
│  │  │  │  Step 5: offset=1                   │                                       │  │   │
│  │  │  │    T0+=T1  ◄── T0 has final sum!    │                                       │  │   │
│  │  │  │  Result: T0 writes to out[row]      │                                       │  │   │
│  │  │  └─────────────────────────────────────┘                                       │  │   │
│  │  └───────────────────────────────────────────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                              │
│  ✅ WINS: Better thread utilization (4096 warps active), warp cooperation                  │
│  ⚠️  STILL: Non-coalesced writes (1 warp = 1 write)                                        │
│                                                                                              │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
OPTIMIZATION: Block-level coalesced writes via shared memory transpose
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│ KERNEL 3: COALESCED WRITE MATMUL                                                            │
│ Source: blog.md Section 3.3                                                                 │
│ Performance: 56.1 tok/s    1.09× additional speedup                                         │
├─────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────────────────┐   │
│  │  PROBLEM: Non-coalesced Global Memory Stores                                         │   │
│  │                                                                                      │   │
│  │  ┌───────────────────────────────────────────────────────────────────────────────┐  │   │
│  │  │  BEFORE (32 warps write separately):                                            │  │   │
│  │  │                                                                                 │  │   │
│  │  │  Warp 0 (Block 0):  [out[0]]  ───────┐                                         │  │   │
│  │  │  Warp 1 (Block 1):  [out[1]]  ───────┤  32 separate 32-byte transactions       │  │   │
│  │  │  Warp 2 (Block 2):  [out[2]]  ───────┤  for 128 bytes of data!                 │  │   │
│  │  │     ...                              │  (4× waste!)                            │  │   │
│  │  │  Warp 31 (Block 31):[out[31]] ───────┘                                         │  │   │
│  │  │                                                                                 │  │   │
│  │  │  Total: 32 × 32B = 1024B transferred for 128B data                              │  │   │
│  │  └───────────────────────────────────────────────────────────────────────────────┘  │   │
│  │                                                                                      │   │
│  │  ┌───────────────────────────────────────────────────────────────────────────────┐  │   │
│  │  │  SOLUTION: Block-and-Warp Transpose via Shared Memory                           │  │   │
│  │  │                                                                                 │  │   │
│  │  │  New Configuration: 32 warps per block (1024 threads)                           │  │   │
│  │  │                                                                                 │  │   │
│  │  │  Block 0:                                                                         │  │   │
│  │  │    ┌─────────────────────────────────────────┐                                   │  │   │
│  │  │    │  Warp 0: computes out[0],  stores ──────┼───┐                               │  │   │
│  │  │    │  Warp 1: computes out[1],  stores ──────┼───┤                               │  │   │
│  │  │    │  Warp 2: computes out[2],  stores ──────┼───┤──► sm[32] (shared memory)     │  │   │
│  │  │    │     ...                                │   │                               │  │   │
│  │  │    │  Warp 31: computes out[31], stores ─────┼───┘                               │  │   │
│  │  │    └─────────────────────────────────────────┘                                   │  │   │
│  │  │                          │                                                      │  │   │
│  │  │    __syncthreads()       ▼                                                      │  │   │
│  │  │                          │                                                      │  │   │
│  │  │    ┌─────────────────────────────────────────┐                                   │  │   │
│  │  │    │  Warp 0 (threads 0-31):                 │                                   │  │   │
│  │  │    │    T0 reads sm[0], writes out[0]        │                                   │  │   │
│  │  │    │    T1 reads sm[1], writes out[1]        │                                   │  │   │
│  │  │    │    ...                                  │                                   │  │   │
│  │  │    │    T31 reads sm[31], writes out[31] ◄───┼── ALL CONTIGUOUS!                │  │   │
│  │  │    └─────────────────────────────────────────┘                                   │  │   │
│  │  │                                                                                 │  │   │
│  │  │  AFTER: 32 floats → 1 × 128-byte transaction ✓                                  │  │   │
│  │  │  4× memory efficiency improvement!                                              │  │   │
│  │  └───────────────────────────────────────────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────────────────┐   │
│  │  NSIGHT WARNING FIXED:                                                               │   │
│  │  "Memory access pattern for stores... only accesses 1.0 sectors"                     │   │
│  │  → Now uses full 128-byte cache lines!                                               │   │
│  └─────────────────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│ KERNEL 4: att_mix (NAIVE) - Long Context Attention                                        │
│ Source: blog.md Section 3.4                                                                 │
│ Performance: ~48 tok/s at 4k context (degraded from 56 tok/s)                               │
├─────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────────────────┐   │
│  │  PROBLEM: Memory Access Pattern Scatters Across Cache Lines                          │   │
│  │                                                                                      │   │
│  │  Tensors:                                                                            │   │
│  │    att: (n_heads, kv_len) - attention scores                                         │   │
│  │    vb:  (max_seq_len, n_kv_heads, head_dim) - value vectors                          │   │
│  │    out: (n_heads, head_dim) - output                                                 │   │
│  │                                                                                      │   │
│  │  ┌───────────────────────────────────────────────────────────────────────────────┐  │   │
│  │  │  MEMORY ACCESS VISUALIZATION                                                    │  │   │
│  │  │                                                                                 │  │   │
│  │  │  vb layout: [T0][T1][T2][T3]...[Tn]  ← contiguous in memory                     │  │   │
│  │  │              ↓  ↓  ↓  ↓      ↓                                                 │  │   │
│  │  │  Block(0,0): reads T0, then T0+stride, then T0+2*stride...                      │  │   │
│  │  │              │                                                            │     │  │   │
│  │  │              ▼                                                            ▼     │  │   │
│  │  │         ┌─────────┐                                                  ┌─────────┐│  │   │
│  │  │         │T0       │───┐                                         ┌───│T0+stride││  │   │
│  │  │         │[0][0][0]│   │  stride = n_kv_heads * head_dim         │   │[1][0][0]││  │   │
│  │  │         └─────────┘   │  = 8 * 128 = 1024 elements!              │   └─────────┘│  │   │
│  │  │                       │  = 4096 bytes!                           │              │  │   │
│  │  │                       └──────────────────────────────────────────┘              │  │   │
│  │  │                                                                                 │  │   │
│  │  │  ⚠️  Each consecutive t is 4096 bytes apart!                                    │  │   │
│  │  │     Threads in warp access DIFFERENT cache lines!                               │  │   │
│  │  │     ZERO memory coalescing!                                                     │  │   │
│  │  └───────────────────────────────────────────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────────────────┐   │
│  │  NSIGHT COMPUTE PROFILER METRICS                                                     │   │
│  ├─────────────────────────────────────────────────────────────────────────────────────┤   │
│  │  ┌──────────────────────────────┬────────────────────────────────────────────────┐  │   │
│  │  │ Metric                       │ Value                                          │  │   │
│  │  ├──────────────────────────────┼────────────────────────────────────────────────┤  │   │
│  │  │ Duration                     │ ~150 μs (same as FFN matmul!)                  │  │   │
│  │  │ Memory Throughput            │ 8.81%  ◄── TERRIBLE!                           │  │   │
│  │  │ DRAM Throughput              │ 1.68%                                          │  │   │
│  │  │ L1/TEX Cache Throughput      │ 5.17%                                          │  │   │
│  │  │ L2 Cache Throughput          │ 8.81%                                          │  │   │
│  │  │ Compute (SM) Throughput      │ 0.47%  ◄── ALMOST NO MATH!                    │  │   │
│  │  │ Elapsed Cycles               │ 5,765,045                                      │  │   │
│  │  └──────────────────────────────┴────────────────────────────────────────────────┘  │   │
│  │                                                                                      │   │
│  │  🔴 NSIGHT WARNING: "Low compute throughput and memory bandwidth utilization...      │   │
│  │     Look at Scheduler Statistics and Warp State Statistics"                         │   │
│  │                                                                                      │   │
│  │  🔴 BOTTLENECK: Strided memory access pattern                                        │   │
│  │     • vb[t][g][i] has stride of 4096 bytes between timesteps                        │   │
│  │     • Threads in same warp read from different cache lines                          │   │
│  │     • Memory latency completely dominates                                           │   │
│  │                                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
OPTIMIZATION: Partition sequence into contiguous chunks
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│ KERNEL 5: att_mix (CHUNKED) - Improved Memory Coalescing                                    │
│ Source: blog.md Section 3.4                                                                 │
│ Performance: ~57 tok/s at 4k context    1.19× speedup                                       │
├─────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────────────────┐   │
│  │  SOLUTION: Contiguous Time Chunks + atomicAdd                                        │   │
│  │                                                                                      │   │
│  │  New Grid: (n_heads, chunks) where chunks = seq_len / max_t_per_thread               │   │
│  │                                                                                      │   │
│  │  ┌───────────────────────────────────────────────────────────────────────────────┐  │   │
│  │  │  TIMELINE PARTITIONING                                                          │  │   │
│  │  │                                                                                 │  │   │
│  │  │  seq_len=4096, max_t_per_thread=256                                             │  │   │
│  │  │                                                                                 │  │   │
│  │  │  Time Axis: 0────256────512────768────...────4096                               │  │   │
│  │  │              │    │     │     │                │                                │  │   │
│  │  │              ▼    ▼     ▼     ▼                ▼                                │  │   │
│  │  │         Chunk 0 Chunk1 Chunk2 Chunk3      Chunk 15                               │  │   │
│  │  │         (Block y=0)     (Block y=2)       (Block y=15)                           │  │   │
│  │  │                                                                                 │  │   │
│  │  │  Block (h, chunk_y):                                                            │  │   │
│  │  │    t_start = chunk_y * 256                                                      │  │   │
│  │  │    t_end = t_start + 256                                                        │  │   │
│  │  │                                                                                 │  │   │
│  │  │    for t = t_start to t_end:                                                    │  │   │
│  │  │      sum += vb[t][g][i] * att[h][t]  ◄── CONTIGUOUS in t!                       │  │   │
│  │  │                                                                                 │  │   │
│  │  │    atomicAdd(&out[h][i], sum)  ◄── Multiple blocks contribute to same output    │  │   │
│  │  └───────────────────────────────────────────────────────────────────────────────┘  │   │
│  │                                                                                      │   │
│  │  ┌───────────────────────────────────────────────────────────────────────────────┐  │   │
│  │  │  MEMORY ACCESS PATTERN (IMPROVED)                                               │  │   │
│  │  │                                                                                 │  │   │
│  │  │  Block(h, 0):  t=0 to 255      ──────┐                                        │  │   │
│  │  │  Block(h, 1):  t=256 to 511    ──────┼──► Each block reads CONTIGUOUS range   │  │   │
│  │  │  Block(h, 2):  t=512 to 767    ──────┤    of vb values!                       │  │   │
│  │  │     ...                              │                                        │  │   │
│  │  │  Block(h, 15): t=3840 to 4095  ──────┘                                        │  │   │
│  │  │                                                                                 │  │   │
│  │  │  Within each block:                                                             │  │   │
│  │  │    Thread 0: vb[0][g][i], vb[0][g][i+32], ...                                  │  │   │
│  │  │    Thread 1: vb[0][g][i+1], vb[0][g][i+33], ...                                │  │   │
│  │  │       ...                                                                       │  │   │
│  │  │    Thread 31: vb[0][g][i+31], ...                                               │  │   │
│  │  │                                                                                 │  │   │
│  │  │  ✅ Threads in same warp now access CONSECUTIVE memory locations!              │  │   │
│  │  │  ✅ Memory loads are COALESCED!                                                │  │   │
│  │  └───────────────────────────────────────────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                              │
│  RESULTS: Short Context: 56 → 63 tok/s | Long Context: 48 → 57 tok/s                        │
│  att_mix time: ~150 μs → ~50 μs (matching attn kernel!)                                     │
│                                                                                              │
│  ⚠️  BUT: Quality degradation! Perplexity increases 5×!                                     │
│     Subnormal float values (1e-40) flushed to zero by atomicAdd!                            │
│                                                                                              │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
OPTIMIZATION: Accumulate in shared memory before writing to global
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│ KERNEL 6: att_mix (SHARED ACCUM) - Fixed Quality Issue                                      │
│ Source: blog.md Section 3.4                                                                 │
│ Performance: ~63.7 tok/s short, ~57 tok/s long    Quality Restored!                         │
├─────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────────────────┐   │
│  │  PROBLEM: atomicAdd to Global Memory Flushes Subnormals to Zero                      │   │
│  │                                                                                      │   │
│  │  ┌───────────────────────────────────────────────────────────────────────────────┐  │   │
│  │  │  Float Representation:                                                          │  │   │
│  │  │                                                                                 │  │   │
│  │  │  Subnormal: 0.000000000000000000000000000000000000001 (1e-40)                   │  │   │
│  │  │             ↓                                                                   │  │   │
│  │  │  atomicAdd(&global_mem, 1e-40)                                                 │  │   │
│  │  │             ↓                                                                   │  │   │
│  │  │  Result: 0.0  ◄── FLUSHED TO ZERO!  (Per forum post from 2013)                 │  │   │
│  │  │                                                                                 │  │   │
│  │  │  This happens in global memory atomics but NOT shared memory atomics!           │  │   │
│  │  └───────────────────────────────────────────────────────────────────────────────┘  │   │
│  │                                                                                      │   │
│  │  ┌───────────────────────────────────────────────────────────────────────────────┐  │   │
│  │  │  SOLUTION: 2D Blocks + Shared Memory Accumulation                               │  │   │
│  │  │                                                                                 │  │   │
│  │  │  New Block Layout: (warp_size=32, t_stride)                                     │  │   │
│  │  │    - x-dimension: 32 threads (1 warp) for head_dim elements                     │  │   │
│  │  │    - y-dimension: multiple warps for time steps                                 │  │   │
│  │  │                                                                                 │  │   │
│  │  │  ┌─────────────────────────────────────────────────────────────────────────┐   │  │   │
│  │  │  │  BLOCK (h) - 2D Layout                                                    │   │  │   │
│  │  │  │                                                                           │   │  │   │
│  │  │  │            t_stride warps                                                 │   │  │   │
│  │  │  │           (time axis)                                                     │   │  │   │
│  │  │  │                │                                                          │   │  │   │
│  │  │  │                ▼                                                          │   │  │   │
│  │  │  │         ┌───┬───┬───┬───┐                                                │   │  │   │
│  │  │  │         │W0 │W1 │W2 │Wn │  ◄── Warp IDs (0 to t_stride-1)                │   │  │   │
│  │  │  │         │   │   │   │   │                                                │   │  │   │
│  │  │  │    T0───┼───┼───┼───┼───┤                                                │   │  │   │
│  │  │  │    T1───┼───┼───┼───┼───┤                                                │   │  │   │
│  │  │  │    ...  │   │   │   │   │  ◄── Threads 0-31 (head_dim)                   │   │  │   │
│  │  │  │   T31───┴───┴───┴───┴───┘                                                │   │  │   │
│  │  │  │                                                                           │   │  │   │
│  │  │  │  __shared__ float shared[32];  ◄── One slot per warp                      │   │  │   │
│  │  │  │                                                                           │   │  │   │
│  │  │  └─────────────────────────────────────────────────────────────────────────┘   │  │   │
│  │  │                                                                                 │  │   │
│  │  │  Execution Flow:                                                                │  │   │
│  │  │  ┌─────────────────────────────────────────────────────────────────────────┐   │  │   │
│  │  │  │  for each head_dim element i:                                           │   │  │   │
│  │  │  │    Step 1: Warp 0 initializes shared[] to 0                             │   │  │   │
│  │  │  │    Step 2: Each warp computes partial sum over its time steps           │   │  │   │
│  │  │  │    Step 3: atomicAdd(&shared[threadIdx.x], sum)  ◄── SHARED MEMORY!     │   │  │   │
│  │  │  │            (preserves subnormals!)                                      │   │  │   │
│  │  │  │    Step 4: __syncthreads()                                              │   │  │   │
│  │  │  │    Step 5: Warp 0 writes shared[] to global out[h][i]                   │   │  │   │
│  │  │  │            (no atomics needed - single writer!)                         │   │  │   │
│  │  │  │    Step 6: Reset shared[] for next i                                    │   │  │   │
│  │  │  └─────────────────────────────────────────────────────────────────────────┘   │  │   │
│  │  └───────────────────────────────────────────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────────────────┐   │
│  │  RESULTS - PERPLEXITY FIXED!                                                         │   │
│  ├─────────────────────────────────────────────────────────────────────────────────────┤   │
│  │  ┌──────────────────────────────┬────────────────────────────────────────────────┐  │   │
│  │  │ Metric                       │ Value                                          │  │   │
│  │  ├──────────────────────────────┼────────────────────────────────────────────────┤  │   │
│  │  │ Short Context Throughput     │ 63.7 tok/s                                     │  │   │
│  │  │ Long Context Throughput      │ 57 tok/s                                       │  │   │
│  │  │ Perplexity Impact            │ FIXED (no degradation!)                        │  │   │
│  │  │ vs llama.cpp (short)         │ 63.7 vs 61.0 tok/s ◄── BEATS llama.cpp!       │  │   │
│  │  │ Shared Memory Used           │ 32 × 4 bytes = 128 bytes per block             │  │   │
│  │  │ Subnormal Preservation       │ ✅ Yes (shared mem atomics don't flush)        │  │   │
│  │  └──────────────────────────────┴────────────────────────────────────────────────┘  │   │
│  │                                                                                      │   │
│  │  ✅ WINS: Quality preserved, beats llama.cpp short context performance               │   │
│  │                                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│ KERNEL 7: att_mix (FP16 KV CACHE) - Compiler Auto-Vectorization Issues                      │
│ Source: blog.md Section 3.5                                                                 │
│ Naive FP16: ~53.6 tok/s (SLOWER than FP32!)                                                 │
│ Optimized: 58.8 tok/s (Expected improvement)                                                │
├─────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────────────────┐   │
│  │  PROBLEM: Compiler Heuristics Break for FP16                                        │   │
│  │                                                                                      │   │
│  │  ┌───────────────────────────────────────────────────────────────────────────────┐  │   │
│  │  │  SASS INSTRUCTION COMPARISON (from ncu UI)                                    │  │   │
│  │  │                                                                                 │  │   │
│  │  │  FP32 Kernel (FAST - 140μs):                                                    │  │   │
│  │  │  ┌─────────────────────────────────────────────────────────────────────────┐   │  │   │
│  │  │  │  LDG.E R0, [R2]      ◄── Load 1                                          │   │  │   │
│  │  │  │  LDG.E R4, [R6]      ◄── Load 2                                          │   │  │   │
│  │  │  │  LDG.E R8, [R10]     ◄── Load 3                                          │   │  │   │
│  │  │  │  LDG.E R12, [R14]    ◄── Load 4                                          │   │  │   │
│  │  │  │  LDG.E R16, [R18]    ◄── Load 5                                          │   │  │   │
│  │  │  │  LDG.E R20, [R22]    ◄── Load 6                                          │   │  │   │
│  │  │  │  LDG.E R24, [R26]    ◄── Load 7                                          │   │  │   │
│  │  │  │  LDG.E R28, [R30]    ◄── Load 8                                          │   │  │   │
│  │  │  │  FFMA R2, R0, R4, R2 ◄── Math on loads 1-2                               │   │  │   │
│  │  │  │  FFMA R2, R8, R12, R2◄── Math on loads 3-4                               │   │  │   │
│  │  │  │  FFMA R2, R16, R20, R2                                                   │   │  │   │
│  │  │  │  FFMA R2, R24, R28, R2                                                   │   │  │   │
│  │  │  │                                                                          │   │  │   │
│  │  │  │  ✅ Loop unrolled 4× (8 loads batched)                                     │   │  │   │
│  │  │  │  ✅ Loads reordered BEFORE math (prefetching!)                            │   │  │   │
│  │  │  │  ✅ Latency hidden: loads 5-8 happen while computing 1-4                  │   │  │   │
│  │  │  └─────────────────────────────────────────────────────────────────────────┘   │  │   │
│  │  │                                                                                 │  │   │
│  │  │  FP16 Kernel (SLOW - 309μs):                                                    │  │   │
│  │  │  ┌─────────────────────────────────────────────────────────────────────────┐   │  │   │
│  │  │  │  LDG.E R0, [R2]      ◄── Load 1                                          │   │  │   │
│  │  │  │  FFMA R4, R0, R8, R4  ◄── Math immediately                                │   │  │   │
│  │  │  │  LDG.E R0, [R6]      ◄── Load 2 (STALL waiting for memory!)              │   │  │   │
│  │  │  │  FFMA R4, R0, R8, R4                                                      │   │  │   │
│  │  │  │  ...                                                                      │   │  │   │
│  │  │  │                                                                          │   │  │   │
│  │  │  │  ❌ NO loop unrolling                                                       │   │  │   │
│  │  │  │  ❌ NO prefetching - load→math→load→math pattern                          │   │  │   │
│  │  │  │  ❌ Every load STALLS waiting for memory (~500 cycles)                    │   │  │   │
│  │  │  │                                                                          │   │  │   │
│  │  │  │  Throughput: 6.8% vs 25.7% for FP32                                       │   │  │   │
│  │  │  │  Time: 309μs vs 140μs                                                       │   │  │   │
│  │  │  └─────────────────────────────────────────────────────────────────────────┘   │  │   │
│  │  └───────────────────────────────────────────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                              │
│  🔴 ROOT CAUSE: Compiler heuristics don't unroll/prefetch FP16 loops!                       │
│     #pragma unroll doesn't help!                                                            │
│     Must manually implement prefetching!                                                    │
│                                                                                              │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
OPTIMIZATION: Manual prefetch with 16× unroll
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│ KERNEL 8: att_mix (FP16 + MANUAL PREFETCH)                                                  │
│ Source: blog.md Section 3.5                                                                 │
│ Performance: 58.8 tok/s long context    75μs vs 140μs (1.87× kernel speedup)                │
├─────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────────────────┐   │
│  │  SOLUTION: Software Pipelining with Manual Unroll                                    │   │
│  │                                                                                      │   │
│  │  Strategy: Unroll 16 iterations, prefetch to registers                               │   │
│  │                                                                                      │   │
│  │  ┌───────────────────────────────────────────────────────────────────────────────┐  │   │
│  │  │  CODE STRUCTURE                                                                 │  │   │
│  │  │                                                                                 │  │   │
│  │  │  // 16 registers to hold prefetched values                                      │  │   │
│  │  │  half2 v01_0;  float att_0;                                                     │  │   │
│  │  │  half2 v01_1;  float att_1;                                                     │  │   │
│  │  │  ...                                                                            │  │   │
│  │  │  half2 v01_15; float att_15;                                                    │  │   │
│  │  │                                                                                 │  │   │
│  │  │  float2 sum01 = make_float2(0.0, 0.0);                                          │  │   │
│  │  │                                                                                 │  │   │
│  │  │  for (int ctr = 0; ctr < seq_len / t_stride; ctr++) {                           │  │   │
│  │  │    int ctr_mod = ctr % 16;  // 16-way unroll                                    │  │   │
│  │  │                                                                                 │  │   │
│  │  │    if (ctr_mod == 0) {                                                          │  │   │
│  │  │      // ╔══════════════════════════════════════════════════════════════════╗   │  │   │
│  │  │      // ║  PREFETCH PHASE: Issue 16 loads at once!                         ║   │  │   │
│  │  │      // ║  These go to separate registers, hiding latency                  ║   │  │   │
│  │  │      // ╚══════════════════════════════════════════════════════════════════╝   │  │   │
│  │  │      v01_0 = *((half2*)&vh[...]);  att_0 = atth[t + 0*t_stride];              │  │   │
│  │  │      v01_1 = *((half2*)&vh[...]);  att_1 = atth[t + 1*t_stride];              │  │   │
│  │  │      ...                                                                        │  │   │
│  │  │      v01_15 = *((half2*)&vh[...]); att_15 = atth[t + 15*t_stride];             │  │   │
│  │  │    }                                                                            │  │   │
│  │  │                                                                                 │  │   │
│  │  │    // ╔════════════════════════════════════════════════════════════════════╗  │  │   │
│  │  │    // ║  USE PHASE: Consume prefetched values (already in registers!)     ║  │  │   │
│  │  │    // ╚════════════════════════════════════════════════════════════════════╝  │  │   │
│  │  │    switch (ctr_mod) {                                                           │  │   │
│  │  │      case 0:  v01 = __half22float2(v01_0);  att_t = att_0;  break;            │  │   │
│  │  │      case 1:  v01 = __half22float2(v01_1);  att_t = att_1;  break;            │  │   │
│  │  │      ...                                                                        │  │   │
│  │  │    }                                                                            │  │   │
│  │  │                                                                                 │  │   │
│  │  │    // Compute (no memory stalls!)                                               │  │   │
│  │  │    sum01.x += v01.x * att_t;                                                    │  │   │
│  │  │    sum01.y += v01.y * att_t;                                                    │  │   │
│  │  │  }                                                                            │  │   │
│  │  └───────────────────────────────────────────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────────────────┐   │
│  │  PIPELINE VISUALIZATION                                                              │   │
│  ├─────────────────────────────────────────────────────────────────────────────────────┤   │
│  │                                                                                      │   │
│  │  Timeline without prefetch (SLOW):                                                   │   │
│  │  ┌─────────────────────────────────────────────────────────────────────────────┐    │   │
│  │  │  LOAD ──► STALL (~500 cycles) ──► MATH ──► LOAD ──► STALL ──► MATH ...     │    │   │
│  │  │  ▲                                                    ▲                    │    │   │
│  │  │  └── Each load waits for previous to complete        └── Total: ~8000 cycles│    │   │
│  │  └─────────────────────────────────────────────────────────────────────────────┘    │   │
│  │                                                                                      │   │
│  │  Timeline WITH prefetch (FAST):                                                      │   │
│  │  ┌─────────────────────────────────────────────────────────────────────────────┐    │   │
│  │  │  LOAD×16 ►►►►►►►►►►►►►►►►►►►► MATH ──► MATH ──► MATH ... ──► LOAD×16 ►►►  │    │   │
│  │  │  (overlapped)                 (using prefetched)          (next batch)      │    │   │
│  │  │  ▲                                                                           │    │   │
│  │  │  └── All 16 loads issued at once, memory latency HIDDEN!                     │    │   │
│  │  │                                                                              │    │   │
│  │  │  Total: ~4000 cycles (2× faster!)                                            │    │   │
│  │  └─────────────────────────────────────────────────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────────────────┐   │
│  │  FINAL RESULTS                                                                       │   │
│  ├─────────────────────────────────────────────────────────────────────────────────────┤   │
│  │  ┌──────────────────────────────┬────────────────────────────────────────────────┐  │   │
│  │  │ Metric                       │ Value                                          │  │   │
│  │  ├──────────────────────────────┼────────────────────────────────────────────────┤  │   │
│  │  │ Toy Benchmark (ncu)          │ 140μs → 75μs (1.87× speedup)                   │  │   │
│  │  │ Long Context Throughput      │ 57.0 → 58.8 tok/s                              │  │   │
│  │  │ Short Context Throughput     │ 63.7 tok/s (unchanged)                         │  │   │
│  │  │ vs llama.cpp (long)          │ 58.7 vs 58.8 tok/s (neck-and-neck!)            │  │   │
│  │  │ vs llama.cpp (short)         │ 63.8 vs 61.0 tok/s (BEATS by 4.6%)             │  │   │
│  │  │ Memory Throughput            │ 25.7% (FP32) → Improved (FP16 + prefetch)      │  │   │
│  │  │ Unroll Factor                │ 16×                                            │  │   │
│  │  │ Prefetch Buffer              │ 16 registers per thread                        │  │   │
│  │  └──────────────────────────────┴────────────────────────────────────────────────┘  │   │
│  │                                                                                      │   │
│  │  ✅ ACHIEVEMENT: Match llama.cpp long context, beat it on short context!             │   │
│  │  ✅ WINS: FP16 memory savings + prefetching = best of both worlds!                   │   │
│  │                                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                              SUMMARY: OPTIMIZATION PROGRESSION                              │
├─────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                              │
│  Kernel                    │ Problem                    │ Solution               │ Speed   │
│  ──────────────────────────┼────────────────────────────┼────────────────────────┼─────────│
│  1. Naive Matmul          │ Under-utilization          │ Warp reduction         │ 2.9     │
│                           │ 4096/16384 threads         │ 1 warp per row         │ tok/s   │
│  ──────────────────────────┼────────────────────────────┼────────────────────────┼─────────│
│  2. Warp Reduction        │ Non-coalesced writes       │ Shared mem transpose   │ 51.7    │
│                           │ 32B transactions           │ 128B transactions      │ tok/s   │
│  ──────────────────────────┼────────────────────────────┼────────────────────────┼─────────│
│  3. Coalesced Write       │ Still room for improvement │ Kernel fusion          │ 56.1    │
│                           │ Separate kernels           │ Fuse matmul + add      │ tok/s   │
│  ──────────────────────────┼────────────────────────────┼────────────────────────┼─────────│
│  4. att_mix (naive)       │ Strided memory access      │ Contiguous chunks      │ 48      │
│                           │ 8% memory throughput       │ atomicAdd              │ tok/s   │
│  ──────────────────────────┼────────────────────────────┼────────────────────────┼─────────│
│  5. att_mix (chunked)     │ Subnormals flushed         │ Shared memory accum    │ 57      │
│                           │ Perplexity degradation     │ No global atomics      │ tok/s   │
│  ──────────────────────────┼────────────────────────────┼────────────────────────┼─────────│
│  6. att_mix (shared)      │ Quality restored           │ FP16 for memory        │ 63.7    │
│                           │ But FP16 slow              │ + manual prefetch      │ tok/s   │
│  ──────────────────────────┼────────────────────────────┼────────────────────────┼─────────│
│  7. att_mix (FP16 naive)  │ No compiler prefetch       │ 16× manual unroll      │ 53.6    │
│                           │ Load→math→load pattern     │ Register prefetch      │ tok/s   │
│  ──────────────────────────┼────────────────────────────┼────────────────────────┼─────────│
│  8. att_mix (prefetch)    │ Optimal!                   │ ─                      │ 58.8    │
│                           │                            │                        │ tok/s   │
│  ──────────────────────────┴────────────────────────────┴────────────────────────┴─────────│
│                                                                                              │
│  FINAL ACHIEVEMENT: 63.8 tok/s short, 58.8 tok/s long                                        │
│  vs llama.cpp: 61.0 tok/s short, 58.8 tok/s long                                             │
│  Beats llama.cpp on short context, matches on long!                                          │
│                                                                                              │
└─────────────────────────────────────────────────────────────────────────────────────────────┘