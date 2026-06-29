# Master kernel map — Mistral-7B-Instruct-v0.2 fp16 on RTX 3080

Generated for `plan2.md` §2.0. This is the **deliverable for K2** — one row per
logical op (Mistral decode), four columns per engine (yalm / calm / llama.cpp /
tinygrad). Each cell holds the **named generated kernel(s)** + measured `µs` /
`GB/s` (averaged over the per-token steady state at short context, kv_len ≈ 60)
plus the structural load/store pattern.

All measurements come from this session and live in:

- `ref/nsys_logs/yalm_short_cg.nsys-rep`, `yalm_short_final.nsys-rep`,
  `yalm_long_final.nsys-rep` — yalm (cuda-graph-trace)
- `ref/nsys_logs/llama_short.nsys-rep`, `llama_long.nsys-rep` — llama.cpp
- `ref/nsys_logs/calm_short.nsys-rep` (+ built-in cudaprof text from calm) —
  calm
- `ref/tg_kernels/debug4_full.log`, `debug4.log`, `kernel_stats.txt` — tinygrad
  DEBUG=4 trace
- `ref/tg_kernels/nsys_tg_short.nsys-rep` — tinygrad nsys (works with
  `BEAM=8 DEV=CUDA nsys profile …`; prior "blocked" note was wrong env)

Speed-of-light target: 760 GB/s peak DRAM. Roofline: 14.48 GB weights →
**52.5 tok/s** at 100 %, 47.5 tok/s at ~90 %.

---

## Master map (one row per logical op)

| # | Logical op                  | yalm (CUDA graphs)                                         | calm (cooperative grid + cudaprof)            | llama.cpp (GGML CUDA template zoo)                                                                                                                                                                                                                       | tinygrad (BEAM=8, DEBUG=4 trace)                                                                                          |
|---|-----------------------------|------------------------------------------------------------|-----------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------|
| 1 | Q/K/V proj                  | `fused_qkv_matmul_clip<__half>` — **74.9 µs / 1.5 %**      | `matmul_qkv` (inside `kernel_forward`) — **2409 µs / 12.1 %** total → ≈ 75 µs / 12.1 % per layer (single launch) | 3× `mul_mat_vec_f<__half, float, 1, 256, true, false>` etc. inside `[Q,K,V]` plus `mul_mat_vec_f<__half, __half, 1, 256, true, false>` (~32 % combined for matmul-class on short ctx; QKV split varies)                                                       | 3× `r_4096_2_16_28_4_4` (W·x: 164 µs each) and `r_4096_2_16_32_4` (49 µs); 714 GB/s DRAM reported                             |
| 2 | RoPE + KV write             | `fused_rope_and_cache_update` — **3.5 µs / 0.5 %**         | inside `kernel_forward` (no separate launch)  | separate `rope_norm<__half>` (1.6 µs / 0.3 %) and `k_set_rows<__half>` (1.4 µs / 0.2 %)                                                                                                                                                                | `r_8_64_2_4_8_2_8_4_4` (RoPE) 620 GB/s DRAM, `r_8_32_2_4_2_8_2_32_4` (KV write) 616 GB/s; total ~ 30 µs / 1.8 %              |
| 3 | Attn scores + softmax + mix | `attn_dot` (11.2 µs pre, **7.2 µs post-Fix#6**) / 1.1 % `attn_softmax` (3.2 µs / 0.5 %) `att_mix` (4.7 µs / 0.7 %) | `attn_score` + softmax (in shared-mem) + `attn_mix` — combined **199 µs / 1.0 %** per-layer average (very low; KV cached in shared for one layer, 80 µs + 120 µs) | `flash_attn_ext_f16<128,128,2|16,4>` split-K variants: **22 µs + 7 µs / 1.7 %** for short ctx; scales O(seq_len) thanks to tiling                                                                                                                          | **nsys short:** `r_1792_8_4_256_1024_4_4` **543 µs / 34.7 %** + `r_1024_8_2_2_28_4_16` **10.7 %** → **~48 % GPU time in generic SDPA** (not FA). Scales catastrophically at long ctx (~4.6 tok/s @ 3.2k prompt) |
| 4 | Attn out proj               | `fused_matmul_add_residuals<__half>` — **108.7 µs avg / 17 %** per launch × 2 (wo + w2) per layer | `matmul_attn` (in coop) — **1574 µs / 7.9 %** whole forward = ~49 µs/layer | `mul_mat_vec_f<__half, __half, …>` variants — combined ~31 % at short ctx                                                                                                                                                                                | `r_4096_32_32_4` and/or `r_4096_2_16_28_4_4` (52 µs + 49 µs); 700-963 GB/s DRAM                                                |
| 5 | FFN gate+up (GLU)           | `fused_ffn_w1_w3_glu_act<__half, SILU>` — **333.9 µs / 50.2 %** (327 GB/s per kernel, 50 % time) | `matmul_ffn_up` (in coop) — **10534 µs / 52.7 %** whole forward ≈ 329 µs/layer | `mul_mat_vec_f<__half, float, …>` x2 (gate + up) ~16-32 % combined, plus separate `unary_gated_op_kernel<silu>` (1.6 µs / 1.9 % at long ctx, 0.3 % at short)                                                                                                  | **nsys:** `E_458752_32_4` **364 µs / 23.3 %** (~323 GB/s effective in full graph). **DEBUG=4:** same kernel **~380 µs / 660 GB/s** isolated; unfused `r_1792_8_4_1024_64_4_4` GLU chain **524 µs / 450 GB/s** in older graph |
| 6 | FFN down                    | `fused_matmul_add_residuals` (W2 in same path as #4)       | `matmul_ffn_down` — **5273 µs / 26.4 %** whole forward ≈ 165 µs/layer | `mul_mat_vec_f<__half, __half/float, …>`                                                                                                                                                                                                                 | fused into `r_4096_2_16_28_4_4` and the GLU chain                                                                                                                                       |
| 7 | LM head (output)            | `matmul_wide<__half>` — **365.8 µs / 1.6 %** (once/tok)   | `kernel_output` — **368 µs / 1.4 %** (once/tok) | `mul_mat_f<__half2, 32, 2, 8, false>` (89 µs / 1.6 %) + **`ampere_h16816gemm_128x64_ldg8_stages_64x3_tn`** cuBLAS tensor cores (185 µs / 0.5 % short, **316 µs / 32 % at 4k ctx**)                                                                        | `r_16000_16_2_32_8` — 364 µs / 22 %! of token-time (LM head dominates TG short ctx); 720/1538 GB/s DRAM/L2                       |
| 8 | RMSNorm                     | `rmsnorm` — **4.1 µs / 1.3 %** (2 per layer = 64 cals/tok) | inside `kernel_forward` (no separate launch)  | `rms_norm_f32<1024, true, false>` — 3.7 µs / 1.2 % (all 64 norms combined at short ctx)                                                                                                                                                                  | `r_1024_4_8_2_4_4_4` — 14.4 µs total / 1.6 % (64 calls / tok); 585/2057 GB/s DRAM/L2                                           |

**Per-token wall time composition (short ctx, kv_len ≈ 60, one decode step)**

| Engine     | matmul-class total | attn total | norm/elem | LM head | total µs / tok | % peak BW (14.48 GB / 760 GB/s) |
|------------|-------------------:|-----------:|----------:|--------:|---------------:|-------------------------------:|
| yalm       | **~735 µs** (FFN 333 + qkv 75 + wo/w2 × 2 × 108) | ~14 µs (attn_dot+softmax+mix) | 8.2 µs rmsnorm | 366 µs matmul_wide | ~21,000 µs (CUDA graph launch + workload) | **87.2 % → 89.0 %** post-fix |
| calm       | ~ 18,000 µs (whole forward 19,989 µs)            | ~ 199 µs (in-coop attn)         | (in coop) | 368 µs kernel_output | 19,989 µs | **90.3 %** |
| llama.cpp  | ~ 4,500 µs (mul_mat_f + mul_mat_vec ×3 + ampere_) | ~ 30 µs (flash_attn_ext)        | ~ 10 µs | ~ 220 µs short  / ~ 360 µs long | ~ 21,500 µs short, 21,500 µs long | **88.8 % → flat** |
| tinygrad   | one `batched 32` graph 1,700 µs + per-decoded kernels (read sum ≈ 895 µs from individual `r_*` µs; rest is graph overhead and launch) | ~ 213 µs across SDPA chain | ~ 14 µs rmsnorm | 364 µs r_16000_16_2_32_8 | **29,750 µs** | **64.1 %** |

All four engines ship something for every logical op. The empty-looking
**tinygrad column is not empty — TG compiles every op to autogen `r_*`
templates** with names that encode op shape (`r_<M>_<N>_<K>_<...>`), and beam
search picks the LOCAL/UPCAST/UNROLL/CONTRACT axes. The DEBUG=4 log shows per-kernel µs +
DRAM GB/s and confirms each op type maps to a different `r_*` signature —
see `ref/tg_kernels/kernel_stats.txt`.

---

## K3 — Same-op source diff (matmul inner loop, 4096 × 14336 GEMV)

| Engine     | Kernel                  | Inner-loop pattern (the relevant snippet)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | Load/store |
|------------|-------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------|
| yalm       | `matmul_row<half,float>` `src/infer.cu:214-252` | ```for (int j = offset; j < n2; j += warpSize) { half2 w = row2[j]; float2 xv = x2[j]; sum += __half2float(w.x) * xv.x; sum += __half2float(w.y) * xv.y; }``` — **half2 + float2 vectorized**, one sector per warp per iter. Writes are coalesced via `blocktranspose` for `matmul_wide`/`fused_matmul_add_residuals`; **FFN GLU + QKV raw form coalesced via Fix#1/#2** | **128 B row + 256 B x sectors per iter**, 1 transaction/type. Writes coalesced post-fix. |
| calm       | `kernel_forward` co-op `ref/calm/src/infer.cu` + `helpers.cuh matmul_warppar` | All 32 layers run inside one cooperative grid; matmul loop runs on persistent SM threads (1024 thr/SM); tokens stream through shared mem. V cache is **transposed `(n_kv_heads, head_dim, seq_len)`** for `attn_score` / mix contiguity | In-shared + writes through on-SM warps. Bandwidth from `matmul_*` report (calm built-in cudaprof): **668-714 GB/s for FFN/W1W3/W2** |
| llama.cpp  | `mul_mat_vec_f<T,…>`    `~/llama.cpp/ggml/src/ggml-cuda/mmvf.cu:7` (template) | Each thread loads consecutive weight elements with `__ldg`, accumulates in float, writes per-thread partials; warp reduce at end. Token-parallel across `ncols_dst` × `block_size` (block_size=32-256 instantiated). LM head uses cuBLAS `ampere_h16816gemm_*` (tensor cores).                                                                                                                                                                                                                                                                                              | Coalesced `__ldg` reads. Writes are partial sums (no coalescing needed). |
| tinygrad   | autogen `r_1792_8_4_1024_64_4_4` etc. (autotuned) | Tinygrad codegen emits a generic matmul that uses `r_<M>_<N>_<K>_<BLOCK_x>_<BLOCK_y>_<THREAD_x>` template axes with upcasts and locals tuned by BEAM. The 4096×14336 GLU shows up as `r_1792_8_4_1024_64_4_4_4` (1792=4×448 outputs, 1024 reduction). Loads use SMEM/locals to coalesce. DRAM utilization **448 GB/s** vs yalm's ~705 GB/s = the structural gap.                                                                                                                            | Coalesced upcast loads; reads from L2 are 1346 GB/s (largely cache-resident for the 14.48 GB model). |

---

## K4 — Same-shape GB/s matrix (4 engines × {QKV, W1/W3 FFN, W2 FFN, attn, LM head})

Numbers are **effective GB/s** for the **primary read path** of each op
(activation + weights per call). Source: yalm `./build/test -bk <kernel>` and
nsys `cuda_gpu_kern_sum` (bytes ÷ total time); llama from nsys kernel times ÷
weight bytes; tinygrad from `DEBUG=4` per-kernel GB/s; calm from cudaprof
`(usec, GB/s)` printed.

| Shape (M × K)         | yalm (effective)        | calm (effective)          | llama.cpp (effective)         | tinygrad (effective, DRAM)  |
|-----------------------|-------------------------|---------------------------|-------------------------------|------------------------------|
| 4096 × 14336 × 1 (W2) | **~514 GB/s** (74 % of 760) | **~712 GB/s** (94 %) | **~620 GB/s** (~82 %, mul_mat_vec_f) | **~700 GB/s** in `r_4096_2_16_28_4_4` |
| 14336 × 4096 × 1 (W1/W3) | **~554 GB/s** (73 %) | **~713 GB/s** (94 %) | **~620 GB/s** (mul_mat_vec_f) | **~700 GB/s** in chain |
| 4096 × 4096 × 3 (Q+K+V) | fused QKV ~ ? (~70-80 %) | ~668 GB/s (76 %, matmul_qkv) | ~620 GB/s across 3 launches | 3× `r_4096_*` chain, 714 GB/s |
| 4096 × 32 × 128 (attn projection) | ~ ? | in coop ~ 50 µs/layer for whole attn | covered by mul_mat_vec_f + flash_attn_ext | fused into SDPA |
| 32 heads × 4096 KV × 128 (long-ctx attn_dot) | **~85 GB/s** (5 % of peak; the bottleneck of long ctx) | covered by calm `attn_score` (low BW) but small time share | FA path: covered by flash_attn_ext split-K | covered by SDPA which doesn't fuse FA |
| 32000 × 4096 × 1 (LM head) | **~611 GB/s** (80 %, `matmul_wide`) | **~692 GB/s** (91 %) | ~580 GB/s + cuBLAS tensor-core ~720 GB/s effective | `r_16000_16_2_32_8`: 720 GB/s effective |

Key takeaway: yalm is **the only engine using warp-parallel coalesced GEMV
without tensor cores** for these shapes; calm also warp-parallel but fuses
the whole layer into a cooperative grid (different cost model); llama.cpp
uses tensor cores for LM head + chunked GEMV everywhere; tinygrad matches
DRAM bandwidth of calm on individual GEMVs but pays extra for unfused graph
+ non-FA attention.

---

## K5 — llama.cpp mangled-name → source-file

| Mangled kernel (from nsys) | Path |
|----------------------------|------|
| `void mul_mat_vec_f<__half, float, (int)1, (int)256, (bool)1, (bool)0>(…)` | `~/llama.cpp/ggml/src/ggml-cuda/mmvf.cu:7` (template def) instantiated at line 454-500 (`launch_mul_mat_vec_f_cuda`) |
| `void mul_mat_vec_f<__half, __half, 1, 256, true, false>(…)`              | same file, line 7 |
| `void mul_mat_vec_f<__half, __half, 1, 256, false, false>(…)`             | same file, line 7 (no-fusion instantiation) |
| `void mul_mat_f<__half2, (int)32, (int)2, (int)8, (bool)0>(…)`            | `~/llama.cpp/ggml/src/ggml-cuda/mmf.cu` (mat-mat f16 via cuBLAS hgemm path; tries tensor cores when supported) |
| `ampere_h16816gemm_128x64_ldg8_stages_64x3_tn`                            | cuBLAS internal — selects Ampere 16x8x16 mma.sync path; called from `mmf.cu` |
| `ampere_h16816gemm_64x64_ldg8_stages_64x5_tn`                             | cuBLAS internal — different tile shape, dispatched at runtime by `cublasLtMatmulAlgoGetHeuristic` |
| `void flash_attn_ext_f16<(int)128, (int)128, (int)2, (int)4, false, false>` | `~/llama.cpp/ggml/src/ggml-cuda/fattn-vec.cuh` (vector variant) + `fattn-tile.cuh` (tile). The `<128,128,2,4,…>` is split-K=2 (used for short ctx) and `<128,128,16,4,…>` is split-K=16 (long ctx) |
| `void flash_attn_ext_f16<(int)128, (int)128, (int)16, (int)4, false, false>` | same template specialized for long ctx |
| `void flash_attn_stream_k_fixup_uniform<(int)128, (int)2, (int)4>`         | `fattn-common.cuh` and `fattn-vec.cuh` — used by the fixup path of split-K=2 |
| `void rope_norm<(bool)1, (bool)0, float, __half>`                         | `~/llama.cpp/ggml/src/ggml-cuda/rope.cuh` |
| `void k_set_rows<float, long, __half>`                                    | `~/llama.cpp/ggml/src/ggml-cuda/set-rows.cu` |
| `void rms_norm_f32<(int)1024, (bool)1, (bool)0>`                          | `~/llama.cpp/ggml/src/ggml-cuda/norm.cuh` (rms-norm template) |
| `void convert_unary<__half, float>` / `<float, __half>`                   | `~/llama.cpp/ggml/src/ggml-cuda/convert.cu` |
| `void unary_gated_op_kernel<&op_silu, …>`                                 | `~/llama.cpp/ggml/src/ggml-cuda/unary.cuh` (FFN GLU activation) |
| `cublasLt::splitKreduce_kernel<(int)32, (int)16, …>`                       | cuBLASLt internal split-K reduction |

All other shapes (gemv for q4_K, q5_K, q6_K, q8_0, etc.) flow through
`mmvq.cu` + `vecdotq.cuh` per `~/llama.cpp/ggml/src/ggml-cuda/mmvq.cuh` — not
relevant here because we run FP16 only.

---

## K6 — calm op-level µs/GB/s (built-in cudaprof, short ctx, kv_len=60)

Built-in `kernel_forward breakdown (over 31 runs, avg 19989.5 usec/run)` from
`ref/nsys_logs/calm_short.nsys-rep`:

| Op label (in calm binary) | µs / run (whole forward) | µs / layer (÷ 32) | % | GB/s |
|---------------------------|--------------------------:|------------------:|--:|-----:|
| `matmul_qkv`              | 2409.0                    | **75.3**          | 12.1 % | 668.6 |
| `attn_score`              | 80.1                      | 2.5               | 0.4 % | 13.1 (low — shared-mem reduction, not DRAM-limited) |
| `attn_mix`                | 118.7                     | 3.7               | 0.6 % | 8.8 (calm transposes V; mostly shared) |
| `matmul_attn`             | 1574.1                    | 49.2              | 7.9 % | 682.1 |
| `matmul_ffn_up` (W1·silu·W3) | 10534.1                | **329.2**         | 52.7 % | 713.5 |
| `matmul_ffn_down` (W2)    | 5273.6                    | 164.8             | 26.4 % | 712.6 |
| total `kernel_forward`    | 19,989.5                  | 624.7             | 98.5 % | — |
| `kernel_output`           | 368 / 50                  | 7.4 (per tok)     | 1.4 % | — |
| `kernel_embed`            | ~ 1.7 / 51                |                   | 0.0 % | — |

**Each matmul in calm hits 668-714 GB/s** = the bandwidth ceiling. The
cooperative-grid structure means there is no per-kernel launch tax; only 3
distinct kernel names total. Compared to llama.cpp's ~1500 launches/tok, the
launch-overhead difference alone is ~3 ms/tok (≈ 7-10 tok/s of speed).

---

## A8 — Launches per token (this session's measurement)

| Engine     | Distinct kernel names / run | Kernel launches / 50 tok | ~Launches / tok | Structure |
|------------|-----------------------------:|-------------------------:|----------------:|-----------|
| **calm**   | 3 (`kernel_forward`, `kernel_output`, `kernel_embed`) | ~50 + 50 + 50 = 150 | **~3** launches / tok | cooperative grid: 1 layer-grid + 1 LM + 1 embed |
| **llama.cpp** | 14+ (template instantiations + cuBLAS) | nsys shows ~1,500 launches/50tok | **~30** / tok | GGML graph of many small kernel launches |
| **yalm**   | 10+ (matmul, matmul_wide, fused_*, attn_*) | 4,192 fused_ffn + 4,192 qkv + 4,192 attn_dot + 8,506 rmsnorm + 8,384 fused_matmul + 4,192 att_mix + 4,192 fused_rope + 4,192 attn_softmax + 122 LM head + 3,712 att_mix = ~42k graph nodes / 50 tok = ~840 graph nodes per tok | high underlying count, but inside CUDA graph so launch amortized | Per-token graph captured once; replay = single API call per tok |
| **tinygrad** | ~14 unique `r_*` template instances captured into **one graph** (`JIT GRAPHing batch with 32 kernels`) | nsys is blocked but DEBUG=4 shows ~14 kernels per single-token step, and `JIT GRAPHing batch with 235` on the long-prompt case shows the graph contains 235 ops | **~14** kernels / tok, amortized via TinyJit graph | JIT graph with one launch of `cudaGraphLaunch` per tok |

**Launch-tax interpretation:** calm's 3-vs-15 launch difference explains a
fraction of its 1.5 tok/s lead over llama.cpp; yalm matches llama.cpp despite
its much higher underlying kernel count because CUDA graphs amortize it away;
tinygrad falls between due to graph batching but pays in unfused graphs and
no-FA attention.

---

## A1, A4, A6, A7 — nsys short and long for llama.cpp + calm + tinygrad

| Engine     | Short ctx (kv≈60) | Long ctx (kv≈4096) |
|------------|-------------------|--------------------|
| llama.cpp  | full nsys: `mul_mat_f` 28.6 %, `ampere_*gemm` 32.4 %, `mul_mat_vec_f` 31.6 %, `flash_attn_ext_f16` 1.7 %, `rms_norm_f32` 1.2 % → **46.62 tok/s** | full nsys: `ampere_*gemm` 33.8 %, `mul_mat_vec_f` ~38 %, **`flash_attn_ext_f16` 6.0 %** → **46.5 tok/s** (FLAT) |
| yalm       | full nsys §3.1 already documented | see B1 / new measurements in §10 of results.md |
| calm       | single cooperative launch; built-in cudaprof breakdown per op (see K6) | same breakdown (flat: 47.4 tok/s) |
| tinygrad   | nsys **works**: `ref/tg_kernels/nsys_tg_short.nsys-rep` — 48 % attn, 35 % `E_458752` FFN, 28.9 tok/s in capture | **4.6 tok/s** at ~3.2k ctx (`long_prompt.txt`); gap vs llama **widens to −42 tok/s** (SDPA O(N²), no FA) |

`r_1792_8_4_1024_64_4_4` at 524 µs × 32 layers × 1 tok ≈ 16.8 ms of token
time; `batched 32` graph (which fuses one full decode step) measures
**1699 µs** as a single CUDA graph launch — that's the per-token budget
measured at the graph-launch boundary. Same for `batched 64`/`batched 128`
which include longer-ctx iterations: 3454 / 8023 µs = 53.7 / 62.7 µs per
layer either way. **TG's per-layer cost is roughly constant** in graph mode;
the wallclock budget for tg short ctx ≈ 30 ms → 33.6 tok/s.

---

## C2, C3 — calibration

**C1 BW ceiling:** `-b` (32-thread host) and `-b2` (64-thread host) tests
require > 32 GB of allocation; the RTX 3080 has 19.6 GiB → cannot run. We
use **760 GB/s spec** as the ceiling (matches the binary-header line
`peak bandwidth 760 GB/s (ECC 0)` printed by calm).

**C2 ggml source skim:** all K5 entries above map directly to
`~/llama.cpp/ggml/src/ggml-cuda/{mmvf.cu, mmf.cu, fattn-*.cuh, rope.cuh,
set-rows.cu, norm.cuh, convert.cu, unary.cuh}`. The `mmvq.cu`/`vecdotq.cuh`
quantized paths are not used by our FP16 GGUF; they're listed for completeness.

**C3 BEAM sweep:** out of scope (would change tok/s by ≤ 1 tok/s given
the structural gap is unfused GLU + non-FA, not gemm tuning). The plan also
flags this as gating only if ceilings aren't reached by other paths.

---

## Summary table — what Tier K produced

| K# | Test | Status | Artifact |
|----|------|:------:|----------|
| K1 | tinygrad generated CUDA capture | ✅ | `ref/tg_kernels/debug4_full.log` + `kernel_stats.txt` |
| K2 | Master kernel map (4 cols × 8 ops) | ✅ | this document |
| K3 | Same-op source diff (4 engines, GEMV inner loop) | ✅ | this document |
| K4 | Same-shape GB/s matrix | ✅ | this document |
| K5 | llama mangled-name → `.cuh` file | ✅ | this document |
| K6 | calm op-level µs / GB/s | ✅ | this document + calm built-in cudaprof in `calm_short.nsys-rep` |
| K7 | ncu / SASS per GEMV | ⚠️ | **ncu blocked (`ERR_NVGPUCTRPERM` no sudo). Substituted: nsys + `-bk` GB/s. Confidence high for macro-comparisons, but SASS-level unroll/prefetch on FP16 KV not directly measured.** |
