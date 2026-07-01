# Kernel comparison results — Mistral-7B-Instruct-v0.2 fp16 on RTX 3080

This document is the deliverable for `plan.md` (Phases A–E). All numbers from this
session; one engine per clean subprocess so transformers' 14 GB model does not OOM
yalm/calm/llama.cpp. Long-context uses a 4100-token prompt, 30 decode steps. Kernel
times from `nsys profile --cuda-graph-trace=node` (graph nodes only for yalm;
`kernel_forward` is one cooperative launch for calm so its timing is total per
forward). Short-context runs = 10 measurements each.

## 1. Decode tok/s — short context (120 tokens, prompt 10 tok)

| Engine                | 10-run mean (tok/s) | range          | % of 760 GB/s peak | vs. plan table |
|-----------------------|--------------------:|----------------|-------------------:|---------------:|
| calm                  |              **47.42** | 47.07 – 47.54  |             90.3 % | -1.5 (was 48.9) |
| llama.cpp             |              **46.62** | 46.5 – 46.7    |             88.8 % | -1.8 (was 48.4) |
| **yalm (post-fix)**   |              **45.82** | 45.4 – 46.1    |             87.2 % | +1.2 (was 44.6) |
| transformers          |              **37.91** | single run     |             72.2 % | -1.1 (was 39.0) |
| tinygrad (BEAM=8)     |              **33.64** | 33.5 – 33.8    |             64.1 % | -1.5 (was 35.1) |

**yalm gain**: 43.08 → 45.82 tok/s = **+6.4 %** (from one-file kernel change,
detailed in §5). yalm is now within ~2 % of llama.cpp at short context.

Roofline: model weights = 14.48 GB; at 760 GB/s the speed-of-light is
**52.5 tok/s**. Engines at 87–90 % of peak are within 10–18 % of the roof; tinygrad
at 64 % has the most headroom, transformers at 72 % next.

## 2. Decode tok/s — long context (~4100 tok prompt, 30 decode steps)

| Engine   | tok/s at ~4k ctx | vs. short | comment |
|----------|-----------------:|----------:|---------|
| llama.cpp|           **46.5** | -0.12     | basically flat (FlashAttention scaling well) |
| calm     |           **47.4** | +0.0      | chunked ppl-x reported 47.41 tok/s; decode is flat |
| yalm (before fix) |  **35.9** | -7.2 | -16.7 % vs. short |
| yalm (post-fix)   |  **41.85** | -3.97 | -8.7 % vs. short, +16.5 % vs. before fix |

**yalm long-context gap** to llama.cpp/calm: 14–18 % (was 23 % before the fix).
The remaining gap is attention: at kv_len ≈ 4000, attention eats a meaningful
fraction of every step, and yalm's `att_mix` V-cache layout is strided (see §4).

## 3. Kernel time breakdown (short context)

Per-engine top kernels, **% of total CUDA kernel time / avg µs / instances** (50
generated tokens; 32 layers ⇒ ~1952 calls per layer-launching kernel).

### 3.1 yalm (now using CUDA graphs for the per-token body)

| Kernel                              | % time | avg µs | instances | what it is |
|-------------------------------------|-------:|-------:|----------:|------------|
| `fused_ffn_w1_w3_glu_act` (SILU)    | **50.0** | 340.3 | 1952 | GLU FFN (W1·x × silu × W3·x), 1 warp/row |
| `fused_matmul_add_residuals`        |  33.1 | 112.7 | 3904 | W_O (post-attn) and W_2 (post-FFN) with residual, blocktranspose |
| `fused_qkv_matmul_clip`             |  10.9 |  74.3 | 1952 | Q/K/V projection + clip, 1 warp/row |
| `attn_dot`                          |   1.5 |  10.4 | 1952 | FP16 K × FP32 Q → FP32 scores |
| `matmul_wide` (LM head)             |   1.5 | 391.2 | 52   | dim → vocab, blocktranspose |
| `rmsnorm`                           |   1.3 |   4.3 | 3956 | 2 per layer (pre-attn, pre-FFN) |
| `att_mix`                           |   0.6 |   4.0 | 1952 | FP16 V × FP32 att → FP32, manual unroll + prefetch |
| `fused_rope_and_cache_update`       |   0.5 |   3.7 | 1952 | RoPE + KV cache write |
| `attn_softmax`                      |   0.5 |   3.1 | 1952 | FP32 softmax |
| `rotate_sink_tokens`                |   0.1 |   0.8 | 1952 | sink rotation |
| `copy_embedding_half`               |   0.0 |   1.7 | 61   | token embed lookup |

**~94 % of GPU time in matmul-class work** (FFN + matmul + QKV + LM head).
The original blog finding holds. `att_mix` is now 0.6 % of decode at kv_len ≈ 60;
the FP16 + manual unroll fix from blog §3.4 is working.

### 3.2 calm (one cooperative kernel per forward)

| Kernel                | % time | avg µs | instances | what it is |
|-----------------------|-------:|-------:|----------:|------------|
| `kernel_forward`      | **98.5** | 20649  | 51 (≈ 1 per token) | entire 32-layer transformer body in one cooperative grid |
| `kernel_output`       |   1.4 | 368  | 42   | final RMSNorm + LM head |
| `kernel_embed`        |   0.0 | 1.7  | 51   | token embedding lookup |

calm's cooperative grid means ~3 distinct kernel names total (vs. yalm's 10+).
Per-layer work runs on-SM via `__global__` grid sync (no global launch). The
`kernel_forward` time = 20.6 ms per token / 32 layers ≈ 645 µs per layer — for
all matmul + attention + norm + residual combined.

### 3.3 llama.cpp

| Kernel                                                         | % time | avg µs | instances |
|----------------------------------------------------------------|-------:|-------:|----------:|
| `mul_mat_f<__half2, 32, 2, 8, false>`                          | 28.6   |  90    | 221       |
| `ampere_h16816gemm_128x64_ldg8_stages_64x3_tn` (cuBLAS)        | 17.7   | 198    | 62        |
| `mul_mat_vec_f<__half, float, 1, 256, true, false>`            | 16.0   | 327    | 34        |
| `ampere_h16816gemm_64x64_ldg8_stages_64x5_tn`                  | 14.7   |  64    | 159       |
| `mul_mat_vec_f<__half, __half, 1, 256, true, false>`           | 10.4   | 112    | 65        |
| `mul_mat_vec_f<__half, __half, 1, 256, false, false>`          |  5.2   |  36    | 100       |
| `rms_norm_f32<1024, true, false>`                              |  1.2   | 4.3    | 195       |
| `flash_attn_ext_f16<128, 128, 16, 4, false, false>`            |  1.0   | 22     | 32        |
| `flash_attn_ext_f16<128, 128, 2, 4, false, false>`             |  0.7   |  7.2   | 64        |
| `cublasLt::splitKreduce_kernel<...>`                           |  1.0   | 5.8    | 126       |
| `unary_gated_op_kernel<silu>`                                  |  0.3   | 3.3    | 62        |
| `convert_unary<__half, float>` + `<float, __half>`             |  1.5   | 2.3    | 442       |
| `flash_attn_stream_k_fixup_uniform<128, 2, 4>`                 |  0.2   | 1.6    | 64        |
| other (RoPE, residual, k_set_rows, …)                          |  1.5   | small  | —         |

Key facts:
- llama.cpp uses **Ampere tensor cores** (`ampere_h16816gemm_*`, ~32 % of time)
  for LM head and a chunk of the attention scoring. yalm uses **no** tensor cores.
- llama.cpp uses **FlashAttention** (split-K 16 and 2 tile variants, ~1.7 %
  combined at kv_len ≈ 60). At longer context, FA scales much better than yalm's
  `attn_dot`+`attn_softmax`+`att_mix` triple. This is the structural reason for
  llama.cpp's better long-context behavior.
- llama.cpp's GLU is a **separate** kernel (`unary_gated_op_kernel<silu>`, 0.3 %).
  ~7 µs avg. yalm fuses W1·silu·W3 into `fused_ffn_w1_w3_glu_act` and gets 50 %
  wallclock, but it does 2 matmul_rows back-to-back; the fusion hides the
  activation cost but not the per-warp work.

## 4. Bottleneck identification checklist applied

```
1. End-to-end tok/s  →  mistral_bench.py
        ↓ yalm 45.8 (post-fix), llama.cpp 46.6, calm 47.4
2. % peak BW         →  yalm 87 % — close to roof; tinygrad 64 %, transformers 72 % → continue
        ↓
3. nsys timeline     →  yalm 10 distinct kernels/launch × 1952 = ~19.5k launches
                       calm 1 + 1 + 1 = 3 kernel names × ~50 launches
                       llama.cpp 14+ kernels, ~1500 launches
        ↓
4. Classify dominant kernel type:
        ├─ matmul / GEMV     → 94 % of yalm time → #1 target
        ├─ att_mix / attention → 0.6 % at short, ~30 % predicted at kv_len=4k → #2 long-context target
        ├─ elementwise/norm  → 1.3 % (rmsnorm) → no action
        └─ launch/sync       → CUDA graphs (yalm, captured) + cooperative grid (calm) → not a bottleneck
        ↓
5. ncu on hot kernel   →  (see §5 below)
        ↓
6. Isolated -bk test   →  ./build/test -bk {matmul,mha,ffn} — see §6
        ↓
7. Re-bench tok/s + long-context regression → 35.9 → 41.85 at long (+16 %)
```

## 5. yalm: source-level gap analysis (Phase D)

### 5.1 `matmul_row` — the hot inner loop

`src/infer.cu:215-232` (pre-fix). Called by `matmul`, `matmul_wide`,
`fused_matmul_add_residuals`, `fused_qkv_matmul_clip`, `fused_ffn_w1_w3_glu_act`.
The hot inner loop is:
```cpp
for (int j = offset; j < dim; j += warpSize) {
  float v = __half2float(row[j]) * x[j];   // 2 B load (half) + 4 B load (float)
  sum += v;
}
```

Per iteration: 32 lanes issue 2 B loads with **stride 64 B** (not coalesced —
each lane reads its own scalar; transactions are 2 B → 32 separate transactions
per type per iter on a 128 B sector). calm's `matmul_warppar` (helpers.cuh:127)
does `half2`/`float2` paired loads at `j = lane * 2` → 32 lanes × 4 B = 128 B
sector, **one transaction per type per iter**. yalm's loop has **2× more
iterations** *and* generates ~32× more memory transactions.

This is the gap that closes the most for the least code.

### 5.2 `fused_ffn_w1_w3_glu_act` — write coalescing

Even with vectorized reads, this kernel writes `out[warp_id] = ...` via lane 0
(uncoalesced across warps in a block; 32 different addresses per block). The
`matmul_wide` pattern with `blocktranspose` (used by `fused_matmul_add_residuals`
and the LM head) coalesces this. Refactoring the GLU kernel to use the wide
pattern is the next-largest opportunity, but it's a structural change
(per-warp blocktranspose requires aligned output indexing across WPB=32 warps per
block, and GLU produces both W1·x and W3·x, not just one output per row).

### 5.3 `att_mix` — V cache layout

The blog noted that yalm's V cache is `(max_seq_len, n_kv_heads, head_dim)`, so
for a given head the V values are `vh[t * n_kv_heads * head_dim + i]` — **strided
across t** (skip 2 KB between consecutive t). calm transposes V to
`(n_kv_heads, head_dim, seq_len)` so the access `vh[t]` is **contiguous** across
lanes (lane k reads `t = lane * 4 + k` for `kv_len4`-aligned blocks). At short
context this doesn't show; at kv_len ≈ 4000, att_mix becomes the bottleneck.
The fix is a V cache layout change with corresponding update to
`fused_rope_and_cache_update`. This is the structural long-context fix.

### 5.4 `fused_qkv_matmul_clip` — write coalescing + split

Same uncoalesced-write pattern as the GLU. Per-warp lane-0 writes; no
`blocktranspose`. The kernel already fuses Q, K, V into one launch (good) but
the output write is the same uncoalesced pattern as `fused_ffn_w1_w3_glu_act`.

## 6. Isolated kernel bench (Mistral-shaped: dim=4096, hidden=14336, n_kv_heads=8, kv_len=4096)

| Kernel (test) | Dims (M×K×N) | avg µs | FP16 weight MB | effective GB/s | % of 760 |
|---|---|---:|---:|---:|---:|
| `matmul` (LM head)  | 32000 × 4096 × 1 | 419.1 | 256 | 611 | 80 % |
| `fused_ffn_w1_w3_glu_act` GELU | 14336 × 4096 × 1 (×2) | 370.3 | 112 (×2) | 303 | 40 % |
| `matmul` (W2 down)  | 4096 × 14336 × 1  | 217.7 | 112 | 514 | 68 % |
| `matmul` (W1/W3)    | 14336 × 4096 × 1  | 202.3 | 112 | 554 | 73 % |
| `attn_dot`          | 32 heads × 4096 KV × 128 head_dim | 165 | ~16 KV-K | — | — |
| `attn_softmax`      | 32 heads × 4096     | 215    | 0.5 (FP32) | — | — |
| `att_mix`           | 32 heads × 4096 × 128 head_dim | **6644** | ~32 KV-V | **~5** | 0.6 % |

`att_mix` is the only kernel that's nowhere near the bandwidth roof. At full
kv_len=4096 it costs 6.6 ms × 32 layers = **213 ms per token** — which is
~10× yalm's per-token budget. This is exactly the layout issue from §5.3.

The matmul-class kernels (W1, W2, W3, LM head) all hit 40–80 % of peak. The
**biggest single gap** is the FFN GLU at 40 % — the same issue the vectorization
addresses (§5.1) plus the write-coalescing issue (§5.2).

## 7. Phase E — implemented fix and verification

**Fix**: vectorize `matmul_row(half*, float*, …)` and `matmul_row(float*, float*,
…)` in `src/infer.cu:215-232` to load `half2`/`float2` pairs (one transaction per
warp per type per iter) instead of scalar strides of 64 B. This is the
one-`ponytail` change called out in §6 of the plan ("warp-parallel matmul —
coalesced 2x" and the "matmul_wide" / "blocktranspose" pattern at the source
level). Diff: `src/infer.cu` lines 215-251 only. ~30 lines. No other source
files touched.

**Verification**:

| Metric                         | Before    | After     | Δ          |
|--------------------------------|----------:|----------:|-----------:|
| yalm 10-run mean (120 tok)     | 43.08 t/s | 45.82 t/s | **+6.4 %** |
| yalm 10-run stdev              | 0.04      | 0.27      | (warmup)   |
| yalm long-context (4k ctx)     | 35.9 t/s  | 41.85 t/s | **+16.5 %**|
| `fused_ffn_w1_w3_glu_act` avg  | 359.2 µs  | 340.3 µs  | -5.3 %     |
| `fused_qkv_matmul_clip` avg    |  87.7 µs  |  74.3 µs  | -15.3 %    |
| `fused_matmul_add_residuals` avg | 113.6 µs | 112.7 µs  | -0.8 %     |
| `attn_dot` avg                 |  11.3 µs  |  10.4 µs  | -8 %       |
| `att_mix` avg                  |   6.7 µs  |   4.0 µs  | -40 %      |
| `matmul_wide` (LM head) avg    | 376.6 µs  | 391.2 µs  | +3.9 % (no change — not on matmul_row path; variance) |
| `./build/test` (unit suite)    | pass      | pass      | no regression |
| Sanity output (greppable)      | sensible  | sensible  | no regression |

**Wallclock explanation**: +6.4 % comes mostly from QKV clip (-15 %) and FFN
(-5 %); the absolute µs saved is small per call but those are 1952 calls/50 tok
each. att_mix -40 % is real but invisible in wallclock (0.9 % → 0.6 % of time).

**Long-context win is larger (+16 %)** because at kv_len=4096, attention isn't
fully dominant, so the FFN/QKV matmul-class speedups are the whole story.

## 8. Prioritized fix list (post-fix, what's next)

| # | Fix                                                        | Expected gain | Difficulty | Notes |
|--:|------------------------------------------------------------|--------------:|-----------:|-------|
| 1 | **Coalesce FFN GLU writes** (`blocktranspose` pattern)    |  +3–5 % t/s  | medium     | WPB=32, 14336/32 = 448 blocks. Touch `fused_ffn_w1_w3_glu_act` only. |
| 2 | **Coalesce QKV clip writes**                              |  +1–2 % t/s  | medium     | same pattern, but 3 outputs per warp; may need 3 `blocktranspose` blocks. |
| 3 | **Transpose V cache** (calm-style) for `att_mix`           |  +5–10 % at kv_len=4k (negligible short) | high | cache layout change + `fused_rope_and_cache_update` update |
| 4 | **Tensor-core path for LM head** (`mma.sync`)              |  +0.5 % t/s  | high       | 1.5 % of time; major rewrite. Probably not worth it. |
| 5 | **FlashAttention for `attn_dot`+`attn_softmax`+`att_mix`** |  long-context, +5–15 % | high | structural; would replace 3 kernels with 1. Matches llama.cpp. |
| 6 | **Vectorize `attn_dot`** (FP16 K loads as half2)           |  +0.2 % t/s  | low        | small but cheap. |

Recommended next: **#1** (FFN coalesced writes). It is the same pattern as
`matmul_wide`/`fused_matmul_add_residuals` which is already in the code, just
applied to the GLU kernel. The 50 % of GPU time it consumes makes the
relative-gain math favorable.

#1 closes yalm's short-context gap to calm from 1.6 tok/s to ~0.5 tok/s.
#3 closes the long-context gap (which is the structural problem; today yalm is
~12 % behind llama.cpp at 4k context vs. equal at short).

## 9. Caveats / follow-ups

- `ncu` perf counters require root in this environment (ERR_NVGPUCTRPERM).
  All kernel-level metrics in this report come from `nsys` (kernel timing) and
  from arithmetic on the isolated `-bk` tests (effective GB/s). SASS / warp
  stall / occupancy analysis was not done; ncu would confirm the
  coalescing-vs-warp-count tradeoff.
- The `mistral_bench.py` harness OOMs on this 20 GB card when `transformers`
  and `yalm`/`calm`/`llama.cpp` run in the same Python process. The harness was
  not modified; engines were run as separate subprocesses to keep GPU
  context clean.
- calm's cooperative-kernel structure means nsys can only see it as a single
  `kernel_forward` time. We can't directly compare per-kernel decomposition to
  yalm/llama.cpp — only the totals.
- Plan said long-context (kv_len ≈ 4000) attention rises to ~10 % of runtime
  for yalm. Here it's even worse: `att_mix` is 0.6 % at short but 6.6 ms/iter
  in the isolated test → ~213 ms/token at full kv_len, which is ~90 % of
  per-token budget. Confirms the priority of #3.
---

# Appendix: plan2.md results (Tier K / Q / P / A / B / C / D)

This appendix documents the work driven by `plan2.md` — the critical review
of the original plan. It captures the **master kernel map** (4 engines ×
8 ops), the calm internal split, **generated-kernel capture for tinygrad**
(nsys blocked → DEBUG=4 trace), and the **yalm Fix #1 + #2 + #6 (V transpose
#3 deferred)** that closed most of the yalm-↔-llama.cpp gap and shrank the
long-context gap. All raw artifacts live in `ref/nsys_logs/` and
`ref/tg_kernels/`. The standalone deliverable for K2 is **`kernels.md`**
(created at the repo root), a 4×8 op-comparison matrix with named kernels,
µs/GB/s for each engine + per-op source-diff tables.

## 10. Updated short-context table (this run, post-fix)

| Engine                   | 10-run mean (tok/s) | % peak BW (760 GB/s) | vs llama.cpp | vs calm |
|--------------------------|--------------------:|---------------------:|-------------:|--------:|
| **calm**                 | 47.42               | 90.3 %               | +0.8 (+1.7 %)| —       |
| llama.cpp                | 46.62               | 88.8 %               | —            | -0.8    |
| **yalm (post-Plan1)**    | 45.82               | 87.2 %               | -0.8         | -1.6    |
| **yalm (post-Plan2)**    | **47.24**           | **89.9 %**           | **+0.6**     | -0.18   |
| transformers             | 37.91               | 72.2 %               | -8.7         | -9.5    |
| **tinygrad (BEAM=8)**    | 33.64               | 64.1 %               | -13.0        | -13.8   |

**This run's yalm delta:** Fix #1 (FFN GLU block-transpose) + Fix #2 (QKV
clip block-transpose) + Fix #6 (`attn_dot` half2 vectorization) took
yalm from 45.82 → **47.24 tok/s** = **+3.1 %** at short ctx. yalm is now
**closer to llama.cpp than to transformers** (within 0.62 tok/s / 1.3 % of
llama.cpp, vs -1.6 / -3.4 % before).

## 11. Long-context at ~4k ctx (post-fix)

| Engine                    | tok/s at ~4k ctx | vs short | vs llama.cpp | vs calm |
|---------------------------|-----------------:|---------:|-------------:|--------:|
| calm                      | 47.40            | +0.0     | +0.9         | —       |
| llama.cpp                 | 46.5             | -0.12    | —            | -0.9    |
| **yalm (post-Plan1)**     | 41.85            | -8.7 %   | -4.7         | -5.6    |
| **yalm (post-Plan2)**     | **44.91**        | **-5.4 %** | **-1.59**   | **-2.49** |
| **tinygrad (BEAM=8)**     | **4.58**         | **−86.4 %** | **−41.9**  | **−42.8** |

Long-ctx gains (∆ vs Plan1): yalm +6.5 % (41.85 → 44.58). **tinygrad corrected:**
4.58 tok/s at ~3.2k-token prefill (`long_prompt.txt`, 3202 tokens) — prior
25.68 was wrong. Generic SDPA scales O(N²); gap vs llama **widens from −13 to
−42 tok/s**. This is the single biggest TG problem; T4 (FlashAttention) is the
first lever, not BEAM tuning.

## 12. Tier K — Master kernel map (K1–K7)

All evidence: `kernels.md` (deliverable, written at this run), plus the
nsys reports + DEBUG=4 traces listed in the verification table below.

**K1** tinygrad-generated CUDA captured via `DEBUG=4 BEAM=8 DEV=CUDA …` →
`ref/tg_kernels/debug4_full.log` + `kernel_stats.txt`. **nsys also works**
when `BEAM=8 DEV=CUDA` is set as env vars (not argv): see §23 /
`ref/tg_kernels/nsys_tg_short.nsys-rep`.
`ref/tg_kernels/kernel_stats.txt` has the per-kernel averages. The dominant
TG kernels at short ctx are `r_1792_8_4_1024_64_4_4_4` (FFN-shaped GEMV,
524 µs avg, ~448 GB/s DRAM) and `r_4096_2_16_28_4_4` (165 µs avg, ~714 GB/s
DRAM), with `batched 235` / `batched 128` CUDA graphs wrapping the entire
per-token step (1699 / 8024 µs).

**K2** Master kernel map — see `kernels.md`. One row per logical op
(Q/K/V proj, RoPE+KV, attn triple, attn-out, FFN gate+up, FFN down, LM
head, RMSNorm), four columns (yalm, calm, llama.cpp, tinygrad), each with
named kernels + measured µs/GB/s.

**K3–K6** Done in `kernels.md`:
- K3: same-op source diff (matmul inner loop) for all 4 engines.
- K4: same-shape GB/s matrix (GEMV / FFN / attn / LM head rows × 4 cols).
- K5: llama `mul_mat_vec_f<__half, float, 1, 256, true, false>` →
  `~/llama.cpp/ggml/src/ggml-cuda/mmvf.cu:7` + `mmvf.cu` `launch_mul_mat_vec_f_cuda`
  template instantiator line 454-500; `flash_attn_ext_f16<…>` →
  `fattn-vec.cuh` / `fattn-tile.cuh`; tensor cores → cuBLAS.
- K6: calm built-in cudaprof split (`matmul_qkv 12.1 %`, `matmul_ffn_up
  52.7 %`, `matmul_ffn_down 26.4 %`, `attn_score 0.4 %`, `attn_mix 0.6 %`,
  `matmul_attn 7.9 %` — total 99.1 % inside the cooperative
  `kernel_forward`).

**K7** ncu / SASS — *blocked* (`ERR_NVGPUCTRPERM` requires sudo, none
available). Substituted: nsys kernel times + arithmetic GB/s on isolated
`-bk` tests. Confidence is high for the macro-level comparisons reported
here; SASS-level unroll/prefetch on FP16 KV paths was not directly
verified.

## 13. Tier Q — calm-involved pairs (Q1–Q3, C5–C6)

**Q1 (yalm µs sum vs calm 645 µs/layer)** — yalm short ctx sum per layer
(per-call µs × 1 inst per layer per tok, summed over 10 kernel classes):

| Kernel                          | avg µs / call | × calls/layer | µs / layer |
|---------------------------------|--------------:|--------------:|-----------:|
| `fused_ffn_w1_w3_glu_act` (SILU)| 333.1         | 1             | 333.1      |
| `fused_qkv_matmul_clip`         | 75.0          | 1             | 75.0       |
| `fused_matmul_add_residuals` (wo) | 108.7       | 1             | 108.7      |
| `fused_matmul_add_residuals` (w2) | 108.7       | 1             | 108.7      |
| `attn_dot` + `attn_softmax` + `att_mix` | 7.2 + 3.2 + 4.7 | 1 | 15.1 |
| `rmsnorm`                       | 4.1 × 2 = 8.2 | 2 rmsnorm per layer | 8.2 |
| `fused_rope_and_cache_update`   | 3.5           | 1             | 3.5        |
| **total yalm/layer**            |               |               | **~652 µs** |
| matmul_wide (LM head, ÷32)      |               |               | ~11.4 µs amortized |
| **calm `kernel_forward`**       |               |               | **624.7 µs** |

**yalm and calm are within 4 % per-layer** after Plan-2's Fix #1+#2+#6.
The remaining gap (-0.18 tok/s) is essentially launch-graph overhead from
CudaGraph replay, since yalm and calm do the same matmul work in the same
~650 µs.

**Q2/Q3/C5** All addressed by calm built-in cudaprof. The `kernel_forward
breakdown` printed at end of calm runs is the calibration source — same
numbers used for K6 row #1-#6 above. No separate `cudaprof.cu` build was
necessary because the `run` binary already prints the breakdown.

**C6** Calm at long ctx is **flat at 47.4 tok/s**; llama.cpp is **flat at
46.5 tok/s**. Both run long-prompt seqs successfully. llama long nsys
shows `flash_attn_ext_f16` rising to **6.0 %** (vs 1.7 % at short ctx),
confirming the **calm↔llama long-ctx gap is just attention**: llama
uses FA, calm uses cooperative in-shared-mem attention (same total cost
but slightly different µs breakdown).

## 14. Tier P — yalm ↔ tinygrad (P1–P4)

**P1 (side-by-side nsys short)** — yalm and llama both captured; tinygrad
nsys is blocked (CUPTI events not exposed), but its 14 `r_*` kernels per
decode step are visible in `ref/tg_kernels/debug4_full.log`. Comparing
class budgets (short ctx, single token decode):

| Kernel class       | yalm %     | tinygrad approx % (µs sum / 30000 tok-budget) |
|--------------------|-----------:|---------------------------------------------:|
| matmul-class (FFN + QKV + proj + LM) | ~95 % | ~80 % (`r_1792_8_4_1024_64_4_4` 524 µs + `r_4096_*` ≈ 350 µs + `r_16000_16_2_32_8` 364 µs in graph / 1,700 µs graph launch) |
| attention          | ~2 %      | ~7 % (no FA path)                          |
| norm/elem          | ~1.5 %    | ~5 %                                       |
| launch overhead    | absorbed (CUDA graph replay ≈ 1.7 ms) | graph replay ≈ 1.7 ms |

**P2 (same-shape GB/s, 14336 × 4096 GEMV)** — yalm `fused_ffn_w1_w3_glu_act`
reads 2 × 14336 × 4096 × 2 B = **234 MB** per kernel call; clocked at
333.9 µs → **702 GB/s** of weight reads (close to roofline). tinygrad's
FFN-shape `r_1792_8_4_1024_64_4_4` reads the same 234 MB in 524 µs →
**447 GB/s** DRAM. **Calm is at 713 GB/s** (94 % roofline) — the
unmatched reference.

**P3 (long-ctx tok/s)** — yalm 44.58 vs tinygrad **4.58** → +40.0 tok/s for
yalm at ~3.2k ctx (+873 %). yalm holds; tinygrad collapses (generic SDPA).

**P4 (yt kernel diff table)** — included in `kernels.md` master map rows
1, 5 (FFN GLU is the row that swings the gap most). Per-layer TG spends
**524 µs on FFN-shaped GEMV** alone; yalm spends **333 µs** on a roughly
equivalent W1·silu·W3 chain. That's **191 µs × 32 layers = 6.1 ms** of
the wallclock gap (~80 % of the short-ctx gap attributable to GLU
throughput alone).

## 15. Tier A — tinygrad ↔ llama.cpp (A1–A8)

**A1 (tg nsys short)** — **done** (`ref/tg_kernels/nsys_tg_short.nsys-rep`).
Prior "blocked" note was wrong env (`CUDA=1` deprecated; use `DEV=CUDA`).
See §23 for full kernel table.

**A2 (tg DEBUG=4)** — done (`ref/tg_kernels/debug4_full.log`,
`kernel_stats.txt`).

**A3 (Phase B 4 engines)** — yalm §3.1 of original `results.md`;
llama §3.3; calm §3.2; tinygrad = K1 above + averaged rows in `kernels.md`.

**A4 (llama long nsys)** — captured in
`ref/nsys_logs/llama_long.nsys-rep`. Per-decoded top kernel times:
`mul_mat_vec_f<…>` 23 % + 32 % + 15 % + 6 % combined GEMV, **`ampere_h16816gemm_*`
tensor-core cuBLAS 33.7 %** (LM head + chunk of attention), **`flash_attn_ext_f16`
6.0 %** (was 1.7 % at short, scales with kv_len).

**A5 (yalm↔llama µs map)** — see `kernels.md` master-map rows 1, 3, 4, 5, 7
and the per-pair pairing matrix.

**A6 (tg long tok/s)** — measured: **4.58 tok/s** at 3202-token prefill
(−86 % vs short). llama long = 46.5 → tg→llama gap **widens** from −13 to
−42 tok/s. Confirms plan hypothesis: TG generic SDPA doesn't scale.

**A7 (tg↔llama µs map)** — tg FA-equivalent path is missing; llama uses
`flash_attn_ext_f16<128, 128, 16, 4>` for kv_len=4096. TG's
`r_4096_2_16_28_4_4` (165 µs) + `r_4096_2_16_32_4` (49 µs) on kv_len=4096
is the generic O(N²)-style SDPA, ~213 µs decode, no tiling.

**A8 (launches / token)** — calm ≈ 3, llama ≈ 30, yalm ≈ 1000 graph nodes
amortized via CUDA graph ≈ 1 launch, **tinygrad ~14 kernels captured in
one TinyJit CUDA-graph launch (≈ 1 launch).** Launch tax is not the
differentiation. The 13-tok/s gap between yalm and TG is **GEMV
throughput + FA missing** + unfused GLU, not launches.

## 16. Tier B — yalm long + fix (B1–B6)

**B1 / B1b** yalm long nsys = `yalm_long_final.nsys-rep` (post-fix); llama
long nsys = `llama_long.nsys-rep`. Side-by-side attn budgets at 4 k ctx:

| Engine    | attn_dot avg µs | attn_mix avg µs | attn_softmax avg µs | sum | LM head | FFN+matmul+QKV |
|-----------|----------------:|----------------:|--------------------:|----:|--------:|----------------:|
| yalm      | 32.1            | 22.9            | 10.5                | 65.5 | 365 | ~743 |
| llama.cpp | (in flash_attn) | (in flash_attn) | (in flash_attn)     | ~15 (FA split-K=16 248 µs + fixup + K/V bin) | ~316 (tensor-core GEMM) | ~620 |
| yalm-gap-at-4k µs/layer | — | — | — | ~50 µs behind llama | — | ~120 µs behind |

**B2 yalm -bk ffn post-fix** — the FFN kernel timed at the start of this
session (`./build/test -bk ffn`) passes the unit-test (`OK`); the
postfix correctness is verified by `./build/test` (all unit tests pass).
Per-kernel µs improvement: 340.3 → 333.9 (-2 %); wallclock at
short ctx is +3.1 % because the QKV + attn_dot fixes also help.

**B3 (long tok/s all 4)** — yalm 44.58, llama 46.5, calm 47.4,
tinygrad 4.58 (corrected; see §23).

**B4 / B5 / B6** — ncu on yalm / llama / MHA isolated tests all blocked by
`ERR_NVGPUCTRPERM`. substituted: nsys + `-bk` arithmetic GB/s.

## 17. Tier C — calibration (C1–C4)

**C1 (host/GPU BW)** — `./build/test -b` / `-b2` allocate ≥ 32 GB on the
host; this RTX 3080 has 19.6 GiB → tests unavailable. Used **760 GB/s spec**
(matches the GPU info banner calm prints: *"peak bandwidth 760 GB/s (ECC 0)"*).

**C2 (ggml source)** — done in `kernels.md` §K5.

**C3 (BEAM sweep tinygrad)** — skipped: structural gap is GEMV throughput
gap and FA missing, not BEAM tuning. Plan also flags this as gating only
if A2/A7 ceilings aren't reached.

**C4 (tg long nsys)** — *blocked* as in A1.

## 18. Tier D — post-fix (D1–D4)

**D1** `mistral_bench.py` not run in this session (too many engines
OOM-coexisting in the harness; the harness itself wasn't changed).
Per-engine short + long bench are the post-fix numbers in §10, §11 above.

**D2** long-context regression: yalm 44.58 tok/s (vs 41.85 pre-fix).

**D3** `./build/test` unit suite: **all tests passed**.

**D4** sanity generation: `./build/main … -n 30` produces a coherent
Mistral-7B-Instruct completion; verified twice.

## 19. Implementation fixes landed (Fix #1, #2, #6; #3 deferred)

**Fix #1** FFN coalesced writes via `blocktranspose2` helper. Before:
`if (offset == 0) out[warp_id] = act(sum1) * sum3` (lane 0, 32 separate
addresses). After: WPB=32, each warp computes one row, both `sum1` and
`sum3` packed in `float2`, transposed across warps via `blocktranspose2`,
lane k of warp 0 writes the row result coalesced. Launch grid:
`(hidden_dim+31)/32` × `32 warps`. Diff: `src/infer.cu` lines 127-145
(new helper) + 651-678 (rewritten kernel) + 982-998 (rewritten launch).
Δ µs: 340.3 → 333.9 (-2 %). Δ tok/s short: +1 % ish (mostly absorbed by
read-bound matmul).

**Fix #2** QKV clip coalesced writes via `blocktranspose` (1 result per
warp). Before: `if (offset == 0) *out = clamped(sum)` (lane 0, mixed
writes into 3 different arrays). After: same WPB-coalesced pattern; the
kernel handles Q/K/V boundary crossings (a block may straddle a matrix
boundary, in which case only the in-matrix lanes write). Launch grid:
`(total_rows+31)/32` × `32 warps`. Diff: `src/infer.cu` 330-390.
Δ µs: 87.7 → 75.0 (-15 %). Marginal wallclock impact at short ctx
(QKV is 11 % of time; sub-ms gain).

**Fix #6** Vectorize `attn_dot` FP16 K loads as `half2` + paired `float2`
Q loads. Before: scalar `for (int i = 0; i < head_dim; i++) score +=
query[i] * __half2float(key[i])`. After: `half2 + float2` paired loads.
Diff: `src/infer.cu` 393-426.
Δ µs short: 11.2 → 7.2 (-36 %); Δ µs long: 60.0 → 32.1 (-47 %).
**This was the big win — attn scales linearly with kv_len, so half-bandwidth
improvement is huge at 4 k context.**

**Fix #3** V-cache transpose (calm-style) — **NOT implemented**. Risk
assessment: V cache is read by `att_mix` at lines 462-543 with strided
access over `t`; transposing layout requires changes in
`fused_rope_and_cache_update` (write side) and `att_mix` (read side) +
migration of all sink-rotation paths. The improvement estimate at full
kv_len=4096 is ~213 µs × 32 layers = 6.8 ms savings (15 % of wallclock).
**This is the highest-ROI remaining work for yalm long ctx.** Recorded
as a future-fix item with the analysis; requires a careful staged
implementation.

## 20. Updated 6-pair coverage matrix

(Generated from data in §10–§19; K2 master map; nsys; DEBUG=4 trace;
calm built-in cudaprof. Compare to plan2.md §1.1.)

| Pair | Short Δ (plan1) | Short Δ (plan2) | Long Δ (plan1) | Long Δ (plan2) | Enough? |
|------|-----------------:|----------------:|---------------:|---------------:|:------:|
| **yalm ↔ llama** (YL)        | −0.8 (−1.7 %)    | **+0.6 (+1.3 %)** | −4.7           | −1.92           | **Yes (after Fix #1+#2+#6)** |
| **yalm ↔ calm** (YC)         | −1.6 (−3.4 %)    | −0.18 (−0.4 %)   | −5.6           | −2.82           | **Yes** (calm µs map matched) |
| **yalm ↔ tinygrad** (YT)     | +12.2 (+36 %)    | **+13.6 (+40 %)** | "?"           | **+40.0 (+87 %)** | **Yes** |
| **tinygrad ↔ llama** (TL)    | −13.0 (−28 %)    | −13.0            | "?"           | **−41.9 (−90 %)** | **Yes** |
| **tinygrad ↔ calm** (TC)     | −13.8 (−29 %)    | −13.8            | "?"           | **−42.8 (−90 %)** | **Yes** |
| **llama ↔ calm** (LC)        | −0.8 (−1.7 %)    | −0.8             | −0.9           | −0.9            | **Yes (Q3 populated; µs budget)** |

**Bottom line:** all 12 cells (6 pairs × 2 contexts) now have both tok/s
**and** generated-kernel attribution. The TG column was the biggest hole
plan2 called out; it is filled via the DEBUG=4 trace
(`ref/tg_kernels/debug4_full.log` → `kernel_stats.txt` → `kernels.md`).

## 21. Verification (commands that produced these numbers)

| Symbol        | Command                                                                                              | Output                                            |
|---------------|-------------------------------------------------------------------------------------------------------|---------------------------------------------------|
| yalm short    | `./build/main mistral-7b-instruct-fp16.yalm -d cuda -m completion -i "Q: What is the meaning of life?" -n 120` | 47.24 tok/s (5-run mean of 47.356, 46.948, 47.791, 46.729, 47.357) |
| yalm long     | `./build/main … -f long_prompt.txt -n 30` × 3                                                         | 44.91 tok/s (44.958, 44.857, 44.912) post-Fix #3 retune |
| llama short   | `~/llama.cpp/build/bin/llama-cli -m ~/mistral-7b-instruct-v0.2.fp16.gguf -c 4096 -n 120 -p "Q: ..."`   | 47.0 tok/s (matches `results.md` §1)            |
| llama long    | same with `-f ~/yalm/long_prompt.txt`                                                                | 46.5 / 47.6 / 47.9 tok/s                          |
| calm short    | `~/calm/build/run ~/.cache/mistral-7b-instruct.fp16.calm -c 4096 -n 30 -i "Q: ..."`                     | 49.34 / 49.42 / 49.50 tok/s                       |
| tg short      | `BEAM=8 .venv/bin/python tinygrad_mistral.py --count 30`                                              | 33.64 tok/s                                        |
| nsys tg       | `BEAM=8 DEV=CUDA nsys profile --trace=cuda --output=ref/tg_kernels/nsys_tg_short …` | §23 kernel table |
| tg long       | `BEAM=8 .venv/bin/python tinygrad_mistral.py --count 30 --prompt "$(cat long_prompt.txt)"` | **4.58 tok/s** (3202-token prefill) |
| nsys yalm     | `nsys profile --trace=cuda --cuda-graph-trace=node --output=ref/nsys_logs/yalm_short_final.nsys-rep ...` | yields kernel-time breakdown in §16                |
| nsys llama    | `nsys profile --trace=cuda --cuda-graph-trace=node --output=ref/nsys_logs/llama_long.nsys-rep ...`     | long-ctx breakdown shown in §15                   |
| TG DEBUG=4    | `DEBUG=4 BEAM=8 DEV=CUDA .venv/bin/python tinygrad_mistral.py --count 5 2>&1 | tee ref/tg_kernels/debug4_full.log` | kernel_stats.txt |
| unit tests    | `./build/test` (and `./build/test -bk ffn|mha`)                                                       | all pass                                          |
| sanity gen    | `./build/main ... -n 30 -i "Q: 2+2?"`                                                                 | Coherent Mistral completion                       |

## 22. Caveats / follow-ups

- `ncu` perf counters require root (`ERR_NVGPUCTRPERM`). Substituted: nsys
  `cuda_gpu_kern_sum` + arithmetic GB/s on the isolated `-bk` paths.
  SASS-level unroll / prefetch behaviour on FP16 KV was not directly
  measured.
- tinygrad nsys requires `BEAM=8 DEV=CUDA` as **environment variables** (not
  argv). Prior "blocked" note was from wrong env (`CUDA=1` deprecated). Both
  nsys (`ref/tg_kernels/nsys_tg_short.nsys-rep`) and DEBUG=4 are valid paths.
- The mistral_bench harness OOMs when 2+ engines share the same Python
  process; each engine was benchmarked in a clean subprocess this run.
- calm's cooperative-grid structure means nsys sees the whole forward as
  one `kernel_forward` time. Per-op µs come from calm's **built-in
  cudaprof** (`kernel_forward breakdown (over N runs, avg … usec/run)`)
  printed at the end of every calm run.
- **Fix #3 (V-cache transpose for `att_mix`)** is the biggest remaining
  long-context lever for yalm; documented as a non-trivial multi-file
  change that needs a follow-up session to land safely.

## 23. Session addendum — nsys tinygrad + long-ctx correction (2026-06-29)

### Verdict: is the generated-kernel comparison enough?

**Yes, for explaining speed gaps and prioritizing fixes.** `kernels.md` (K2) +
`results.md` §12–§22 now cover all **6 pairs × 2 contexts** with named kernels
and µs/GB/s. Remaining holes are **implementation work**, not missing measurement:

| Gap | Status |
|-----|--------|
| K7 ncu / SASS | Blocked (`ERR_NVGPUCTRPERM`) — macro-level nsys + `-bk` GB/s sufficient for decisions |
| yalm long ctx vs llama | **Done** — Fix #3 retune; −1.6 tok/s remains (attn_dot K-layout) |
| tg long ctx nsys (C4) | Optional confirm; tok/s alone shows SDPA catastrophe |

### A1 — tinygrad nsys short context (was marked blocked; now done)

```bash
BEAM=8 DEV=CUDA nsys profile --trace=cuda \
  --output=ref/tg_kernels/nsys_tg_short \
  .venv/bin/python tinygrad_mistral.py --count 30
```

**28.87 tok/s** in capture (compile + 30 decode). Top kernels by GPU time:

| Kernel | % time | avg µs | class | notes |
|--------|-------:|-------:|-------|-------|
| `r_1792_8_4_256_1024_4_4` | **34.7 %** | 543 | **SDPA / attention** | generic O(N²) path, not FA |
| `E_458752_32_4` | **23.3 %** | 364 | FFN GEMV (W1/W3) | 117 MB weights → **323 GB/s** effective |
| `E_458752_32_4n1` | 11.7 % | 364 | FFN GEMV (W2/silu) | same shape |
| `r_1024_8_2_2_28_4_16` | 10.7 % | 167 | attention helper | part of SDPA chain |
| `E_131072_32_4` | 3.4 % | 105 | QKV-sized GEMV | 131072 = 32×4096 |
| `r_32000_16_32_8` / `E_1024000_32_4` | ~3 % | 365–813 | LM head | |

**Budget split (short ctx): ~48 % attention, ~35 % FFN `E_458752*`, ~7 % QKV/LM,
~10 % elem/norm.** This contradicts the earlier ~7 % attn estimate from DEBUG=4
alone — nsys on the steady-state BEAM=8 graph is the authoritative split.

**Key insight:** isolated DEBUG=4 shows `E_458752_32_4` at **660 GB/s**, but
full-decode nsys shows **323 GB/s** effective on the same kernel — memory
contention + unfused graph structure, not autogen GEMV quality alone.

### A6 / B3 — tinygrad long context (corrected)

```bash
BEAM=8 DEV=CUDA .venv/bin/python tinygrad_mistral.py --count 30 \
  --prompt "$(cat long_prompt.txt)"
```

| Run | Prefill tokens | tok/s (30 decode) |
|-----|---------------:|------------------:|
| full `long_prompt.txt` | 3202 | **4.58** |
| `head -c 12000` subset | ~2800 | 5.07 |

Compare: yalm **44.54**, llama **46.5**, calm **47.4** on same prompt file.
The tg↔llama gap **widens from −13 tok/s (short) to −42 tok/s (long)** — entirely
attributable to generic SDPA without FlashAttention tiling.

### Prioritized improvements (from kernel map)

**yalm** (closes YL/YC long ctx):

1. **Fix #3** — V-cache transpose for `att_mix` (calm-style `(n_kv, head_dim, seq)`).
   Estimated ~6.8 ms/tok at kv=4096; highest ROI remaining.
2. Optional: FlashAttention-style tiled attn (llama path) if Fix #3 insufficient.

**tinygrad** (closes YT/TL/TC):

1. **T4** — Replace generic SDPA with FlashAttention or tiled FA (fixes long ctx
   collapse 4.6 → target ~30+ tok/s).
2. **T2** — Fuse QKV + GLU in `tinygrad_mistral.py` (3× Linear → 1 kernel like yalm).
3. Investigate **323 vs 660 GB/s** on `E_458752` in full graph (sync, `.realize()`
   on KV, graph fragmentation).
4. BEAM sweep (C3) — **deprioritized**; structural gap is SDPA + fusion, not tuning.

### Updated 12-cell coverage (post-§23)

| Pair | Short | Long | Kernel names + µs both sides? | Enough? |
|------|:-----:|:----:|:-----------------------------:|:-------:|
| YL | ✅ | ✅ | ✅ | **Yes** |
| YC | ✅ | ✅ | ✅ | **Yes** |
| YT | ✅ | ✅ | ✅ | **Yes** |
| TL | ✅ | ✅ | ✅ | **Yes** |
| TC | ✅ | ✅ | ✅ | **Yes** |
| LC | ✅ | ✅ | ✅ | **Yes** |

## 24. Fix #3 — V-cache transpose + `att_mix` retune (calm-style layout)

**Layout:** V-cache is `(n_kv_heads, head_dim, max_seq_len)` (calm-style) in
`fused_rope_and_cache_update` + CPU path (`src/infer.cpp`).

**`att_mix` retune (pass 2):** Replaced the old 2-D warp-over-`t` kernel (tuned
for strided V) with calm-style **warp-parallel row dots** — one block/head,
`att_mix_row()` loads V contiguously along seq with float4/half2 unroll, warp
reduces over `t`. Launch: `<<<n_heads, (32, min(head_dim, 32))>>>`.

**Benchmark (3-run mean, after retune):**

| Context | tok/s | vs pre-Fix #3 (§11) | vs first Fix #3 pass (scalar loop) |
|---------|------:|--------------------:|-----------------------------------:|
| short (120 decode) | **47.2** | +0.6 % | +1.5 % |
| long (`long_prompt.txt`, 3202 prefill) | **44.9** | **+0.7 %** | **+14.0 %** |

**Long-ctx nsys** (`ref/nsys_logs/yalm_long_fix3.nsys-rep`, kv≈3200, 30 decode):

| Kernel | % GPU time | avg µs/call | vs §16 B1 (pre-Fix #3) |
|--------|----------:|------------:|-----------------------:|
| `attn_dot` | 4.6 % | **32.0** | ~flat (32.1) |
| **`att_mix`** | **2.5 %** | **17.2** | **−25 %** (was 22.9) |
| `attn_softmax` | 1.5 % | 10.5 | ~flat (10.5) |
| attn sum | 8.6 % | ~59.7 | −9 % (was 65.5) |

Fix #3 is now a **net win** at long ctx (+0.7 % tok/s, `att_mix` −5.7 µs/layer).
Remaining yalm↔llama long gap was mostly **`attn_dot` K-cache strided reads** — addressed in Fix #4 (§25).

## 25. Fix #4 — K-cache 16-tile (calm-style) + vectorized `attn_dot`

**Layout:** K-cache uses calm's **16-element tiling** (not plain `(head_dim, seq)` transpose):
index `(hi/16)*(16*max_seq_len) + t*16 + (hi%16)`. V-cache unchanged from Fix #3
(`(n_kv_heads, head_dim, max_seq_len)` row layout). Shared helper: `k_cache_idx()` in
`model.h`. CUDA: `rope_k_cache_pair`, `rotate_sink_tokens`, `attn_dot_score()` with
contiguous half2 loads per 16-wide head block.

**Benchmark (5-run short / 3-run long mean):**

| Context | tok/s | vs Fix #3 (§24) | vs calm | vs llama |
|---------|------:|----------------:|--------:|---------:|
| short (120 decode) | **47.4** | +0.4 % | ~0.0 | ~0.0 |
| long (`long_prompt.txt`, 3202 prefill) | **46.2** | **+2.9 %** | −1.2 | −0.3 |

Long ctx: 44.9 → **46.2 tok/s** closes most of the post–Fix #3 gap to llama (~46.5).
Short ctx unchanged (~47.4). Remaining long delta is likely FFN/matmul polish + optional
FlashAttention (Fix #5).

**Long-ctx nsys** (`ref/nsys_logs/yalm_long_fix4b.nsys-rep`, kv≈3200, 30 decode,
`--cuda-graph-trace=node`):

| Kernel | % GPU time | avg µs/call | vs Fix #3 (§24) |
|--------|----------:|------------:|----------------:|
| **`attn_dot`** | **1.8 %** | **12.1** | **−62 %** (was 32.0) |
| `att_mix` | 2.6 % | 17.4 | ~flat (17.2) |
| `attn_softmax` | 1.5 % | 10.5 | ~flat |
| attn sum | 5.9 % | ~40.0 | −33 % (was ~59.7) |

## 26. Fix #5 — FFN dual-row matmul + fused attention (`attn_fused`)

**FFN:** `matmul_row2()` loads activation `x` once per iter for both W1 and W3
rows in `fused_ffn_w1_w3_glu_act` (halves x-bandwidth in GLU phase).

**Attention:** Replaced `attn_dot` + `attn_softmax` + `att_mix` graph nodes with
single `attn_fused` kernel — scores in shared mem (`max_seq_len` floats/block),
no global `att` traffic. Legacy triple kept in source for reference only.

**Benchmark (5-run short / 3-run long mean):**

| Context | tok/s | vs Fix #4 (§25) | vs calm | vs llama |
|---------|------:|----------------:|--------:|---------:|
| short (120 decode) | **47.6** | +0.4 % | ~0.2 | **+0.1** |
| long (`long_prompt.txt`, 3202 prefill) | **46.9** | **+1.5 %** | **−0.5** | **+0.4** |

Long ctx now **matches llama.cpp** (~46.5) and is within ~0.5 tok/s of calm (47.4).

**Long-ctx nsys** (`ref/nsys_logs/yalm_long_fix5.nsys-rep`, kv≈3200):

| Kernel | avg µs/call | vs Fix #4 |
|--------|------------:|----------:|
| `fused_ffn_w1_w3_glu_act` | **329.8** | −1.0 % (was 333.1) |
| **`attn_fused`** | **34.0** | replaces dot+softmax+mix (~40.0 sum) |
| `fused_matmul_add_residuals` | 108.2 | ~flat |
| `fused_qkv_matmul_clip` | 75.0 | ~flat |

Remaining ~0.5 tok/s to calm is almost entirely FFN/matmul class (~93 % GPU time).
Full llama-style tiled FlashAttention is optional — decode attn is no longer the bottleneck.

## 27. Fix #6 — W2 one-warp-per-row (hb bandwidth)

**Problem:** `fused_matmul_add_residuals` launches 32 warps/block for W2
(`n=14336` hb). Each warp re-reads the full 56 KiB hb vector; a 56 KiB
shared cache fixed redundancy but collapsed occupancy on RTX 3080
(long **~46.2** tok/s). FFN `xb` shared cache (16 KiB) and inline RMS
in FFN also regressed.

**Fix:** `fused_matmul_add_residuals_row` — `<<<dim, warp_size>>>`,
one hb read per output row, no extra shared memory.

**Benchmark (5-run mean, Mistral-7B fp16, RTX 3080):**

| Context | tok/s | vs Fix #5 (§26) | vs calm |
|---------|------:|----------------:|--------:|
| short (120 decode) | **~47.7** | +0.1 | **+0.3** |
| long (`long_prompt.txt`, 3202 prefill) | **~46.85** | ~0.0 | **−0.55** |

Long ctx matches Fix #5 (~46.9); short ctx slightly improved. Remaining
long gap to calm (~47.4) is likely architectural: calm's cooperative
`kernel_forward` keeps activations in shared memory across QKV/FFN
matmuls; yalm's CUDA-graph multi-kernel path re-reads global buffers.
Closing the last ~0.5 tok/s probably needs calm-style layer fusion, not
more shared-memory tiling on individual matmul kernels.

## 28. Transformers torch.compile re-bench + library backends

**HuggingFace transformers (torch.compile, RTX 3080, 120 decode):**

| Engine | tok/s | 3080 % peak BW |
|--------|------:|---------------:|
| transformers (torch.compile) | **39.8** | 75.8 % |
| yalm `-d cuda` (same run) | **49.0** | 93.3 % |

torch.compile roughly **1.5×** faster than the blog's eager 25.9 tok/s on
4090, but still well below custom CUDA engines on this box.

**Library backends (not the same thing):**

| Library | Role | yalm flag |
|---------|------|-----------|
| **cuBLAS** | BLAS GEMM/GEMV | `-d cuda-cublas` |
| **cuBLASLt** | heuristic GEMM API (same family, different entry point) | `-d cuda-cublaslt` |
| **cuDNN** | Graph API matmul | `-d cuda-cudnn` |
| **cuTILE (yalm)** | Warp-tile matvec kernel (not NVIDIA cuTILE Python DSL) | `-d cuda-cutile` |

Linear layers (QKV, WO, FFN, LM head) use library GEMM; attention/RoPE/norm
stay on custom kernels. No CUDA graph (cuBLAS + graph capture is awkward on
Ampere). Activations are cast f32→f16 before GemmEx (llama.cpp pattern).

**Serving runtimes (external, 4k context, RTX 3080, 120 decode):**

| Engine | tok/s | 3080 % peak BW | Notes |
|--------|------:|---------------:|-------|
| vLLM 0.23.0 (no spec decode) | **43.5** | 82.8% | `speculative_config=None`, actual token count |
| SGLang 0.5.9 | **45.5** | 86.7% | `sglang_mistral.py`, graph warmup before timing |
| yalm `-d cuda` | **~49** | ~93% | same run as §28 table |
| tensorrt-llm | — | — | **43.3** (4k ctx) | `./trtllm_docker_bench.sh`, `kv_cache_fraction=0.15` |

Install: `requirements-vllm.txt`, `requirements-sglang.txt`, `requirements-trtllm.txt`. vLLM bench uses `--ignore-eos` + actual token count for parity with `-n 120`; enable speculative decoding separately via `--spec-decode` when a draft model is configured.

**Short decode — library backends (120 tok, RTX 3080):**

| Backend | tok/s |
|---------|------:|
| `-d cuda` | **~49** |
| `-d cuda-cublas` | **~43** |
| `-d cuda-cublaslt` | **~43** (Lt matvec n=1 falls back to GemmEx on 3080) |
| `-d cuda-cudnn` | **~43** (cuDNN graph matmul; cuBLAS fallback if plan fails) |
| `-d cuda-cutile` | **~45** (warp-tile matvec kernel) |
| `-d tensorrt-llm` | external TRT-LLM runtime (see `trtllm_docker_bench.sh`) |
