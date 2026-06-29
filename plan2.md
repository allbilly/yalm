# plan2.md — Review: are the kernel results enough?

Critical read of `results.md` / `plan.md` §11 against the original goal: explain decode speed differences and drive concrete improvements in **yalm** and **tinygrad**.

**Four engines:** **yalm**, **tinygrad**, **llama.cpp**, **calm**. Speed work targets yalm and tinygrad; comparisons use **all 6 pairwise combinations** (4 choose 2) × short and long context = **12 cells**.

| Role | Engine | Short tok/s |
|------|--------|-------------|
| **Shipping standard** | llama.cpp | 46.62 |
| **Fastest measured** | calm | 47.42 |
| **Hand CUDA (yalm repo)** | yalm | 45.82 |
| **Compiler (BEAM)** | tinygrad | 33.64 |

Each pair needs: (1) tok/s short **and** long, (2) kernel-level attribution, (3) named tests, (4) **generated-kernel comparison** — actual kernel names + source patterns per logical op (§2.0, Tier **K**).

Hardware: RTX 3080 (760 GB/s spec), Mistral-7B-Instruct-v0.2 fp16, short = 120 decode tokens, long = ~4100-token prompt + 30 decode steps.

---

## 2.0 Generated-kernel comparison (required deliverable)

Speed gaps must be explained by **what CUDA actually runs**, not just tok/s. Each engine generates kernels differently; the plan must **capture, name, and compare** them on the **same logical ops** (QKV, FFN W1/W3, FFN W2, attention, LM head).

### How each engine generates kernels

| Engine | Generation model | Where to read kernels | nsys visibility |
|--------|------------------|----------------------|-----------------|
| **yalm** | Hand-written CUDA in `src/infer.cu`; fusion by author | `src/infer.cu`, `./build/test -bk` | ✅ 10+ named kernels (`results.md` §3.1) |
| **calm** | Hand-written cooperative grid (`kernel_forward`) | `ref/calm/src/infer.cu`, `helpers.cuh` (`matmul_warppar`) | ⚠️ 1 blob; **C5 cudaprof** for internal split |
| **llama.cpp** | GGML CUDA template zoo (compile-time instantiation) | `~/llama.cpp/ggml/src/ggml-cuda/` (`mmvq.cuh`, `fattn-*.cuh`) | ✅ 14+ mangled names (`results.md` §3.3) |
| **tinygrad** | **Runtime compiler** + BEAM search over render configs | `DEBUG=4` / `CUDA=1` log → autogen `r_*` CUDA | ❌ not captured yet; **K1** |

### Master kernel map (fill from tests K2, nsys, cudaprof)

One row per **logical op**; all four columns must be filled before claiming a pair is “explained”:

| Logical op (Mistral decode) | yalm (`infer.cu`) | calm (`ref/calm`) | llama.cpp (ggml) | tinygrad (autogen) |
|-----------------------------|-------------------|-------------------|------------------|---------------------|
| Q/K/V proj | `fused_qkv_matmul_clip` | inside `kernel_forward` | `mul_mat_vec_f<half,...>` ×3 | `r_*` matmul ×3 (**K1**) |
| RoPE + KV write | `fused_rope_and_cache_update` | inside `kernel_forward` | RoPE / `k_set_rows` | elem + assign (**K1**) |
| Attn scores + softmax + mix | `attn_dot`, `attn_softmax`, `att_mix` | inside `kernel_forward` | `flash_attn_ext_f16` (+ fixup) | SDPA → `r_*` (**K1**) |
| Attn out proj | `fused_matmul_add_residuals` | inside `kernel_forward` | `mul_mat_vec_f` | `r_*` matmul |
| FFN gate+up (GLU) | `fused_ffn_w1_w3_glu_act` | inside `kernel_forward` | `mul_mat_vec` + `unary_gated_op_kernel<silu>` | w1/silu/w3 chain (**K1**) |
| FFN down | `fused_matmul_add_residuals` | inside `kernel_forward` | `mul_mat_vec_f` | `r_*` matmul |
| LM head | `matmul_wide` | `kernel_output` | `mul_mat_f` + cuBLAS GEMM | `r_*` matmul |
| RMSNorm | `rmsnorm` | inside `kernel_forward` | `rms_norm_f32` | `r_*` reduce |

**Status today:** yalm + llama columns mostly filled from `results.md`; calm column needs **C5**; tinygrad column **empty** until **K1**.

### What to compare per kernel (same op, four implementations)

For each row in the master map, record:

| Field | yalm | calm | llama.cpp | tinygrad |
|-------|------|------|-----------|----------|
| **Kernel name(s)** | nsys / source | cudaprof region / source | nsys mangled name | `r_*` from DEBUG=4 |
| **Avg µs** (short ctx) | nsys §3.1 | C5 / 645 µs÷layer | nsys §3.3 | A1 |
| **Effective GB/s** | `./build/test -bk` | C5 peak BW | infer from bytes÷µs | DEBUG=4 log |
| **Load pattern** | `half2`/`float2` `matmul_row` | `matmul_warppar` half2 | warp GEMV in `mmvq` | **K3** SASS/source skim |
| **Write pattern** | fused / `blocktranspose` | cooperative shared | coalesced GEMV | **K3** |
| **Fusion** | fused kernels | full layer coop | partial (silu separate) | unfused graph |
| **Launch count** / token / layer | ~19.5k graph nodes | ~3 | ~1500 | A8 |

### Tier K — capture & compare generated kernels (ALL pairs)

| # | Pairs | Test | Command / output | Deliverable |
|---|-------|------|------------------|-------------|
| **K1** | ALL | **Capture tinygrad generated CUDA** | `DEBUG=4 BEAM=8 CUDA=1 .venv/bin/python tinygrad_mistral.py --count 5 2>&1 \| tee ref/tg_kernels/debug4.log`; grep `r_`, `opt`, `GB/s`, `CUDA` | Autogen kernel names + reported BW per op |
| **K2** | ALL | **Fill master kernel map** | Merge K1 + `results.md` §3.1/§3.3 + C5 calm split + `src/infer.cu` / `ref/calm` symbols | **`kernels.md`** (or § in results) — 4 columns complete |
| **K3** | YT, YL, YC, TL | **Same-op source diff** | Compare inner loop: yalm `matmul_row` (`infer.cu:233`) vs calm `matmul_warppar` (`helpers.cuh`) vs llama `mmvq.cuh` vs tg CUDA from K1 for **4096×14336 GEMV** | Load/store/coalesce table per pair |
| **K4** | YT, TL | **Same-shape GB/s matrix** | `./build/test -bk matmul\|ffn\|mha` vs tg DEBUG=4 shapes; llama/calm from nsys bytes÷µs | GB/s grid: 4 engines × {QKV, FFN, attn, LM} |
| **K5** | TL, LC | **llama hottest kernel → source** | From nsys top name (e.g. `mul_mat_f<half2,32,...>`) → locate template in `ggml-cuda/` (**C2**) | Link mangled name ↔ `.cuh` file |
| **K6** | YC, LC, TC | **calm cudaprof → logical op %** | **C5** output mapped to master map rows | calm column µs + GB/s per op |
| **K7** | ALL | **ncu / SASS on one GEMV each** | yalm `-bk matmul`; tg single `r_` from K1; llama `mul_mat_vec` if sudo; calm via cudaprof | Optional: DRAM% confirms K3 |

**K2 is the gate:** no pair comparison is complete until its op rows have **kernel names + µs or GB/s** on both sides.

### Pair coverage via kernel comparison

| Pair | Kernel comparison requires |
|------|----------------------------|
| **YL** | yalm names + llama names; K3 matmul loop vs `mmvq`; A5 µs map |
| **YC** | yalm §3.1 + calm C5/K6; K3 vs `matmul_warppar` |
| **YT** | yalm §3.1 + tg K1/A1; **K3 + K4** (hand vs autogen) |
| **TL** | llama §3.3 + tg K1/A1; K5 template lookup |
| **TC** | calm C5/K6 + tg K1; coop+fused vs unfused autogen |
| **LC** | llama §3.3 + calm C5/K6; tensor-core GEMM vs coop warp-GEMV |

---

## 1.1 Four-engine pairwise coverage matrix (6 pairs × 2 contexts)

**Pair tags:** `YL` yalm↔llama, `YT` yalm↔tinygrad, `YC` yalm↔calm, `TL` tinygrad↔llama, `TC` tinygrad↔calm, `LC` llama↔calm.

### Short context

| Pair | Δ tok/s | Δ % | End-to-end | **Generated kernels compared?** | Tests | Enough? |
|------|--------:|----:|:----------:|:---------------------------------:|:-----:|:-------:|
| **yalm vs llama** (YL) | −0.8 | −1.7 % | ✅ | ⚠️ names yes; **K3 loop diff open** | K2,K3,K5,A5 | **Partial** |
| **yalm vs calm** (YC) | −1.6 | −3.4 % | ✅ | ⚠️ yalm yes; calm blob | **K3,K6,C5**,Q1 | **Partial** |
| **yalm vs tinygrad** (YT) | +12.2 | +36 % | ✅ | ❌ tg not captured | **K1,K3,K4**,P1–P4 | **No** |
| **tinygrad vs llama** (TL) | −13.0 | −28 % | ✅ | ❌ tg not captured | **K1,K5**,A1,A7 | **No** |
| **tinygrad vs calm** (TC) | −13.8 | −29 % | ✅ | ❌ tg not captured | **K1,K6**,C5,A1 | **No** |
| **llama vs calm** (LC) | −0.8 | −1.7 % | ✅ | ⚠️ llama names; calm blob | **K5,K6**,C5,Q3 | **Partial** |

### Long context (~4100 prompt, 30 decode)

| Pair | Δ tok/s | End-to-end | **Kernels @4k** | Tests | Enough? |
|------|--------:|:----------:|:---------------:|:-----:|:-------:|
| **yalm vs llama** (YL) | −4.7 | ✅ | ❌ FA vs `att_mix` names only | A4,B1,K2 attn rows | **Partial** |
| **yalm vs calm** (YC) | −5.6 | ✅ | ❌ V layout vs coop attn | B1,C5,K6 | **Partial** |
| **yalm vs tinygrad** (YT) | **?** | yalm ✅ / tg ❌ | ❌ | P3,B3,K1 SDPA rows | **No** |
| **tinygrad vs llama** (TL) | **?** | llama ✅ / tg ❌ | ❌ | A6,C4,K1 FA vs SDPA | **No** |
| **tinygrad vs calm** (TC) | **?** | calm ✅ / tg ❌ | ❌ | A6,C5,K6 | **No** |
| **llama vs calm** (LC) | −0.9 | ✅ | ❌ | A4,C6,K6 | **Partial** |

### Verdict on “all combinations of the 4”

| Question | Answer |
|----------|--------|
| Does the plan list all **6 pairs**? | Yes — §1.1 + §2.4–§2.9 |
| Does plan **compare generated kernels** (§2.0)? | **Yes** — `kernels.md` + `results.md` §12–§23; K1/K6/C5 done; **A1 nsys tg works** (see §23) |
| How many of **12 cells** fully explained (tok/s + kernels)? | **10/12 Yes**, **2/12 Partial** (YL/YC long ctx: Fix #3 V-transpose open) |
| Minimum to cover all 6 pairs | **Done** except K7 (ncu blocked) and C4 (tg long nsys optional confirm) |

---

## 1. Executive verdict

| Pair group | Short ctx | Long ctx | **Generated kernels** | Tests |
|------------|-----------|----------|----------------------|:-----:|
| **yalm ↔ llama** (YL) | **Yes** (post Fix #1+#2+#6) | Partial (attn @4k) | ✅ K2,K3,K5 | Fix #3 V-transpose |
| **yalm ↔ calm** (YC) | **Yes** (µs within 4 %) | Partial (attn @4k) | ✅ K3,K6,C5 | Fix #3 |
| **yalm ↔ tinygrad** (YT) | **Yes** | **Yes** (tg ~4.6 tok/s @3.2k ctx) | ✅ K1,K3,K4,A1 | — |
| **tinygrad ↔ llama** (TL) | **Yes** | **Yes** | ✅ K1,K5,A1,A7 | T4 FA first |
| **tinygrad ↔ calm** (TC) | **Yes** | **Yes** | ✅ K1,K6,C5 | T4 FA first |
| **llama ↔ calm** (LC) | **Yes** | **Yes** (FA vs coop attn) | ✅ K5,K6,C5 | — |

**Bottom line (updated):** The generated-kernel comparison in `kernels.md` + `results.md` §12–§23 **is sufficient** to explain speed gaps and prioritize fixes. Remaining gaps: **K7 ncu/SASS** (blocked), **yalm Fix #3** (V-cache transpose for long ctx), **tinygrad T4** (FlashAttention — tg long ctx is ~4.6 tok/s, not ~26).

---

## 2. What the results successfully explain

### 2.1 End-to-end speed ranking (short context)

| Engine | tok/s | ~% peak BW | Gap to roof (52.5 tok/s) |
|--------|------:|-----------:|-------------------------:|
| calm | 47.42 | 90.3 % | −10 % |
| llama.cpp | 46.62 | 88.8 % | −11 % |
| yalm (post-fix) | 45.82 | 87.2 % | −13 % |
| transformers | 37.91 | 72.2 % | −28 % |
| tinygrad | 33.64 | 64.1 % | −36 % |

Roofline framing is sound: decode is weight-bandwidth-bound; engines at 87–90 % are within ~10–15 % of the theoretical ceiling. The **rank order** is explained qualitatively:

- **calm / llama.cpp / yalm** — hand-tuned or fused GEMV, CUDA graphs or cooperative kernel, mature KV layouts.
- **transformers** — unfused PyTorch ops, framework overhead.
- **tinygrad** — compiler-generated generic kernels; BEAM search cannot match hand fusion on bandwidth-bound GEMV.

### 2.2 yalm vs llama.cpp — short context (0.8 tok/s gap)

| Metric | llama.cpp | yalm (post-fix) | Δ |
|--------|----------:|----------------:|--:|
| tok/s | 46.62 | 45.82 | **−1.7 %** |
| ~% peak BW | 88.8 % | 87.2 % | −1.6 pp |
| Long ctx tok/s | 46.5 (flat) | 41.85 | **−10.1 %** |

The short-context story vs llama.cpp is mostly complete:

1. **nsys (both sides)** — yalm: 94 % matmul-class; llama.cpp: ~92 % in `mul_mat_*` + cuBLAS GEMM. Different decomposition but same roofline class.
2. **Validated fix** — yalm `matmul_row` vectorization closed much of the gap; yalm is now within ~2 % of llama.cpp at short ctx (`results.md` §1).
3. **Structural llama.cpp advantages still open:**

| llama.cpp kernel / feature | % time (short) | yalm equivalent | yalm % | Gap implication |
|----------------------------|---------------:|-----------------|-------:|-----------------|
| `mul_mat_f` + `mul_mat_vec_f` (GEMV) | ~80 % combined | `fused_ffn_*`, `fused_matmul_*`, `fused_qkv_*` | ~94 % | yalm FFN fusion is good but FFN `-bk` = 40 % GB/s vs plain matmul 73 % — write coalescing |
| `ampere_h16816gemm_*` (tensor cores) | ~32 % | none (warp GEMV only) | 0 % | LM head + some attn via cuBLAS; yalm LM head = 1.5 % — **small short-ctx factor** |
| `flash_attn_ext_f16` | ~1.7 % | `attn_dot`+`softmax`+`att_mix` | ~3 % | Negligible at short ctx; **dominant at long ctx** |
| `unary_gated_op_kernel<silu>` (separate GLU) | 0.3 % | fused into FFN (50 % time) | 50 % | llama.cpp splits GLU; yalm fuses — fusion is correct, inner matmul still slow |

**What is missing vs llama.cpp specifically:**

- No **per-op µs table** mapping yalm kernel → llama.cpp kernel for the same layer (e.g. yalm `fused_ffn` 340 µs vs llama `mul_mat_vec`+`silu` combined).
- No **ncu** on llama.cpp `mul_mat_vec_f` vs yalm `matmul_row` to confirm both hit similar DRAM % after the vectorization fix.
- Tensor-core 32 % looks large in nsys but much of it is LM head / batched paths — **not yet decomposed** into "how many tok/s that 32 % is worth" for decode.

### 2.3 yalm vs llama.cpp — long context (~4.7 tok/s gap, −10 %)

llama.cpp is the right standard here: it stays **flat** (46.5 vs 46.62 short); yalm drops **8.7 %** (45.82 → 41.85).

Partially explained:

- **llama.cpp:** FlashAttention (`flash_attn_ext_f16` + fixup) scales with KV; nsys at short ctx already names the kernels.
- **yalm:** `-bk att_mix` at kv_len=4096 = 6644 µs × 32 layers ≈ 213 ms/token, ~5 GB/s — V-cache layout `(seq_len, n_kv_heads, head_dim)` is strided; llama.cpp FA avoids the separate `att_mix` pattern entirely.

**Missing vs llama.cpp (critical):**

| Test | Status | Why it matters |
|------|--------|----------------|
| llama.cpp **nsys at kv_len ≈ 4000** | ❌ | Confirm FA % rises; get target µs/layer for yalm to match |
| yalm **nsys at kv_len ≈ 4000** | ❌ | Prove `att_mix` (or triple) dominates wallclock, not extrapolation |
| Side-by-side **attention µs/layer** yalm vs llama | ❌ | Fix #4 (FlashAttention) needs a numeric target from the standard |
| llama.cpp **kernel source skim** (`mmvq`, `fattn-common.cuh`) vs yalm `att_*` | Planned in plan.md §7D, not in results | Port path should reference ggml, not only calm |

calm's V-transpose layout is a valid *implementation hint*, but the **product target** is llama.cpp behavior: flat decode tok/s from ~4k ctx. Fix #3 (V transpose) and fix #4 (FA) should both be validated against **llama.cpp long-context nsys**, not calm totals.

### 2.4 llama.cpp vs calm (LC) — short −0.8 tok/s; calm faster

Fastest two engines; gap is small but **unexplained at kernel level on the calm side**.

| Metric | calm | llama.cpp | Δ |
|--------|-----:|----------:|--:|
| tok/s short | 47.42 | 46.62 | calm **+0.8 (+1.7 %)** |
| tok/s long | 47.4 | 46.5 | calm **+0.9** |
| ~% peak BW short | 90.3 % | 88.8 % | +1.5 pp |
| nsys short | 1 blob (`kernel_forward` 98.5 %) | 14 kernels decomposed | asymmetric |
| nsys long | ❌ | ❌ | |

**What results explain (LC short):**

- Both flat at long context — mature KV + attention paths on both sides.
- llama: tensor-core cuBLAS (~32 %), FlashAttention, many `mul_mat_vec` variants.
- calm: one cooperative `kernel_forward` ≈ **645 µs/layer** all-in (`results.md` §3.2); ~3 kernel launches/token vs llama ~1500.
- calm wins likely from: cooperative grid (no per-op launch tax), transposed V cache, tuned `matmul_warppar` — **source in `ref/calm/`**, not measured vs llama per-op.

**Missing for LC (tests Q3, C5, C6):**

| Test | Status | Answers |
|------|--------|---------|
| calm **`cudaprof.cu`** internal split | ❌ C5 | matmul vs attn % inside `kernel_forward` — compare to llama §3.3 |
| llama ↔ calm **µs/layer budget** | ❌ Q3 | Where does +0.8 tok/s come from? |
| Both **long nsys @4k** | ❌ C6, A4 | Confirm both stay flat; FA vs calm internal attn |
| ncu / GB/s calm `matmul_warppar` vs llama `mul_mat_vec` | ❌ | Same-shape GEMV throughput |

### 2.5 yalm vs calm (YC) — short −1.6 tok/s; long −5.6 tok/s

yalm is ported from calm ideas; calm remains the **in-repo reference** (`ref/calm/src/infer.cu`).

| Metric | calm | yalm (post-fix) | Δ |
|--------|-----:|----------------:|--:|
| tok/s short | 47.42 | 45.82 | **−1.6 (−3.4 %)** |
| tok/s long | 47.4 | 41.85 | **−5.6 (−11.8 %)** |
| nsys | `kernel_forward` blob | 10+ named kernels | yalm decomposable |
| µs/layer (all-in) | ~645 µs | sum of parts in §3.1 | **Q1** compares |

**What results explain (YC short, partial):**

1. **Source diff done** — `matmul_row` scalar loads vs calm `matmul_warppar` half2/float2; vectorization fix +6.4 % (`results.md` §5, §7).
2. **Open gaps** — FFN/QKV uncoalesced writes (yalm 40 % GB/s FFN in `-bk`); calm fuses entire layer cooperatively.
3. **Long context** — calm flat; yalm drops 8.7 %. calm has transposed V; yalm `att_mix` strided — `-bk` shows 6.6 ms at kv=4096.

**Missing for YC (tests Q1, Q2, C5):**

| Test | Status | Answers |
|------|--------|---------|
| **Q1** yalm nsys µs sum vs calm 645 µs/layer | ❌ | Quantify −1.6 tok/s by op |
| **Q2** calm cudaprof vs yalm §3.1 kernel classes | ❌ C5 | matmul/attn/norm split both sides |
| yalm long nsys @4k vs calm flat tok/s | ❌ B1 | Prove V-layout / no-FA story |
| Port calm V-transpose → measure YC long gap closure | fix #3 | Target: yalm 47.4 at 4k |

**Note:** llama.cpp remains the **shipping standard** for product targets; calm is the **in-repo performance ceiling** for hand-written CUDA.

### 2.6 calm — nsys limitation (affects YC, LC, TC)

calm appears as **`kernel_forward`** (98.5 % time, ~20.6 ms/token) in nsys. Per-kernel comparison to yalm/llama/tinygrad **requires `ref/calm/tools/cudaprof.cu` (test C5)** — planned in original `plan.md` but **not run**. Until C5: calm pairs rely on end-to-end tok/s + source diff + yalm-side decomposition only.

### 2.7 tinygrad vs llama.cpp (TL) — short −13.0 tok/s, −28 %

This is the **largest gap among the three “chaser” pairs** (YL, YC, TL). llama.cpp has full nsys (`results.md` §3.3); tinygrad has **only tok/s**.

| Metric | llama.cpp | tinygrad (BEAM=8) | Δ |
|--------|----------:|------------------:|--:|
| tok/s | 46.62 | 33.64 | **−13.0 tok/s (−27.8 %)** |
| ~% peak BW | 88.8 % | 64.1 % | **−24.7 pp** |
| vs yalm (post-fix) | −0.8 tok/s | −12.2 tok/s | tinygrad is **15× further** from standard than yalm |
| Long ctx tok/s | **46.5** (measured) | **?** (not run) | Cannot compare flat-curve behavior |

**Roofline read:** tinygrad achieves ~64 % of theoretical weight-bandwidth vs llama.cpp's ~89 %. That 25 pp gap is far too large to be launch overhead alone — it implies autogen GEMV/attention kernels are **systematically underfilling DRAM**, plus extra traffic from unfused ops and temporaries.

#### Op-by-op map (one decode step, one layer)

From `tinygrad_mistral.py` vs llama.cpp nsys kernel classes — **structure only, µs not measured on tinygrad side**:

| Logical op | tinygrad (Python graph) | llama.cpp (nsys) | Structural disadvantage |
|------------|-------------------------|------------------|---------------------------|
| Embed | `tok_embeddings` | (host/graph) | minor |
| Pre-attn norm | `attention_norm` (RMSNorm) | `rms_norm_f32` (1.2 %) | likely similar |
| Q/K/V proj | **3×** `nn.Linear` (`wq`,`wk`,`wv`) | `mul_mat_vec_f` variants (~32 % combined) | 3 launches + 3 weight reads vs tuned GEMV; no QKV fusion |
| RoPE | `apply_rotary_emb` (elem) | RoPE kernels (small) | fused in graph, not hand-tuned |
| KV write | `cache_kv.assign(...).realize()` | `k_set_rows` etc. | **`.realize()` = hard sync** each token |
| GQA expand | `repeat_kv` on K/V | handled inside FA/GEMV | **extra KV traffic** (32 heads from 8) |
| Attention | `scaled_dot_product_attention` | `flash_attn_ext_f16` (1.7 % short) | generic SDPA, **not FlashAttention**; scales badly at long ctx |
| Attn out proj | `wo` Linear | `mul_mat_vec_f` | same GEMV class as QKV |
| Residual | `x + ...` | fused in ggml graphs | extra elem kernel + temp buffer |
| Pre-FFN norm | `ffn_norm` | `rms_norm_f32` | similar |
| FFN gate/up | `w1(x).silu() * w3(x)` | `mul_mat_vec` + `unary_gated_op_kernel<silu>` (0.3 %) | **unfused**: 3 matmuls + silu + mul vs llama's split but tuned GEMV |
| FFN down | `w2(...)` | `mul_mat_vec_f` | GEMV class |
| Layout fixes | `.contiguous().contiguous_backward()` × several | minimal converts (1.5 %) | **forced layout copies** in hot path |
| LM head | `output(norm(h))` | `mul_mat_f` + cuBLAS GEMM (~46 % combined) | no tensor-core path |
| Decode JIT | `TinyJit` (1 tok/step) | graph / batched launches | comparable amortization |

**Per-layer launch budget (estimate):** llama.cpp ~14 kernel names, ~30–50 launches/token/layer in nsys aggregate. tinygrad likely **many more** `r_*` kernels (matmul + elem + reduce + copy) × 32 layers — exact count needs A1.

#### Hypothesized −13 tok/s budget (needs nsys to confirm)

Qualitative split until A1/A2 run; percentages are ** guesses** to prioritize tests:

| Hypothesis | Est. share of gap | llama.cpp has | tinygrad lacks | Confirm with |
|------------|------------------:|---------------|----------------|--------------|
| Autogen GEMV <60 % peak BW | **40–50 %** | `mul_mat_vec` ~80 %+ effective | BEAM-tuned generic `r_` matmul | A2 DEBUG=4 GB/s; map to llama µs |
| Unfused FFN/QKV (extra passes + temps) | **20–25 %** | tuned per-op GEMV | w1/silu/w3/w2 chain | A1 kernel count; A7 µs map |
| Generic SDPA vs FlashAttention | **10–15 %** short; **dominant long** | `flash_attn_ext_f16` | `scaled_dot_product_attention` | A1 + A6 long ctx |
| `repeat_kv` + bad KV layout | **5–10 %** | FA/GQA-aware | explicit repeat | A1; compare at kv_len=4k |
| `.realize()` / sync in KV update | **5–10 %** | async cache write | sync per token (`line 74`) | nsys CPU/GPU gaps |
| No tensor cores (LM head + some attn) | **5–10 %** | cuBLAS ~32 % nsys | none | A1 top kernels |
| `contiguous_backward()` copies | **5 %** | rare converts | multiple per layer | A1; DEBUG=4 |

**Important:** these are not measured — they explain *why profiling is urgent*, not *why tinygrad is slow*.

#### What we can say without tinygrad nsys

1. **End-to-end:** tinygrad is 28 % slower than the standard; 24 pp lower roofline utilization.
2. **Architecture:** the Python graph in `tinygrad_mistral.py` is strictly **less fused** than llama.cpp's ggml CUDA for the same Mistral graph (separate Q/K/V, unfused GLU, generic SDPA, explicit syncs).
3. **Compiler path:** BEAM=8 searches render configs but cannot invent QKV fusion or FlashAttention — those require graph changes or upstream tinygrad.
4. **Not the same problem as yalm:** yalm is −1.7 % from llama.cpp with identified kernel fixes; tinygrad is −28 % with **zero kernel names**. Improving yalm patterns in tinygrad (port `infer.cu`) is premature until A1 shows where time goes.

#### tinygrad vs llama.cpp — long context

| | llama.cpp | tinygrad |
|--|-----------|----------|
| tok/s at ~4k ctx | **46.5** | **not measured** |
| Expected behavior | flat (FA scales) | likely **large drop** (generic SDPA O(n²) traffic + repeat_kv) |

Long-context tinygrad bench is **mandatory** (test A6). Without it we cannot know if tinygrad's gap vs llama.cpp widens like yalm's (−10 %) or worse.

**Missing vs llama.cpp (tinygrad-specific):**

| Test | Status | Why it matters |
|------|--------|----------------|
| tinygrad **nsys short ctx** | ❌ | Name `r_*` kernels; % time vs llama §3.3 |
| tinygrad **DEBUG=4 GB/s** | ❌ | Compare autogen matmul to llama `mul_mat_vec` throughput |
| tinygrad **long ctx tok/s** | ❌ | vs llama 46.5 — does gap grow to 40 %+? |
| tinygrad **long ctx nsys** | ❌ | SDPA % vs llama FA at 4k |
| **µs map** tinygrad op → llama kernel | ❌ | Turn hypotheses into fix list |
| **Kernel launches/token** | ❌ | Quantify fusion gap |
| BEAM sweep vs llama tok/s | ❌ (C3 planned) | Rule out "wrong BEAM" before graph changes |

### 2.8 yalm vs tinygrad (YT) — direct pair (+12.2 tok/s short, +36 %)

Third pairwise comparison. Previously **missing as its own section** — only mentioned as “tinygrad is 15× further from llama than yalm.”

| Metric | yalm (post-fix) | tinygrad (BEAM=8) | Δ |
|--------|----------------:|------------------:|--:|
| tok/s short | 45.82 | 33.64 | **yalm +12.2 tok/s (+36 %)** |
| ~% peak BW | 87.2 % | 64.1 % | **+23 pp** for yalm |
| tok/s long (~4k) | 41.85 | **?** | not measured |
| nsys short | ✅ §3.1 | ❌ | one-sided |
| nsys long | ❌ | ❌ | neither |

**What we can infer without direct tg profile:**

| Factor | yalm | tinygrad | Explains part of +12.2? |
|--------|------|----------|:------------------------:|
| Kernel fusion | FFN, QKV, matmul+residual fused in `infer.cu` | Unfused Python graph (`tinygrad_mistral.py`) | **Yes** (qualitative) |
| GEMV implementation | Hand warp-GEMV, vectorized loads | BEAM autogen `r_*` | **Likely** — needs P2 GB/s |
| Attention | 3 small kernels (short ctx) | Generic SDPA | **Yes at long ctx**; small short |
| CUDA graphs / JIT | CUDA graphs | TinyJit | Similar intent |
| Sync points | minimal | `cache_kv.realize()` per token | **Likely** — needs P1 timeline |
| Tensor cores | none | none | neutral vs each other |

**What we cannot infer (needs direct pairwise tests P1–P4):**

- Whether yalm's advantage is **mostly fusion** or **better autogen-vs-hand GEMV throughput**
- Per-op µs: yalm `fused_ffn` 340 µs vs tinygrad equivalent matmul chain
- Launch count ratio: yalm ~19.5k graph nodes vs tinygrad `r_*` count
- Long-context: does yalm's +12.2 tok/s **shrink or grow** at 4k? (yalm drops; tg unknown)

**Deriving from the other two pairs is insufficient:**

```
yalm − llama  = −0.8 tok/s   (mostly explained)
tg   − llama  = −13.0 tok/s  (unexplained)
─────────────────────────────
yalm − tg     = +12.2 tok/s  (by arithmetic)

But: attributing the +12.2 to specific kernels requires tg nsys (A1) AND
side-by-side diff (P1), not subtraction of two incomplete explanations.
```

**Missing vs tinygrad (yalm-specific direct tests):**

| Test | Status | Answers |
|------|--------|---------|
| Side-by-side nsys (same prompt, 120 tok) | ❌ P1 | Kernel class % diff: fusion vs launches vs GEMV |
| Same-shape GB/s: yalm `-bk` vs tg DEBUG=4 | ❌ P2 | 4096×14336 GEMV, FFN, MHA isolation |
| Long ctx tok/s both | ❌ P3 (in B3) | +12.2 at short → ? at 4k |
| **Diff table** yalm kernel ↔ tg `r_*` | ❌ P4 | Fill after A1 + existing yalm §3.1 |

### 2.9 tinygrad vs calm (TC) — short −13.8 tok/s, −29 %

**Largest absolute gap** in the 4-engine set (calm fastest, tinygrad slowest among the four).

| Metric | calm | tinygrad (BEAM=8) | Δ |
|--------|-----:|------------------:|--:|
| tok/s short | 47.42 | 33.64 | **−13.8 (−29 %)** |
| ~% peak BW | 90.3 % | 64.1 % | **−26 pp** |
| tok/s long | 47.4 | **?** | not measured |
| nsys | 1 blob | ❌ | neither decomposed |

**Qualitative story (needs A1 + C5):**

- calm: cooperative kernel, fused layer, tuned warp-GEMV, transposed V — **~90 % peak BW**.
- tinygrad: unfused graph, autogen kernels, generic SDPA — **~64 % peak BW**.
- TC gap ≈ TL gap + LC gap in tok/s (−13.8 ≈ −13.0 + −0.8) but **cannot attribute by subtraction** — calm and llama optimize differently (coop grid vs tensor-core ggml).

**Missing for TC:**

| Test | Pairs | Status |
|------|-------|--------|
| tinygrad nsys | TC, TL, YT | ❌ A1 |
| calm cudaprof | TC, YC, LC | ❌ C5 |
| Long ctx tok/s calm 47.4 vs tg | TC | ❌ A6, B3 |
| tg ↔ calm **roofline %** side-by-side | TC | ❌ A2 + C5 |

**Improvement target for tinygrad vs calm:** not necessarily match calm (different stack) — but **explain** which of {GEMV, fusion, FA, sync} accounts for the 26 pp BW gap.

---

## 3. What the results do **not** explain

### 3.1 calm-involved pairs (YC, LC, TC)

| Pair | Short explained? | Long explained? | Blocker |
|------|:----------------:|:---------------:|---------|
| **yalm vs calm** (YC) | Partial — source diff, yalm nsys | Partial — tok/s only | C5 cudaprof, Q1 µs/layer |
| **llama vs calm** (LC) | Partial — llama nsys, calm blob | Partial — tok/s flat both | C5, Q3 |
| **tinygrad vs calm** (TC) | **No** | **No** | A1 + C5 |

### 3.2 yalm vs tinygrad (YT)

See §2.8. Without P1–P4, the **+12.2 tok/s** advantage is assumed but not measured op-by-op.

### 3.3 tinygrad vs llama (TL)

See §2.7. tinygrad has **no nsys** — largest hole among llama-involved pairs.

### 3.4 ncu / hardware metrics — blocked, not substituted

`ERR_NVGPUCTRPERM` prevented ncu. The report substitutes:

- nsys kernel times
- arithmetic GB/s from `-bk` tests

**Not confirmed:**

- Uncoalesced store warnings (L1TEX→L2)
- DRAM throughput % per kernel
- Warp stall breakdown
- SASS unroll/prefetch behavior on FP16 KV paths

The FFN write-coalescing hypothesis (§5.2 in `results.md`) is **source-pattern inference**, not measured. High confidence from the blog + llama.cpp `mul_mat_vec` coalescing patterns, but ncu would de-risk before a structural GLU refactor.

### 3.5 Phase B comparison matrix — 4 engines

| Kernel class | yalm | calm | llama.cpp | tinygrad |
|--------------|------|------|-----------|----------|
| Matmul/GEMV % | ~94 | ~98* | ~92 | **?** |
| Attention % | ~3 | * | ~2 | **?** |
| Norm/elem % | ~1.3 | * | ~1.2 | **?** |
| Launch overhead | low (graphs) | low (coop) | moderate | **?** |
| Top-1 kernel | `fused_ffn_*` | `kernel_forward` | `mul_mat_f` | **?** |
| Short tok/s | 45.82 | **47.42** | 46.62 | 33.64 |

\* calm internal split needs **C5 cudaprof**; tinygrad row needs **A1**.

**Cross-pair use:** one filled table supports explanation of all **6 pairs** (compare columns).

### 3.6 Other gaps (all pairs)

| Gap | Pairs affected | Test |
|-----|----------------|------|
| **Master kernel map empty (tg column)** | ALL | **K1, K2** → `kernels.md` |
| **Same-op source diff not done** | YL, YC, YT, TL | **K3** |
| **Same-shape GB/s grid** | YT, TL, YC | **K4** |
| llama mangled name → `.cuh` | TL, LC | **K5, C2** |
| calm op-level µs in map | YC, LC, TC | **K6, C5** |
| Memory BW ceiling (`-b`, `-b2`) | all | C1 |
| llama long nsys @4k | YL, LC, TL | A4, C6 |
| calm cudaprof | YC, LC, TC | **C5** |
| tinygrad nsys | YT, TL, TC | **A1** |
| yalm long nsys @4k | YL, YC, YT | B1 |
| Long tok/s all **4** engines | all long cells | **B3** |
| yalm ↔ llama µs map | YL | A5 |
| yalm µs sum vs calm 645 µs/layer | YC | **Q1** |
| llama ↔ calm µs budget | LC | **Q3** |

---

## 4. Sufficiency for code improvement

### 4.1 yalm — **proceed now**

| Fix | Evidence strength | Ready to implement? |
|-----|-------------------|---------------------|
| **#1 FFN GLU write coalescing** (`blocktranspose`) | Strong — 50 % GPU time, 40 % GB/s in `-bk`, pattern exists in `fused_matmul_add_residuals` | **Yes** — highest ROI |
| **#2 QKV clip write coalescing** | Strong — 11 % time, same anti-pattern | **Yes** |
| **#3 V cache transpose OR FlashAttention** | Strong for long ctx — `-bk` + llama.cpp FA reference | **Yes**, but validate vs **llama.cpp long nsys** (B1b), not calm |
| **#4 FlashAttention** (replace attn triple) | llama.cpp has working FA in production | Design against ggml `fattn-*`; **primary long-ctx path to match standard** |
| **#5 Tensor-core LM head** | llama.cpp uses cuBLAS ~32 % nsys; yalm 1.5 % | Defer — unlikely to explain 0.8 tok/s gap |

**Expected closure vs llama.cpp:** #1+#2 → parity at short ctx (~46.5 tok/s); #3 or #4 → close long-context gap (target: 46.5 tok/s flat like llama.cpp).

### 4.2 tinygrad vs llama.cpp — profile first, then pick layer

**Target:** 46.62 tok/s short (+13 tok/s), 46.5 tok/s long (flat like llama.cpp). Current: 33.64 — **only 73 % of standard speed**.

Improvement paths depend on A1/A2 split vs llama.cpp. Do **not** assume the same fixes as yalm.

| Priority | Action | When | vs llama.cpp |
|----------|--------|------|--------------|
| **T0** | Run A1, A2, A6, C3 | Before any code | Establish kernel % and long-ctx gap |
| **T1** | BEAM / env tuning | If C3 shows headroom | Cheapest; may gain 1–3 tok/s |
| **T2** | Graph fusion in `tinygrad_mistral.py` | If A1 shows many small kernels between GEMVs | Fuse QKV (like yalm `fused_qkv`), fuse GLU (like yalm `fused_ffn`); llama keeps silu separate but GEMV is fast |
| **T3** | Remove sync hot spots | If nsys shows gaps at `cache_kv.realize()` | Match llama async KV write pattern |
| **T4** | Replace SDPA with FA path | If A6 shows gap widening at 4k | Match llama `flash_attn_ext_f16`; may need upstream tinygrad or custom kernel |
| **T5** | Upstream tinygrad issue | If A2 shows autogen matmul consistently <50 % peak | File with DEBUG=4 trace; compare to ggml `mmvq` |
| **T6** | Custom CUDA / Metal | Last resort | Only if T1–T5 ceiling proven |

**Do not** port yalm `infer.cu` into tinygrad until A7 maps which llama.cpp kernel class (GEMV, FA, silu) tinygrad is missing throughput on.

**Success metrics (report vs llama.cpp only):**

| Milestone | tok/s short | tok/s long (~4k) | % peak BW |
|-----------|------------:|-----------------:|----------:|
| Today | 33.64 | ? | 64 % |
| After T0–T1 | ? | ? | aim 70 %+ |
| After T2–T3 | ? | ? | aim 80 %+ |
| Parity | **≥46.5** | **≥46** | **≥88 %** |

### 4.3 yalm vs tinygrad — explain after P1–P4

Do not port yalm kernels into tinygrad blindly. Sequence:

1. **P1 + A1** — side-by-side nsys; classify time into GEMV / attn / elem / sync
2. **P2** — if yalm `-bk` GB/s ≫ tg DEBUG=4 on same dims → tg compiler issue (T5 upstream)
3. **P2** — if GB/s similar but tg has 3× launches → fusion issue (T2 in `tinygrad_mistral.py`)
4. **P3** — long ctx: if gap shrinks, tg attn/SDPA is the lever; if gap grows, both need FA

| If P1/P2 shows… | Action |
|-----------------|--------|
| tg GEMV ≪ yalm `-bk` on FFN shape | Upstream tinygrad / BEAM (T5); not yalm port |
| tg many kernels, similar GB/s | Fuse QKV/GLU like yalm (T2) |
| tg timeline gaps at `.realize()` | Remove sync (T3) |
| yalm `-bk att_mix` ≪ tg SDPA at 4k | Both need FA; yalm fix #4, tg T4 |

---

## 5. Recommended additional tests

Prioritized by information gain per hour. Tests tagged by pair(s) and whether they fill **§2.0 master kernel map**.

**Tier K — generated kernel capture & compare:** full table in **§2.0** (K1–K7). Run **K1→K2 first** — everything else hangs off the master map.

**Pair key:** `YL` yalm↔llama, `YT` yalm↔tinygrad, `YC` yalm↔calm, `TL` tinygrad↔llama, `TC` tinygrad↔calm, `LC` llama↔calm, `ALL` = all four.

### Tier Q — calm-involved pairs (YC, LC, TC)

| # | Pairs | Test | Command / method | Answers |
|---|-------|------|------------------|---------|
| **Q1** | YC | **yalm µs sum vs calm all-in** | Sum yalm §3.1 avg µs × instances per token ÷ 32 layers; compare to calm **645 µs/layer** | Where −1.6 tok/s lives |
| **Q2** | YC | **calm cudaprof vs yalm kernel classes** | Build/run `ref/calm/tools/cudaprof.cu` | matmul/attn/norm % inside calm vs yalm §3.1 |
| **Q3** | LC | **llama ↔ calm µs budget** | llama §3.3 per-class µs vs C5 calm internal split | Where calm +0.8 tok/s vs llama |
| **C5** | YC, LC, TC | **calm cudaprof** (same as Q2) | Per `ref/calm/tools/cudaprof.cu` | **Unblocks all calm kernel attribution** |
| **C6** | LC, YL, TL | **calm + llama long nsys** | Same prompt as A4/B1; calm still 1 blob but confirms flat + total µs | Long LC/YC context |

### Tier P — yalm vs tinygrad (YT)

| # | Pairs | Test | Command / method | Answers |
|---|-------|------|------------------|---------|
| **P1** | YT, ALL | **Side-by-side nsys short** | Same prompt, 120 tok: yalm (`results.md` §3.1 exists) + A1 tinygrad; diff kernel-class % | +12.2 tok/s: fusion vs GEMV vs launches |
| **P2** | YT | **Same-shape GB/s** | yalm `./build/test -bk {matmul,ffn,mha}` vs tg **K1** log; see also **K4** | Hand CUDA vs autogen throughput |
| **P3** | YT, ALL | **Long ctx tok/s both** | `mistral_bench.py` / manual: 4100 prompt, 30 decode, separate subprocesses | +12.2 at short → ? at 4k |
| **P4** | YT, ALL | **Kernel diff table** | Subset of **K2** master map (yalm + tg columns) | YT op-aligned rows |

### Tier A — tinygrad vs llama.cpp (priority block)

All outputs use **llama.cpp §3.3** as the reference column.

| # | Pairs | Test | Command / method | Answers |
|---|-------|------|------------------|---------|
| A1 | TL, YT, ALL | **tinygrad nsys (short ctx)** | Warmup `BEAM=8 .venv/bin/python tinygrad_mistral.py --count 120`; then `nsys profile --trace=cuda --stats=true BEAM=8 .venv/bin/python tinygrad_mistral.py --count 120` | `r_*` % vs llama §3.3; enables P1/P4 |
| A2 | TL, YT | **tinygrad DEBUG=4** | Same as **K1** | Autogen CUDA + GB/s; feeds K2,K3,K4 |
| A6 | TL, TC, YT, ALL | **tinygrad long ctx** | Same 4100-token prompt; 30 decode | TC, TL, YT long cells |
| A7 | TL | **Fill §2.7 µs map (tg↔llama)** | A1 + llama §3.3 | TL −13 tok/s budget |
| A8 | TL, YT, LC | **Launch count / token** | nsys all 4 where available | calm ~3 vs llama ~1500 vs yalm ~19.5k vs tg ? |
| A3 | ALL | **Phase B table (4 engines)** | A1 + C5 + llama + yalm §3.1 | All **6 pairs** from one table |
| A4 | YL, LC, TL, ALL | **llama long nsys** | `nsys profile llama-cli -c 4096`, `long_prompt.txt`, `-n 30` | Long attn budget |
| A5 | YL | **yalm ↔ llama µs map** | Spreadsheet from existing nsys | YL −0.8 tok/s |

### Tier B — yalm vs llama long-context + fixes

| # | Pairs | Test | Command / method | Answers |
|---|-------|------|------------------|---------|
| B1 | YL, YC, YT, ALL | **yalm nsys long ctx** | `long_prompt.txt`, `-n 30` | YL, YC long attn |
| B1b | YL, LC, TL, ALL | **llama nsys long ctx** | Same as B1 / A4 | Standard for all chasers |
| B2 | YL, YC | **yalm `-bk ffn` post fix #1** | `./build/test -bk ffn` | Close YL + YC short gap |
| B3 | **ALL 4 engines** | **Long ctx tok/s** | 4100 prompt, 30 decode, subprocess each | **Fill all 6 long cells** |
| B4 | YL | **ncu yalm FFN** | `sudo ncu ... ./build/test -bk ffn` | Coalesce proof |
| B6 | YL, YT | **yalm `-bk mha` kv=4096** | `./build/test -bk mha` | yalm attn cost; compare P2 tg SDPA |

### Tier C — calibration and reference

| # | Test | Command | Answers |
|---|------|---------|---------|
| C1 | **Host/GPU BW ceiling** | `./build/test -b && ./build/test -b2` | Realistic peak for % calculations |
| **B5** | **ncu llama.cpp `mul_mat_vec_f` vs yalm matmul** | ncu on both `-bk matmul` paths | Standard GEMV DRAM % comparison |
| C2 | YL, LC | **ggml source skim** | `~/llama.cpp/ggml/src/ggml-cuda/mmvq.cuh`, `fattn-common.cuh` | llama side of LC, YL |
| C3 | TL, TC | **BEAM sweep tinygrad** | `BEAM=1,4,8,16` × 10 runs vs llama **and** calm tok/s | TC/TL ceiling |
| C4 | TL, TC | **tinygrad long nsys** | After A6, same prompt as C6 | SDPA vs calm/llama @4k |

### Tier D — after fixes land

| # | Test | When |
|---|------|------|
| D1 | Full `mistral_bench.py --runs 10` yalm only | After fix #1 |
| D2 | Long-context regression (4100 prompt) | After fix #3 |
| D3 | `./build/test` unit suite | Every kernel change |
| D4 | Sanity generation output | Every kernel change |

---

## 6. Test execution order (suggested)

```
Week 1 — generated kernels + all 6 pairs
  K1 → K2 → K3 → K4       (capture tg CUDA; fill kernels.md; source + GB/s)
  C5, K6, Q1–Q3           (calm cudaprof → calm column in map)
  A1 → P1 → P4 → A3       (tg nsys µs; 4-engine Phase B)
  A6 → P3 → B3             (long tok/s all 4)
  A4, B1, B1b, C6         (long nsys)
  K5, A7, A8, C3          (llama template lookup; launches; BEAM)

Week 2 — yalm closes YL + YC (validate in kernels.md rows)
  Fix #1 FFN coalesce → B2, B4, B5, D1; re-run K4 on FFN row

Week 3 — long ctx (update K2 attn rows @4k)
  yalm fix #3/#4; tinygrad T2–T4; C4 tg long nsys
```

---

## 7. Answers — all six pairwise combinations (4 engines)

### Is the result enough to explain the speed difference?

| Pair | Short Δ | Long Δ | Enough? | Key tests |
|------|--------:|-------:|:-------:|-----------|
| **yalm vs llama** (YL) | +0.6 (post-fix) | −1.9 | **Yes** short / Partial long | Fix #3 V-transpose |
| **yalm vs calm** (YC) | −0.18 | −2.8 | **Yes** short / Partial long | Fix #3 |
| **yalm vs tinygrad** (YT) | +13.6 | **+40 tok/s** (@3.2k ctx) | **Yes** | tg SDPA + unfused GLU |
| **tinygrad vs llama** (TL) | −13.0 | **−42 tok/s** (@3.2k ctx) | **Yes** | T4 FA, T2 fusion |
| **tinygrad vs calm** (TC) | −13.8 | **−43 tok/s** (@3.2k ctx) | **Yes** | T4 FA, T2 fusion |
| **llama vs calm** (LC) | −0.8 | −0.9 | **Yes** | coop launch tax only |

### Minimum tests (tok/s **and** generated kernels)

| # | Delivers | Test |
|---|----------|------|
| 1 | **Master kernel map** | **K1 → K2** (`kernels.md`: 4 engines × 8 ops) |
| 2 | Source/load patterns | **K3** (matmul inner loops) |
| 3 | Same-shape GB/s | **K4** + P2 |
| 4 | calm column in map | **C5, K6** |
| 5 | tg µs in map | **A1** |
| 6 | llama name→source | **K5, C2** |
| 7 | All 6 pairs long tok/s | **B3** |
| 8 | Direct yalm↔tg | **P1, P4** |

**Optional:** C1, C2, C3, B5 ncu, D1–D4.

---

## 8. Risk register

| Risk | Mitigation |
|------|------------|
| FFN coalesce refactor breaks GLU numerics | `./build/test` + greppable sanity output |
| `-bk att_mix` extrapolation wrong at full model | B1 nsys at 4k ctx |
| tinygrad −13 tok/s is SDPA not matmul | A1 + A6; compare attn % to llama FA |
| tinygrad gap widens >40 % at 4k | C4; prioritize T4 (FA) over T2 (fusion) |
| yalm−tg gap assumed to be fusion without P1 | P1 side-by-side nsys |
| Subtracting YL and TL to explain YT | P1–P4 direct measurement |
| calm cudaprof blocked / not built | Q1 yalm-side µs sum vs 645 µs/layer only |
| LC +0.8 tok/s attributed without C5 | Q3 waits on C5 |
| ncu stays blocked | `-bk` GB/s + nsys µs for yalm; document caveat |
| tg autogen GEMV uninspectable | **K1** DEBUG=4 log + save to `ref/tg_kernels/` |
| Speed gap without kernel name on both sides | **K2** gate — row incomplete until both columns filled |
| OOM in combined bench harness | Separate subprocesses per engine |

---

## 9. References

- **Deliverable:** **`kernels.md`** — master map from §2.0 (K2 output)
- **tinygrad autogen:** `ref/tg_kernels/debug4.log` from **K1**
- Full tok/s: `results.md` §3.1 (yalm), §3.2 (calm), §3.3 (llama)
- yalm source: `src/infer.cu`
- calm source: `ref/calm/src/infer.cu`, `ref/calm/tools/cudaprof.cu`
- llama source: `~/llama.cpp/ggml/src/ggml-cuda/`
- tinygrad graph: `tinygrad_mistral.py`
- Bench: `mistral_bench.py`, `tinygrad_mistral.py`, calm `~/calm/build/run`
