# YALM

Re-run all engines and print the comparison table (RTX 3080, Mistral-7B-Instruct-v0.2):

```
pip install -r requirements.txt
.venv/bin/python mistral_bench.py
```

HF weights: pass `--model mistralai/Mistral-7B-Instruct-v0.2` (resolved via `huggingface_hub` from `~/.cache/huggingface/hub`, no snapshot hash in config). Local dir: `--weights /path` or `MISTRAL_PATH`.

Script: `mistral_bench.py` — all engines use **`--runs 10`** by default (warmup + timed runs; vLLM/SGLang/TRT-LLM load the model once and run 10 timed decodes in-process). Tinygrad uses `tinygrad_mistral.py` (pip install tinygrad; no repo clone).

Comparison vs blog (Mistral-7B-Instruct-v0.2 fp16, 4k context, prompt "Q: What is the meaning of life?", 120 generated tokens, **`mistral_bench.py --runs 10`** unless noted):

Card peak memory bandwidth: RTX 4090 = 1008 GB/s (24GB GDDR6X, 384-bit, 21.0 Gbps [videocardz](https://videocardz.net/nvidia-geforce-rtx-4090)). RTX 3080 = 760 GB/s (20GB GDDR6X on this box, 320-bit, 19.0 Gbps [videocardz](https://videocardz.net/nvidia-geforce-rtx-3080)). For each engine, "BW used" = model size (14.48 GB fp16) × tok/s, i.e. the minimum bytes that must move through DRAM per token for a fully memory-bandwidth-bound decode.

| Engine                       | RTX 4090 tok/s (blog) | 4090 % peak BW | RTX 3080 tok/s (this box) | 3080 % peak BW |
| ---------------------------- | --------------------- | -------------- | ------------------------- | -------------- |
| huggingface transformers    | 25.9                  | 37.2%          | —                         | —              |
| transformers (torch.compile) | —                     | —              | **39.8** (10-run avg)     | **75.8%**      |
| llama.cpp                    | 61.0                  | 87.6%          | ~46.5 long / ~48.4 short  | ~88–92%        |
| calm                        | 66.0                  | 94.8%          | **~48.9** (matched long)  | **~93%**       |
| yalm (`-d cuda`)             | 63.8                  | 91.6%          | **~49.3 short / ~48.9 long** | **~93%**    |
| yalm (`-d cuda-coop`)        | —                     | —              | **~49.3 short / ~48.9 long** | **~93%**    |
| yalm (`-d cuda-cublas`)      | —                     | —              | **~43**                   | ~82%           |
| yalm (`-d cuda-cudnn`)       | —                     | —              | **~43**                   | ~82%           |
| yalm (`-d cuda-cutile`)      | —                     | —              | **~45**                   | ~87%           |
| vllm (no spec decode)        | —                     | —              | **43.8** (10-run avg, stdev ~0.02) | **83.4%**      |
| sglang                       | —                     | —              | **43.6** (10-run avg, stdev ~0.19) | **83.0%**      |
| tensorrt-llm                 | —                     | —              | **43.2** (10-run avg, stdev ~0.08) | **82.3%**      |
| nano-vllm                    | —                     | —              | **43.2** (10-run avg, stdev ~0.02) | **82.3%**      |
| tinygrad (fused)             | —                     | —              | **44.3** (10-run avg, stdev ~0.03) | **84.5%**      |

Log: `ref/tg_kernels/bench_serving_10run.log` (vLLM/SGLang), `ref/tg_kernels/bench_beam8_10run.log` (tinygrad), `ref/tg_kernels/bench_trtllm_10run.log` (TRT-LLM), `ref/nano-vllm/bench_mistral_10run.log` (nano-vllm).


**RTX 3080 serving runtimes (120 tok decode, `mistral_bench.py --runs 10`, Jul 2026):**

| Engine | tok/s | Notes |
|--------|------:|-------|
| vLLM 0.23.0 (no spec decode) | **43.8** | 10-run avg (stdev ~0.02); `speculative_config=None`; `ignore_eos` |
| SGLang 0.5.9 | **43.6** | 10-run avg (stdev ~0.19); `--graph-warmup` before timed runs |
| yalm `-d cuda` | **~49.3** | custom CUDA graph |
| calm | **~48.9** | cooperative kernel |
| tensorrt-llm 1.3.0rc19 | **43.2** | 10-run avg (stdev ~0.08); Docker; 4k ctx, `kv_cache_fraction=0.15` |
| nano-vllm 0.2.0 | **43.2** | 10-run avg (stdev ~0.02); `ref/nano-vllm/.venv` py3.12 + flash-attn 2.7.4 |
| tinygrad (fused, BEAM=3) | **44.3** | 10-run avg (stdev ~0.03); `FUSE_QKV=1 FUSE_GLU=1 NO_CB=1` |
| tinygrad (fused, BEAM=8) | **44.3** | 10-run avg (stdev ~0.02); was 35.1 unfused |

**RTX 3080 decode (Mistral-7B fp16, Jun 2026, yalm/calm/llama.cpp):**

| Context | Prompt | yalm `-d cuda` | yalm `-d cuda-coop` | calm | llama.cpp |
|---------|--------|---------------:|--------------------:|-----:|----------:|
| Short | 120 tok decode | **~49.3** | **~49.3** | ~47.4 | ~46.5 |
| Long | `long_prompt.txt` (~3202 prefill + 30 decode) | **~48.9** | **~48.9** | **~48.9** | ~46.5 |

All three native CUDA paths (yalm graph, yalm coop, calm) are **within ~1%** on this box — near the ~52 tok/s memory roofline (760 GB/s ÷ 14.48 GB weights). nsys: one `kernel_forward` ≈ **20.4 ms**/token at kv≈3200 (`ref/nsys_logs/yalm_coop_long.nsys-rep`, `calm_long.nsys-rep`).

**CUDA backends**

| `-d` flag | What it runs |
|-----------|----------------|
| `cuda` (default) | Per-kernel CUDA graph (`fused_ffn`, `attn_fused`, …) |
| `cuda-coop` | calm-style cooperative `kernel_forward` (one launch/token) |
| `cuda-cublas` | Linear layers via cuBLAS `GemmEx`; custom attn/RoPE/norm |
| `cuda-cublaslt` | Linear layers via cuBLASLt (falls back to GemmEx on matvec n=1) |
| `cuda-cudnn` | Linear layers via cuDNN graph matmul (falls back to cuBLAS if unsupported) |
| `cuda-cutile` | Linear layers via warp-tile matvec kernel (not NVIDIA cuTILE Python DSL) |
| `tensorrt-llm` | NVIDIA TensorRT-LLM via `trtllm_mistral.py` (separate runtime; `.yalm` checkpoint unused) |
| `cpu` | CPU reference |

Library backends need `pip install -r requirements.txt` (cuDNN comes from the `nvidia-cudnn` wheel). TensorRT-LLM: see `requirements-trtllm.txt` and [install docs](https://nvidia.github.io/TensorRT-LLM/installation.html).

**vLLM / SGLang**

Use **separate uv venvs** — do not install into yalm's `.venv` (CUDA/PyTorch wheel conflicts). Run **one install at a time**. Python **3.12** recommended.

**vLLM** ([install docs](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/)):

```
uv venv --python 3.12 ~/.cache/uv/environments-v2/vllm-bench/.venv
uv pip install --python ~/.cache/uv/environments-v2/vllm-bench/.venv/bin/python -r requirements-vllm.txt
```

`ninja` is required for FlashInfer JIT at first run — it must be on `PATH` (the venv `bin/` is prepended automatically by `mistral_bench.py`).

On this box (20 GB VRAM), `vllm_mistral.py` sets `max_model_len=4096` so fp16 weights (~14.5 GB) plus KV cache fit.

Verify:

```
.venv/bin/python bench_engines.py vllm
~/.cache/uv/environments-v2/vllm-bench/.venv/bin/python vllm_mistral.py --count 8
```

**SGLang** ([install docs](https://docs.sglang.io/docs/get-started/install)):

```
uv venv --python 3.12 ~/.cache/uv/environments-v2/sglang-bench/.venv
uv pip install --python ~/.cache/uv/environments-v2/sglang-bench/.venv/bin/python -r requirements-sglang.txt
```

First install pulls large CUDA wheels (~2 GB); expect several minutes. `sglang_mistral.py` uses `context_length=4096`.

Verify:

```
.venv/bin/python bench_engines.py sglang
~/.cache/uv/environments-v2/sglang-bench/.venv/bin/python sglang_mistral.py --count 8
```

**CUDA 12 hosts:** SGLang defaults to CUDA 13 wheels. If import or kernel load fails, follow the [CUDA 12 override steps](https://docs.sglang.io/docs/get-started/install) (`sglang-kernel` from `docs.sglang.ai/whl/cu129/`).

**Docker (optional, for serving):** official images `vllm/vllm-openai` and `lmsysorg/sglang:latest-runtime` — see each project's Docker docs. The bench scripts here use in-process Python APIs, not the server containers.

**nano-vllm** (Python 3.12 only — separate venv, not main `.venv`):

```
bash scripts/install_nano_vllm.sh
.venv/bin/python mistral_bench.py --engines nano-vllm --count 120 --runs 10
```

Uses `ref/nano-vllm/.venv` (torch cu124 + flash-attn 2.7.4 wheel; FA 2.8 breaks on torch 2.6+cu124 ABI).

**TensorRT-LLM (Docker bench, podman on this box)**

Image: `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc19`. On this box `docker` is podman — GPU needs **`--runtime=/usr/bin/nvidia-container-runtime`** (handled automatically by `mistral_bench.py`).

Verify:

```
podman run --rm --runtime=/usr/bin/nvidia-container-runtime \
  nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc19 \
  python3 -c "import tensorrt_llm; print('ok')"
```

Benchmark (first run builds the TRT engine; several minutes):

```
.venv/bin/python mistral_bench.py --engines tensorrt-llm --trtllm-docker --count 120 --runs 10
```

On this GPU (20 GB), TRT-LLM defaults to reserving 90% of free VRAM for KV (~38k tokens) and OOMs after loading ~14 GB weights. `trtllm_mistral.py` caps with `KvCacheConfig(max_tokens=4096, free_gpu_memory_fraction=0.15)`.

**SGLang timing:** `sglang_mistral.py` runs one untimed full decode before the timed run (`--graph-warmup`, default on) so CUDA graph capture is excluded from tok/s.

Bench via `mistral_bench.py` (auto-finds uv venvs under `~/.cache/uv`):

```
.venv/bin/python mistral_bench.py --engines vllm --count 120
.venv/bin/python mistral_bench.py --engines sglang --count 120
```

Override Python if needed: `--vllm-python PATH` / `--sglang-python PATH`.

Example:
```
./build/main mistral-7b-instruct-fp16.yalm -d tensorrt-llm -m completion -i "Q: What is the meaning of life?" -n 120
./build/main mistral-7b-instruct-fp16.yalm -d cuda-cudnn -m completion -i "Q: What is the meaning of life?" -n 120
./build/main mistral-7b-instruct-fp16.yalm -d cuda-coop -m completion -f long_prompt.txt -n 30
```

Full kernel comparison, bottleneck analysis, and per-kernel timing breakdown from `nsys`/`-bk` tests: see [`results.md`](results.md).
Commands run on this box:
- transformers (torch.compile): `.venv/bin/python mistral_bench.py --engines "huggingface transformers" --count 120`
- tensorrt-llm: `.venv/bin/python mistral_bench.py --engines tensorrt-llm --trtllm-docker --count 120 --runs 10` (auto-Docker if native import fails)
- vllm: `.venv/bin/python mistral_bench.py --engines vllm --count 120` (uses `vllm_mistral.py`; auto-finds uv venv)
- nano-vllm: `bash scripts/install_nano_vllm.sh` then `.venv/bin/python mistral_bench.py --engines nano-vllm --count 120 --runs 10`
- sglang: `.venv/bin/python mistral_bench.py --engines sglang --count 120` (uses `sglang_mistral.py`)
- yalm graph: `./build/main mistral-7b-instruct-fp16.yalm -d cuda -m completion -i "Q: What is the meaning of life?" -n 120`
- yalm library backends: `-d cuda-cublas`, `-d cuda-cudnn`, `-d cuda-cutile`, …
- yalm coop: `./build/main mistral-7b-instruct-fp16.yalm -d cuda-coop -m completion -f long_prompt.txt -n 30`
- llama.cpp: `~/llama.cpp/build/bin/llama-cli -m ~/mistral-7b-instruct-v0.2.fp16.gguf ...`
- calm long (matched workload): `calm/build/run ~/.cache/mistral-7b-instruct.fp16.calm -c 4096 -n 3232 -i - < long_prompt.txt` (3202 prompt + 30 decode steps; **not** `-n 30` alone)
- tinygrad: `FUSE_QKV=1 FUSE_GLU=1 NO_CB=1 BEAM=3 DEV=CUDA .venv/bin/python tinygrad_mistral.py --count 120` (defaults on; `BEAM=8` same tok/s, slower compile; `--no-fuse` for old path)

### tinygrad: exported kernels, profiler, and BEAM limits

Full 4-engine kernel map: [`kernels.md`](kernels.md). Raw artifacts: `ref/tg_kernels/`.

**Fastest on this box:** yalm/calm ~**49 tok/s** (~93% DRAM roofline) vs tinygrad ~**44.3 tok/s** (~85%) — slightly ahead of vLLM (**43.8**) and SGLang (**43.6**, all 10-run avg). Remaining ~5 tok/s gap to yalm is mostly **SDPA + no full CUDA graph**, not BEAM tile tuning.

#### Yes — tinygrad auto-fuses kernels (scheduler, not BEAM)

tinygrad **does** fuse ops automatically. This is separate from BEAM:

| Layer | Where | What it does |
|-------|--------|--------------|
| **Scheduler fusion** | `tinygrad/schedule/rangeify.py` → `remove_bufferize` | Merges UOp chains into one `CALL` / one GPU kernel by **removing intermediate buffers** |
| **BEAM** | `tinygrad/codegen/opt/search.py` | Picks `UPCAST`/`LOCAL`/… **inside** each already-scheduled kernel |
| **Hand kernels** | `extra/thunder/{amd,cuda}/fa.py`, `Tensor.custom_kernel` | Explicit FlashAttention / GEMM — not inferred from SDPA |

Official docs: [developer intro](https://docs.tinygrad.org/developer/developer/) — *“One `CALL` is one kernel; the scheduler breaks the graph into subgraphs that fit in a kernel.”* tinygrad README demo: matmul+broadcast+sum → **one kernel** via laziness.

**How scheduler fusion works** (`rangeify` / `remove_bufferize` in `ref/tinygrad/tinygrad/schedule/rangeify.py`):

1. **`realize_map`** — ops that must materialize to DRAM (`COPY`, `CONTIGUOUS`, `ASSIGN`, multi-consumer splits, …) insert `BUFFERIZE` → **fusion stop**.
2. **`remove_bufferize`** — if cheap, drop the temp buffer and keep compute in one kernel (recompute vs reload tradeoff).
3. **`PCONTIG`** — aggressiveness for partial contiguous fusion across reduces (`helpers.py`; `PCONTIG=2` used in FA-related tests).

Use **`VIZ=1`** or the viz README (*“see why two kernels didn't fuse”*) to debug boundaries.

**What auto-fuses today** (from `ref/tinygrad/test/`):

- Elementwise chains: `exp`, `+`, `*` without `.realize()` between them.
- `matmul → relu → … → softmax` in one chain (`test_softmax_fusion.py::test_fuse_gemm_softmax`).
- Double matmul `a@b@c` when rangeify allows (`test_double_matmul`, needs `RANGEIFY>1` for full single-kernel path — several fusion tests are `@unittest.skip("needs RANGEIFY>1")`).

**What does *not* become one FlashAttention-style kernel**:

- **`scaled_dot_product_attention`** — schedule tests expect **4 kernels** (Q@K matmul, softmax path, @V matmul, plus pieces): `test_scaled_dot_product_attention_fusion` / `_causal_fusion` in `test/null/test_schedule.py`.
- **Softmax alone** — default path is **3 kernels**; single-kernel softmax needs a manual reshape trick (`single_kernel_softmax` in `test_softmax_fusion.py`) or future `RANGEIFY>1` (`test_auto_softmax` is skipped).
- **Three `nn.Linear` (Q/K/V)** — three separate weight tensors → three matmul kernels; scheduler cannot merge unrelated `PARAM`s into yalm’s `fused_qkv`.
- **Mistral model code breaks fusion on purpose** — `extra/models/llama.py`:
  - `w3(x.contiguous_backward())` — comment: *“fixes a strange fusion that makes tensor cores miss”*
  - `cache_kv[…].assign(…).realize()` — KV write is an explicit sync point every token
  - `.contiguous().contiguous_backward()` after each block

Maintainers (2023): FA is **not inferred** from lazy graph; SDPA is ~6 ops and needs explicit fusion rules ([tinygrad#1505](https://github.com/tinygrad/tinygrad/discussions/1505)).

So: **auto-fuse yes**, but it fuses **generic UOp subgraphs**, not **LLM-specific algorithms** (fused QKV, GLU+SiLU, tiled FA). Our nsys capture (~48% time in `r_1792_*` SDPA) is consistent with the **4-kernel SDPA schedule**, not missing BEAM on AMD.

#### Export generated CUDA + run tinygrad’s profiler

tinygrad’s built-in profiler is **`DEBUG=4`**: it prints each compiled kernel’s **full CUDA source**, BEAM-chosen `Opt(...)` tuple, and **DRAM/L2 GB/s** per launch.

```bash
# Export kernels + GB/s (also writes CUDA into the log)
DEBUG=4 BEAM=8 DEV=CUDA .venv/bin/python tinygrad_mistral.py --count 5 \
  2>&1 | tee ref/tg_kernels/debug4_full.log

# Summarize kernel names → avg µs / GB/s (parse the *** NV lines)
grep '^\*\*\* NV' ref/tg_kernels/debug4_full.log | \
  awk '{print $3, $NF}' | sort | uniq -c   # manual; see kernel_stats.txt

# Steady-state % time (authoritative for decode graph)
BEAM=8 DEV=CUDA nsys profile --trace=cuda --output=ref/tg_kernels/nsys_tg_short \
  .venv/bin/python tinygrad_mistral.py --count 30
```

**What got exported** (`ref/tg_kernels/`):

| File | Contents |
|------|----------|
| `debug4_full.log` | Full `extern "C" __global__` sources for `E_*` / `r_*` kernels + `opts: (UPCAST, LOCAL, …)` chosen by BEAM |
| `kernel_stats.txt` | Aggregated avg µs and DRAM GB/s per kernel name |
| `nsys_tg_short.nsys-rep` | Steady-state decode: **~48% SDPA**, **~35% FFN** (`E_458752_32_4`) |

Example FFN kernel BEAM picked for W1/W3 (`debug4_full.log`):

```
E_458752_32_4  opts: (UPCAST axis=0 arg=4, LOCAL axis=0 arg=32)
*** NV  E_458752_32_4  … 660|660 GB/s   (isolated microbench)
```

Same kernel in full decode graph (`nsys`): **364 µs, ~323 GB/s effective, 23% GPU time** — contention + unfused neighbors, not a bad single-kernel pick.

Steady decode captures **`JIT GRAPHing batch with 32 kernels`** (TinyJit): ~14 distinct `r_*`/`E_*` launches per token vs yalm’s **~10 fused types × 32 layers** in one CUDA graph, or calm’s **one `kernel_forward`**.

#### Why BEAM cannot find yalm / llama.cpp kernels

BEAM (`ref/tinygrad/tinygrad/codegen/opt/search.py`) is **not** searching over kernel algorithms. It only applies a fixed menu of **`OptOps`** to an already-lowered generic matmul/reduce graph:

| BEAM can tune | BEAM cannot invent |
|---------------|-------------------|
| `UPCAST`, `UNROLL`, `LOCAL`, `GROUP`, `THREAD`, `SWAP`, optional `TC` | Fused QKV (`fused_qkv_matmul_clip`) |
| Block/grid sizes for **one** `Scheduler` op | Fused FFN W1+W3+SiLU (`fused_ffn_w1_w3_glu_act`) |
| Pick min **single-kernel** wall time (3 timed runs, 10s compile timeout) | FlashAttention / split-K FA (`flash_attn_ext_f16`) |
| Cache hits on `(ast.key, amt, device)` | Warp-per-row GEMV with `blocktranspose` writes |
| | Cooperative 32-layer grid (`kernel_forward`) |
| | llama.cpp’s **template zoo** (`mul_mat_vec_f`, `mmf.cu`, cuBLAS LM head) |

**yalm** and **llama.cpp** win because the optimal decode path is **chosen at graph design time** (hand fusion + layout + FA templates), not by autotuning tile sizes on unfused `Linear → silu → mul` chains.

Concrete mismatches on Mistral decode (see `kernels.md` master map):

| Op | yalm / llama.cpp | tinygrad (BEAM=8) |
|----|------------------|-------------------|
| QKV | 1 fused launch, ~75 µs | 1× `r_6144_*` with `FUSE_QKV=1` (was 3× `r_4096_*`) |
| FFN GLU | 1 fused launch, ~334 µs, 327 GB/s | 1× `w13` + SiLU with `FUSE_GLU=1` (was `E_458752_32_4` + silu chain) |
| Attention | ~14 µs short (custom); FA at long ctx | **~48% GPU time** in generic SDPA (`r_1792_8_4_256_1024_4_4`); **4.6 tok/s** at 3.2k ctx |
| LM head | `matmul_wide` ~366 µs, 611 GB/s | `r_16000_16_2_32_8` ~364 µs but **22%** of token time |

Isolated GEMV can hit **660–714 GB/s** in DEBUG=4 — close to calm — but unfused end-to-end was **~67% peak BW**; with QKV/GLU fusion + `NO_CB=1` it reaches **~85%** (44.3 tok/s, 10-run avg). BEAM still optimizes **kernels in isolation**, not **tokens/sec through the full graph**.

**Fusion (Jul 2026):** `FUSE_QKV=1 FUSE_GLU=1 NO_CB=1` in `tinygrad_mistral.py` → **44.3 tok/s** (10-run, stdev ~0.03) vs **35.1** unfused — **+9.2 tok/s (+26%)**. Log: `ref/tg_kernels/bench_beam8_10run.log`. QKV weights are interleaved per KV head (`_pack_wqkv`), matching the forward reshape.

**BEAM sweep (C3, fused, Jul 2026):** `BEAM=3` vs `BEAM=8`, **`--runs 10`** each — both **44.3 tok/s** (stdev ~0.03 vs ~0.02; ~0.0 swing, noise). Use **BEAM=3** for faster compile. Logs: `ref/tg_kernels/bench_beam3_10run.log`, `ref/tg_kernels/bench_beam8_10run.log`.

#### AMD vs CUDA: Thunder FA is not scheduler auto-fusion

AMD can look like it “auto-fuses to FlashAttention” because tinygrad ships **hand-written** FA there — but that is a **third path**, not what `scaled_dot_product_attention` uses:

| Path | What happens | Default Mistral? |
|------|----------------|------------------|
| **Scheduler fusion** | `remove_bufferize` merges UOp chains; SDPA still **4 kernels** | Yes (partial) |
| **BEAM** | Tile tuning per kernel | Yes |
| **Thunder** `extra/thunder/amd/fa.py` | `custom_kernel` + WMMA → **one HIP FA kernel** | **No** — opt-in, D=128, causal |
| **Thunder** `extra/thunder/cuda/fa.py` | Kittens CUDA FA | **No** |

Same on CUDA and AMD for the default model: scheduler + BEAM on unfused SDPA. Thunder FA is the AMD (and CUDA) equivalent of wiring llama.cpp’s `flash_attn_ext_f16` — **explicit swap**, not beam width.

```python
# To use real FA (check GQA, mask, head_dim constraints):
from extra.thunder.amd.fa import flash_attention  # DEV=AMD
# attn, _, _ = flash_attention(q, k, v, is_causal=True)
```

#### How to improve BEAM (tinygrad-side)

Changes that would actually move tok/s toward yalm (~49), in priority order:

1. **Score full decode steps, not single kernels** — beam `_time_program()` times one launch; Mistral decode needs a cost model over the **fused graph** (or at least the TinyJit replay), including L2 residency of 14.48 GB weights.

2. **Add fusion to the search space** — new rewrite rules before BEAM: `Linear×3 → fused_qkv`, `Linear+silu+Linear → fused_glu`, `SDPA → flash_attn` when head_dim/seq match templates (mirror llama.cpp `fattn-*.cuh`).

3. **Mistral-shaped GEMV template** — for `M=1, K=4096, N=14336`, include a **warp-per-row** candidate (yalm `matmul_row` / ggml `mul_mat_vec_f`) alongside generic tiled matmul; BEAM’s `TC`/upcast path targets larger batch matmuls, not decode GEMV.

4. **Attention in BEAM budget** — today SDPA lowers to many `r_*` ops; beam picks locals per sub-kernel while missing that **FA is a different algorithm**, not a tile size.

5. **Longer search / better cache key** — `BEAM_TIMEOUT_SEC=10`, `BEAM_UOPS_MAX=3000`, and cache keyed on `ast.key` miss graph-level context (prefill vs decode, kv length). `BEAM_MIN_PROGRESS=0.01µs` stops early when tile tweaks plateau.

6. **Local workaround (this repo)** — **done:** fuse QKV/GLU + skip `contiguous_backward` in `tinygrad_mistral.py` (T2 in `results.md` §23) → **44.3 tok/s**. Remaining **~5 tok/s** to yalm needs FA + graph capture.

```bash
# Optional: watch BEAM pick opts live
BEAM_DEBUG=1 DEBUG=2 BEAM=8 DEV=CUDA .venv/bin/python tinygrad_mistral.py --count 1
```

- Ranking on this 3080 (Jul 2026): **yalm `-d cuda` ≈ calm > tinygrad (fused) > vLLM (no spec) ≈ sglang > tensorrt-llm > library backends > llama.cpp > transformers (torch.compile)**. Custom kernels ~49 tok/s; tinygrad ~44.3; vLLM ~43.8; SGLang ~43.6; TRT-LLM ~43.2 (all 10-run avg). See [`results.md`](results.md) §28.

TensorRT-LLM
```
docker pull nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc19
docker run --rm -it --ipc host --gpus all --ulimit memlock=-1 --ulimit stack=67108864 -p 8000:8000 nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc19
```

run profile
```
/usr/local/cuda-12.6/nsight-systems-2024.4.2/target-linux-x64/nsys profile --trace=cuda,nvtx,osrt --stats=true ./build/main tinyllama.yalm  -i "Q: What is meaning of life in the age of AGI, give a long ans" -d cuda > profile.out



===============
Original README.md

yalm (Yet Another Language Model) is an LLM inference implementation in C++/CUDA, using no libraries except to load and save frozen LLM weights.
- This project is intended as an **educational exercise** in performance engineering and LLM inference implementation. 
- The codebase therefore emphasizes documentation, whether external or in comments, scientific understanding of optimizations, and readability where possible. 
- It is not meant to be run in production. See [limitations](#limitations) section at bottom.
- See my blog post [Fast LLM Inference From Scratch](https://andrewkchan.dev/posts/yalm.html) for more.

Latest benchmarks with Mistral-7B-Instruct-v0.2 in FP16 with 4k context, on RTX 4090 + EPYC 7702P:

| Engine      | Avg. throughput (~120 tokens) tok/s | Avg. throughput (~4800 tokens) tok/s |
| ----------- | ----------- | ----------- |
| huggingface transformers, GPU | 25.9 | 25.7 |
| llama.cpp, GPU | 61.0 | 58.8 |
| calm, GPU | 66.0 | 65.7 |
| yalm, GPU | 63.8 | 58.7 |

# Instructions

yalm requires a computer with a C++20-compatible compiler and the CUDA toolkit (including `nvcc`) to be installed. You'll also need a directory containing LLM safetensor weights and configuration files in huggingface format, which you'll need to convert into a `.yalm` file. Follow the below to download Mistral-7B-v0.2, build `yalm`, and run it:

```
# install git LFS
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get -y install git-lfs
# download Mistral
git clone git@hf.co:mistralai/Mistral-7B-Instruct-v0.2
# clone this repository
git clone git@github.com:andrewkchan/yalm.git

cd yalm
pip install -r requirements.txt
python convert.py --dtype fp16 mistral-7b-instruct-fp16.yalm ../Mistral-7B-Instruct-v0.2/
make && ./build/main mistral-7b-instruct-fp16.yalm -i "What is a large language model?" -m c
```

# Usage

See the CLI help documentation below for `./build/main`:

```
Usage:   main <checkpoint> [options]
Example: main model.yalm -i "Q: What is the meaning of life?" -m c
Options:
  -h Display this help message
  -d [cpu,cuda,cuda-coop,cuda-cublas,cuda-cublaslt,cuda-cudnn,cuda-cutile] which device to use (default - cuda)
  -m [completion,passkey,perplexity] which mode to run in (default - completion)
  -T <int> sliding window context length (0 - max)

Perplexity mode options:
  Choose one:
    -i <string> input prompt
    -f <filepath> input file with prompt
Completion mode options:
  -n <int>    number of steps to run for in completion mode, default 256. 0 = max_seq_len, -1 = infinite
  -t <float> temperature (default - 1.0)
  Choose one:
    -i <string> input prompt
    -f <filepath> input file with prompt
Passkey mode options:
  -n <int>    number of junk lines to insert (default - 250)
  -l <int>    passkey position (-1 - random)
```

# Tests and benchmarks

yalm comes with a basic test suite that checks implementations of attention, matrix multiplications, feedforward nets in the CPU and GPU backends. Build and run it like so:

```
make test
./build/test
```

The test binary also includes benchmarks for individual kernels (useful for profiling with `ncu`) and broader system tools such as 2 benchmarks to determine main memory bandwidth:

```
# Memory benchmarks
./build/test -b
./build/test -b2

# Kernel benchmarks
./build/test -k [matmul,mha,ffn]
```

# Limitations

- Only completions may be performed (in addition to some testing modes like computing perplexity on a prompt or performing a [passkey test](https://github.com/ggerganov/llama.cpp/pull/3856)). Chat interface has not been implemented.
- An NVIDIA GPU is required.
- The GPU backend only works with a single GPU and the entire model must fit into VRAM.
- As of Dec 31, 2024 only the following models have been tested:
  - Mistral-v0.2 
  - Mixtral-v0.1 (CPU only)
  - Llama-3.2

# Acknowledgements

- [calm](https://github.com/zeux/calm) - Much of my implementation is inspired by Arseny Kapoulkine’s inference engine. In a way, this project was kicked off by “understand calm and what makes it so fast.” I’ve tried to keep my code more readable for myself though, and as much as possible scientifically understanding optimizations, which means foregoing some advanced techniques used in calm like dynamic parallelism.
- [llama2.c](https://github.com/karpathy/llama2.c) - Parts of the CPU backend come from Andrej Karpathy’s excellent C implementation of Llama inference.