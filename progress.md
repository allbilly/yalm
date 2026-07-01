# Progress log (Jul 2026)

Session goal: enable tinygrad graph fusion on Mistral, clone/build/run **nano-vllm**, add benchmark results to README.

---

## Why the process “crashed” (did not finish)

Nothing segfaulted. The tinygrad bench **died from CUDA VRAM OOM**, not Linux kernel OOM and not agent timeout alone.

### Confirmed: CUDA MemoryError (log exists)

**Log:** `ref/tg_kernels/crash_test.log` (full traceback)

```
MemoryError: Allocation of 224.00 MB failed on CUDA. Used: 18.82 GB
```

On **20 GB RTX 3080**:

| Item | Size |
|------|------|
| Model weights | 14.48 GB |
| VRAM at crash | **18.82 GB** |
| Free headroom | ~1.2 GB |
| Failed alloc | 224 MB (first prefill kernel) |

**Not** kernel OOM: `dmesg` / `journalctl -k` show no `Out of memory: Kill process`.

**Why silent in `bench_fused_120.log`:** only 3 lines printed before prefill; Python exited on `MemoryError` before decode timing. Log shows:

```
loaded weights in 28453.98 ms, 14.48 GB loaded at 0.51 GB/s
fusion: FUSE_QKV=1 FUSE_GLU=1 NO_CB=1
```

### Root cause: post-load `merge_fused_weights`

Old path: load 14.48 GB into **separate** `wq`/`wk`/`wv`/`w1`/`w3` → then `Tensor.cat` + new `Linear` while old weights still on GPU → **~19 GB peak** → 224 MB alloc fails.

**Fix (applied):** `fuse_state_dict()` merges QKV/GLU **before** `load_state_dict`; model init uses `wqkv`/`w13` directly — no post-load merge spike.

### Other failures (nano-vllm install)

**Log:** `ref/nano-vllm/install.log`, `ref/bench_run.log`

| Issue | Detail |
|-------|--------|
| Python 3.14 | Main `.venv` cannot install nano-vllm (`requires <3.13`) |
| torch cu130 vs CUDA 12.6 | flash-attn source build fails: PyTorch cu130 vs nvcc 12.6 |
| Install interrupted | torch cu124 download killed mid-way (~732 MB wheel) |

Use `scripts/install_nano_vllm.sh` with **py3.12 venv** + **torch cu124**.

### Agent timeout (secondary)

Long `BEAM=8` first compile still looks hung; use `nohup` + `PYTHONUNBUFFERED=1`. OOM was the hard failure for tinygrad.


---

## Completed work

### tinygrad fusion (`tinygrad_mistral.py`)

| Change | Purpose |
|--------|---------|
| `merge_fused_weights()` | After HF load: **Q+K+V → one `wqkv` Linear**, **W1+W3 → one `w13` Linear** (matches upstream `WQKV` / yalm fused matmul idea) |
| `NO_CB=1` default | Skip `contiguous_backward()` barriers that block scheduler fusion (fp16 inference) |
| Drop `assign(…).realize()` on KV cache | Remove per-layer sync (`KV_REALIZE=1` to restore old behavior) |
| Remove block-end `.contiguous().contiguous_backward()` | Default off; `BLOCK_CB=1` restores |
| `--no-fuse` | A/B baseline: `FUSE_QKV=0 FUSE_GLU=0 NO_CB=0 KV_REALIZE=1 BLOCK_CB=1` |

**Not verified yet:** no successful timed run after fusion; no `DEBUG=2` kernel-count before/after; README still shows old **35.1 tok/s** unfused number.

### README

- Large section on tinygrad scheduler fusion vs BEAM vs Thunder FA.
- AMD vs CUDA clarification.

### nano-vllm clone + Mistral hook (local patch)

| Path | Status |
|------|--------|
| `ref/nano-vllm/` | Cloned (`git clone --depth 1`) |
| `ref/nano-vllm/nanovllm/models/mistral.py` | **Added** — Llama/Mistral decoder (no Q/K norm), fused QKV + gate/up via existing nano-vllm layers |
| `ref/nano-vllm/nanovllm/engine/model_runner.py` | **Patched** — selects `MistralForCausalLM` when `hf_config.model_type` is `mistral` or `llama` |
| `nano_vllm_mistral.py` | **Added** — Mistral decode bench (120 tok, `ignore_eos`, mirrors `vllm_mistral.py`) |

**Not run:** no nano-vllm throughput number for README.

---

## Still open

1. **Prove tinygrad fusion** — run sequentially (no parallel GPU/pip):
   ```bash
   # fused (defaults)
   FUSE_QKV=1 FUSE_GLU=1 NO_CB=1 BEAM=8 DEV=CUDA \
     .venv/bin/python tinygrad_mistral.py --count 30

   # baseline
   .venv/bin/python tinygrad_mistral.py --count 30 --no-fuse
   ```
   Optional: `DEBUG=2 … --count 1` and compare kernel launch count / names.

2. **Finish nano-vllm install** (Python 3.12 venv only):
   ```bash
   ref/nano-vllm/.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu124
   ref/nano-vllm/.venv/bin/pip install flash-attn --no-build-isolation   # long compile
   ref/nano-vllm/.venv/bin/pip install -e ref/nano-vllm
   ```

3. **Run nano-vllm Mistral bench** (single process, after install):
   ```bash
   ref/nano-vllm/.venv/bin/python nano_vllm_mistral.py --count 120
   ```

4. **Update README** table with fused tinygrad tok/s + nano-vllm row (once measured).

5. **Attention gap** — fusion fixes QKV/GLU/sync; **SDPA → FlashAttention** still needs Thunder FA or upstream pattern match (largest long-context gap). Not started in code.

---

## Recommended next run (avoid repeat “crash”)

1. **One command at a time**; set `block_until_ms` ≥ 600000 (10 min) for first tinygrad BEAM run, or use `BEAM=0` for a quick fusion smoke test then `BEAM=8` for final number.
2. **Do not** start nano-vllm `pip install` while any CUDA benchmark runs.
3. Use **`ref/nano-vllm/.venv` (3.12)** for nano-vllm; never main `.venv` (3.14).
4. Expect **flash-attn** build to dominate install time; run install to completion before benchmarking.

---

## File map

| File | Role |
|------|------|
| `tinygrad_mistral.py` | Fused Mistral + `--no-fuse` |
| `nano_vllm_mistral.py` | nano-vllm Mistral bench wrapper |
| `ref/nano-vllm/` | Upstream + local Mistral model patch |
| `ref/tg_kernels/` | Prior DEBUG=4 / nsys artifacts (unfused baseline) |
| `README.md` | Engine table + tinygrad fusion docs (numbers partially stale) |
| `kernels.md` / `results.md` | Full kernel analysis |
