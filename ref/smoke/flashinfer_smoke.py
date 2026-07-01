#!/usr/bin/env python3
"""flashinfer 1-token decode attention smoke test.

Exercises the real "1 token decode" primitive from the README:
    flashinfer.single_decode_with_kv_cache(q, k, v)
where q is a single query and k/v is a synthetic KV cache.

This is the smallest possible kernel-level smoke. It does NOT touch Mistral
— flashinfer is a kernel library, not a model server, and the 4 reference
repos (per ref/smoke/NOTES.md) do not support Mistral.

Run:
    /home/a/yalm/.venv/bin/python ref/smoke/flashinfer_smoke.py
"""
import os

# flashinfer JIT builds the decode module via ninja + nvcc; make sure both
# are reachable from the subprocess (otherwise FileNotFoundError on ninja).
# The venv is project-local (/home/a/yalm/.venv); `~` resolves to /home/a.
VENV_BIN = "/home/a/yalm/.venv/bin"
os.environ["PATH"] = (
    os.environ.get("PATH", "")
    + os.pathsep + VENV_BIN
    + os.pathsep + "/usr/local/cuda/bin"
)

import torch
import flashinfer

# 1 query token, 32 Q heads, 8 KV heads (GQA 4:1 like Mistral-7B),
# head_dim 128, fp16, on CUDA. This matches the shape family Mistral,
# Qwen, and Llama all use.
NUM_QO_HEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 128
KV_LEN = 256
DTYPE = torch.float16
DEVICE = "cuda"

torch.manual_seed(0)
q = torch.randn(NUM_QO_HEADS, HEAD_DIM, device=DEVICE, dtype=DTYPE)
k = torch.randn(KV_LEN, NUM_KV_HEADS, HEAD_DIM, device=DEVICE, dtype=DTYPE)
v = torch.randn(KV_LEN, NUM_KV_HEADS, HEAD_DIM, device=DEVICE, dtype=DTYPE)

print(f"flashinfer version: {flashinfer.__version__}")
print(f"q={tuple(q.shape)} k={tuple(k.shape)} v={tuple(v.shape)} dtype={DTYPE}")

out = flashinfer.single_decode_with_kv_cache(q, k, v)

# Smoke contract: the kernel produced a tensor of the right shape, dtype,
# and is not all-zero / not NaN. A single all-zero result is the common
# failure mode for a broken launch (illegal address, wrong grid).
assert out.shape == (NUM_QO_HEADS, HEAD_DIM), f"bad shape: {out.shape}"
assert out.dtype == DTYPE, f"bad dtype: {out.dtype}"
assert out.device.type == "cuda", f"bad device: {out.device}"
assert torch.isfinite(out).all(), "NaN/Inf in output"
assert out.abs().sum() > 0, "all-zero output (kernel did not run)"

print(f"output shape={tuple(out.shape)} dtype={out.dtype} "
      f"sum={out.float().abs().sum().item():.4f} "
      f"mean|.|={out.float().abs().mean().item():.6f}")
print("PASS: flashinfer 1-token decode attention")
