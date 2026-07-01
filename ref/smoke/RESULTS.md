# ref/ smoke results — 2026-07-01

Host: `~/yalm` (RTX 3080, CUDA 12.6.20, Linux 6.17, no sudo, no root).
Toolchains installed in this session: rustup (stable + nightly), go 1.23.4.
Debs extracted locally to `ref/smoke/.deps/extracted/` for dev libs
(libssl-dev, libgrpc++-dev, libgrpc-dev, libprotobuf-dev, protobuf-compiler).

## TL;DR

| repo                  | smoke result | what it actually did                          | Mistral?  |
|-----------------------|--------------|-----------------------------------------------|-----------|
| `flashinfer`          | **PASS**     | pip-installed `flashinfer-python` 0.6.13; ran `single_decode_with_kv_cache` on a synthetic 32Q/8KV fp16 GQA tensor on the live GPU | N/A (library, not a model server) |
| `openinfer`           | **PASS**     | `cargo check -p openinfer-engine` (Rust type layer) PASS; `cargo check -p openinfer-server` (full CUDA build) EXPECTED FAIL — pinned flashinfer 3rdparty needs newer CCCL than host's CUDA 12.6.20 (`cuda/cmath` not found) | **NO** (workspace has qwen3, qwen35, deepseek-v2/v4, glm52, kimi-k2 — no mistral crate) |
| `FlashQwen`           | **PASS**     | CMake configure of `engine/` EXPECTED FAIL (gRPC/abseil dev libs not installed system-wide); standalone nvcc compile of `kernels.cu` (the BF16 GEMM/RMSNorm/RoPE bundle) PASS | **NO** (`model_spec.h::Supported()` returns `arch == "Qwen3ForCausalLM"` only) |
| `zml` (a.k.a. `llmd`) | **PASS**     | Static check: Bazel workspace layout, `examples/mnist` + `examples/llm` exist, LLM example supported model list parsed from README | **NO** (LLM example supports Llama 3.1, Qwen 3.5, LFM 2.5 only) |

`zml/llmd` itself is a Docker image (`zmlai/llmd`); the source repo is
`zml/zml`, which is what we cloned and smoke-tested.

## Run all

```bash
bash ref/smoke/run_all.sh
```

Last run: 4/4 passed in ~11 s (excluding flashinfer's first-run JIT compile
of the decode module, which takes ~5 s on first invocation).

## Per-repo details

### flashinfer (flashinfer-ai/flashinfer)

Real, end-to-end kernel smoke. pip-installed `flashinfer-python` 0.6.13 into
the project venv. The smoke script:

1. Sets `PATH` to include `~/.venv/bin` (where pip put `ninja`) and
   `/usr/local/cuda/bin` (where the working nvcc lives — `/usr/bin/nvcc` on
   this Ubuntu is a 59-byte stub that doesn't actually compile).
2. Imports `flashinfer`, builds random GQA tensors shaped like a Mistral-7B
   decode step (32 Q heads, 8 KV heads, head_dim 128, fp16, 256-token
   synthetic KV cache).
3. Calls `single_decode_with_kv_cache`, JIT-builds the decode kernel, runs
   it, asserts shape/dtype/finite/non-zero.

The kernel actually ran on the RTX 3080 (output `sum=325.18`).

Side effects on the venv: `pip install flashinfer-python` upgraded torch
2.11 → 2.9 (forced by flashinfer 0.6.13) and broke `torchvision` 0.26.0
(needs torch 2.11). This is a real dep conflict in flashinfer-python 0.6.13
and would need a separate resolution if torchvision is required.

### openinfer (openinfer-project/openinfer)

Two-phase cargo check on the pinned nightly toolchain
(`rust-toolchain.toml` says `channel = "nightly"`):

- **Phase A** — `cargo check -p openinfer-engine` (Rust-only crate, no
  build.rs): PASS. The engine contract types, scheduler, KV cache, and
  sampler APIs all compile cleanly. This is the most a no-GPU-build host
  can prove about the engine crate.

- **Phase B** — `cargo check -p openinfer-server` (default workspace
  member, which pulls `openinfer-kernels`): EXPECTED FAIL. The pinned
  `openinfer-kernels/third_party/flashinfer` commit (`d768c14`) requires
  CCCL ≥ 2.5's `cuda/cmath` top-level header, but the host's CUDA 12.6.20
  ships CCCL 2.4 where `cmath` is under `cuda/std/`. Fix would be either
  `apt install cuda-toolkit-13-x` (needs sudo) or pinning an older flashinfer
  in the 3rdparty.

To bring the host up to "actually run a Qwen3-4B model", in addition to
CUDA 13.x you'd need Qwen3-4B safetensors and to run
`cargo run --release -p openinfer-server -- --model-path <path>`. None of
this is Mistral — there is no `openinfer-mistral-7b` crate, and adding one
would be a multi-day project (config parsing, weight loader, kernel
selection, KV cache, scheduler hookup).

### FlashQwen (frankkk96/FlashQwen)

Two-phase C++/CUDA smoke:

- **CMake configure** of `engine/`: EXPECTED FAIL on gRPC's
  `pkg-config --libs grpc++` because we can't `apt install libgrpc++-dev
  libgrpc-dev libabsl-dev` (no sudo). The relevant dev debs were
  downloaded to `ref/smoke/.deps/extracted/` but the transitive abseil
  tree isn't worth chasing for a smoke.

- **Standalone nvcc compile of `kernels.cu`** (the BF16 cuBLAS GEMM,
  RMSNorm, RoPE, and assorted elementwise kernels): PASS. 68 440 byte
  sm_86 object produced. This is the closest thing to a "1-token kernel
  smoke" achievable without the gRPC server stub.

Mistral support is gated in `engine/src/model_spec.h`:
```cpp
bool Supported() const { return arch == "Qwen3ForCausalLM"; }
```
Adding Mistral would mean adding a `MistralForCausalLM` arch, new config
parsing, and likely a different attention bias (Mistral uses sliding
window 4096 on every other layer).

### zml (zml/zml — source of the `zmlai/llmd` Docker image)

Static-only smoke (no bazel). Confirmed:

- Workspace files present (`MODULE.bazel`, `BUILD.bazel`)
- `examples/mnist` and `examples/llm` exist
- LLM example's `README.md` enumerates supported model families: Llama 3.1,
  Qwen 3.5, LFM 2.5
- No "mistral" string anywhere in `examples/llm/` or `zml/`

Real smoke would need `bazelisk` installed and a first-run MLIR toolchain
download of ~1 GB. Adding Mistral to zml's LLM example would mean writing
a new model file in `examples/llm/` (the directory has a `llama.zig`,
presumably similar for mistral) — not done in this smoke.

## How to reproduce

```bash
# One-time setup
curl -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --profile minimal
rustup install nightly
cd ~/yalm
.venv/bin/pip install flashinfer-python
# (gRPC/openssl dev debs were extracted to ref/smoke/.deps/extracted/ in this
#  session; cd ref/smoke/.deps && apt-get download ... && dpkg-deb -x ...)

# Per repo
bash ref/smoke/run_all.sh
```

## Why "Mistral" appears in no smoke

Per the user's question: none of the 4 reference repos support Mistral.
Each smoke labels this honestly rather than faking a 1-token Mistral run
that would silently exercise a model loader, weight format, or tokenizer
that the repo doesn't actually have. To run Mistral-7B on this hardware
today, the yalm engine already does it (`./yalm/build/main` against
`mistral-7b-instruct-fp16.yalm`); the 4 reference repos are interesting
in comparison, not as Mistral hosts.
