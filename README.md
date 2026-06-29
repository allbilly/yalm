# YALM

TODO
- make 3d plot

benchmark

Re-run all engines and print the comparison table (RTX 3080, Mistral-7B-Instruct-v0.2):

```
pip install -r requirements.txt
.venv/bin/python mistral_bench.py
```

HF weights: pass `--model mistralai/Mistral-7B-Instruct-v0.2` (resolved via `huggingface_hub` from `~/.cache/huggingface/hub`, no snapshot hash in config). Local dir: `--weights /path` or `MISTRAL_PATH`.

Script: `mistral_bench.py` — benchmarks transformers (1 run), yalm, llama.cpp, calm, tinygrad (default 10 runs each), then prints the markdown comparison table. Tinygrad uses `tinygrad_mistral.py` (pip install tinygrad; no repo clone).

Result on this machine (single run, 129 generated tokens, prompt "Q: What is the meaning of life?"):
- throughput: **39.0 tok/s**
- latency: 0.0256 s/tok
- total: 3.31 s

Comparison vs blog (Mistral-7B-Instruct-v0.2 fp16, 4k context, prompt "Q: What is the meaning of life?", 120 generated tokens):

Card peak memory bandwidth: RTX 4090 = 1008 GB/s (24GB GDDR6X, 384-bit, 21.0 Gbps [videocardz](https://videocardz.net/nvidia-geforce-rtx-4090)). RTX 3080 = 760 GB/s (10GB GDDR6X, 320-bit, 19.0 Gbps [videocardz](https://videocardz.net/nvidia-geforce-rtx-3080)). For each engine, "BW used" = model size (14.48 GB fp16) × tok/s, i.e. the minimum bytes that must move through DRAM per token for a fully memory-bandwidth-bound decode.

| Engine                       | RTX 4090 tok/s (blog) | 4090 % peak BW | RTX 3080 tok/s (this box) | 3080 % peak BW |
| ---------------------------- | --------------------- | -------------- | ------------------------- | -------------- |
| huggingface transformers    | 25.9                  | 37.2%          | —                         | —              |
| transformers (torch.compile) | —                     | —              | **39.8** (single run)     | **75.8%**      |
| llama.cpp                    | 61.0                  | 87.6%          | ~46.5 long / ~48.4 short  | ~88–92%        |
| calm                        | 66.0                  | 94.8%          | **~48.9** (matched long)  | **~93%**       |
| yalm (`-d cuda`)             | 63.8                  | 91.6%          | **~49.3 short / ~48.9 long** | **~93%**    |
| yalm (`-d cuda-coop`)        | —                     | —              | **~49.3 short / ~48.9 long** | **~93%**    |
| yalm (`-d cuda-cublas`)      | —                     | —              | **~43**                   | ~82%           |
| yalm (`-d cuda-cudnn`)       | —                     | —              | **~43**                   | ~82%           |
| yalm (`-d cuda-cutile`)      | —                     | —              | **~45**                   | ~87%           |
| vllm                         | —                     | —              | (see setup)               | —              |
| sglang                       | —                     | —              | (see setup)               | —              |
| tensorrt-llm                 | —                     | —              | (requires install)        | —              |
| tinygrad                     | —                     | —              | 35.1 (10-run avg, stdev ~0.04) | 66.9%      |

**RTX 3080 decode (Mistral-7B fp16, Jun 2026, after Fix #3–#6 + `cuda-coop`):**

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

On RTX 3080 10GB, `vllm_mistral.py` sets `max_model_len=4096` so fp16 weights (~14.5 GB) plus KV cache fit.

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

First install pulls large CUDA wheels (~2 GB); expect several minutes. `sglang_mistral.py` uses `context_length=4096` for the same 10GB VRAM reason.

Verify:

```
.venv/bin/python bench_engines.py sglang
~/.cache/uv/environments-v2/sglang-bench/.venv/bin/python sglang_mistral.py --count 8
```

**CUDA 12 hosts:** SGLang defaults to CUDA 13 wheels. If import or kernel load fails, follow the [CUDA 12 override steps](https://docs.sglang.io/docs/get-started/install) (`sglang-kernel` from `docs.sglang.ai/whl/cu129/`).

**Docker (optional, for serving):** official images `vllm/vllm-openai` and `lmsysorg/sglang:latest-runtime` — see each project's Docker docs. The bench scripts here use in-process Python APIs, not the server containers.

**TensorRT-LLM (Docker bench)**

Image on this box: `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc19`. GPU inside the container needs [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html):

```
# Ubuntu (summary — see NVIDIA docs for your distro)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Verify GPU in container:

```
docker run --rm --gpus all nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc19 \
  python3 -c "import tensorrt_llm; print('ok')"
```

Benchmark (first run builds the TRT engine; can take several minutes):

```
chmod +x trtllm_docker_bench.sh
./trtllm_docker_bench.sh --count 120
.venv/bin/python mistral_bench.py --engines tensorrt-llm --trtllm-docker --count 120
```

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
- transformers (torch.compile): `.venv/bin/python mistral_bench.py --engines "huggingface transformers" --count 120 --runs 1`
- tensorrt-llm: `./trtllm_docker_bench.sh --count 120` or `.venv/bin/python mistral_bench.py --engines tensorrt-llm --trtllm-docker --count 120`
- vllm: `.venv/bin/python mistral_bench.py --engines vllm --count 120` (uses `vllm_mistral.py`; auto-finds uv venv)
- sglang: `.venv/bin/python mistral_bench.py --engines sglang --count 120` (uses `sglang_mistral.py`)
- yalm graph: `./build/main mistral-7b-instruct-fp16.yalm -d cuda -m completion -i "Q: What is the meaning of life?" -n 120`
- yalm library backends: `-d cuda-cublas`, `-d cuda-cudnn`, `-d cuda-cutile`, …
- yalm coop: `./build/main mistral-7b-instruct-fp16.yalm -d cuda-coop -m completion -f long_prompt.txt -n 30`
- llama.cpp: `~/llama.cpp/build/bin/llama-cli -m ~/mistral-7b-instruct-v0.2.fp16.gguf ...`
- calm long (matched workload): `calm/build/run ~/.cache/mistral-7b-instruct.fp16.calm -c 4096 -n 3232 -i - < long_prompt.txt` (3202 prompt + 30 decode steps; **not** `-n 30` alone)
- tinygrad: `BEAM=8 .venv/bin/python tinygrad_mistral.py --count 120`

- Ranking on this 3080 (Jun 2026): **yalm `-d cuda` ≈ calm > library backends > llama.cpp > transformers (torch.compile)**. Custom kernels ~49 tok/s; library paths ~43–45 tok/s. See [`results.md`](results.md) §28.

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