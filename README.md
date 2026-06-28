# YALM

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
| huggingface transformers    | 25.9                  | 37.2%          | 39.0                      | 74.3%          |
| llama.cpp                    | 61.0                  | 87.6%          | 48.4 (10-run avg, stdev ~0.1) | 92.2%       |
| calm                        | 66.0                  | 94.8%          | 48.9 (10-run avg, range 48.65-49.18) | 92.1% |
| yalm                         | 63.8                  | 91.6%          | 44.6 (10-run avg, stdev ~0.04) | 84.2%      |
| tinygrad                     | —                     | —              | 35.1 (10-run avg, stdev ~0.04) | 66.9%      |

Commands run on this box:
- huggingface transformers: `.venv/bin/python mistral_bench.py --engines transformers` (single run)
- yalm: `./build/main mistral-7b-instruct-fp16.yalm -d cuda -m completion -i "Q: What is the meaning of life?" -n 120` (looped 10x, see /tmp/yalm_10runs.log)
- llama.cpp: `~/llama.cpp/build/bin/llama-cli -m ~/mistral-7b-instruct-v0.2.fp16.gguf ...` The GGUF was converted from HF safetensors in the hub cache.
- calm: `~/calm/build/run ~/.cache/mistral-7b-instruct.fp16.calm -c 4096 -n 120 -i "Q: What is the meaning of life?"` (looped 10x, see /tmp/calm_10runs.log). The .calm file at `~/.cache` was pre-converted; on this CPU, calm builds and runs fine despite the blog's footnote 9 (which I think applies only to a different machine).
- tinygrad: `BEAM=8 .venv/bin/python tinygrad_mistral.py --count 120` (pip install tinygrad; no repo clone)

Notes:
- Ranking on this 3080: **calm > llama.cpp > yalm > transformers > tinygrad** (see table). On the blog 4090, llama.cpp and yalm swap places; calm still leads.
- All native engines are stable once warmed (stdev < 0.5 tok/s). tinygrad is unoptimized fp16 + naive attention; `BEAM=8` helps but it still trails transformers on this card.

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
  -d [cpu,cuda] which device to use (default - cuda)
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