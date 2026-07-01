#!/usr/bin/env python3
"""Mistral-7B decode via TensorRT-LLM LLM API (optional: pip install tensorrt-llm)."""

import argparse
import sys
import time
from pathlib import Path

DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
DEFAULT_PROMPT = "Q: What is the meaning of life?"


def main() -> None:
  p = argparse.ArgumentParser(description="Mistral decode benchmark with TensorRT-LLM")
  p.add_argument("--count", type=int, default=120)
  p.add_argument("--runs", type=int, default=1, help="Timed decode runs (model loaded once).")
  p.add_argument("--temperature", type=float, default=0.7)
  p.add_argument("--prompt", default=DEFAULT_PROMPT)
  p.add_argument("--prompt-file", default=None)
  p.add_argument("--model", default=DEFAULT_MODEL)
  p.add_argument("--warmup", type=int, default=1)
  p.add_argument("--ignore-eos", dest="ignore_eos", action="store_true", default=True,
                 help="Generate exactly --count tokens (bench parity; default on).")
  p.add_argument("--no-ignore-eos", dest="ignore_eos", action="store_false")
  p.add_argument("--max-seq-len", type=int, default=4096)
  p.add_argument("--kv-cache-fraction", type=float, default=0.15,
                 help="Cap KV pool (default 0.15; TRT default 0.9 OOMs on 20GB with Mistral-7B fp16).")
  args = p.parse_args()

  prompt = args.prompt
  if args.prompt_file:
    prompt = Path(args.prompt_file).read_text()

  try:
    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig
  except ImportError:
    print(
      "tensorrt-llm is not installed.\n"
      "Native: https://nvidia.github.io/TensorRT-LLM/installation.html\n"
      "Docker: mistral_bench.py --engines tensorrt-llm --trtllm-docker --count 120",
      file=sys.stderr,
    )
    sys.exit(1)

  llm = LLM(
    model=args.model,
    max_seq_len=args.max_seq_len,
    max_num_tokens=args.max_seq_len,
    max_batch_size=1,
    kv_cache_config=KvCacheConfig(
      max_tokens=args.max_seq_len,
      free_gpu_memory_fraction=args.kv_cache_fraction,
    ),
  )
  samp_kw = dict(max_tokens=args.count, temperature=args.temperature)
  if args.ignore_eos:
    samp_kw["ignore_eos"] = True
    samp_kw["min_tokens"] = args.count
  warm = SamplingParams(max_tokens=min(args.count, 8), temperature=args.temperature)
  full = SamplingParams(**samp_kw)

  for _ in range(args.warmup):
    llm.generate([prompt], warm)

  rates: list[float] = []
  text = ""
  for run in range(args.runs):
    t0 = time.time()
    outputs = llm.generate([prompt], full)
    elapsed = time.time() - t0
    n_out = len(outputs[0].outputs[0].token_ids) if outputs and outputs[0].outputs else 0
    rate = n_out / elapsed if elapsed > 0 else 0.0
    rates.append(rate)
    print(f"  run {run + 1}: {rate:.2f} tok/s", file=sys.stderr)
    text = outputs[0].outputs[0].text if outputs and outputs[0].outputs else ""

  avg = sum(rates) / len(rates)
  n_out = len(outputs[0].outputs[0].token_ids) if outputs and outputs[0].outputs else 0
  elapsed = n_out / avg if avg > 0 else 0.0
  print(text, end="" if text.endswith("\n") else "\n")
  print(
    f"\nGeneration stats:\n"
    f"  {n_out} tokens (requested {args.count})\n"
    f"  throughput: {avg:.5f}tok/s\n"
    f"  latency: {elapsed / n_out:.5f}s/tok\n"
    f"  total: {elapsed:.5f}s\n"
  )


if __name__ == "__main__":
  main()
