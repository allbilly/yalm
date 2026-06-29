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
  p.add_argument("--temperature", type=float, default=0.7)
  p.add_argument("--prompt", default=DEFAULT_PROMPT)
  p.add_argument("--prompt-file", default=None)
  p.add_argument("--model", default=DEFAULT_MODEL)
  p.add_argument("--warmup", type=int, default=1)
  args = p.parse_args()

  prompt = args.prompt
  if args.prompt_file:
    prompt = Path(args.prompt_file).read_text()

  try:
    from tensorrt_llm import LLM, SamplingParams
  except ImportError:
    print(
      "tensorrt-llm is not installed.\n"
      "Native: https://nvidia.github.io/TensorRT-LLM/installation.html\n"
      "Docker: ./trtllm_docker_bench.sh --count 120",
      file=sys.stderr,
    )
    sys.exit(1)

  llm = LLM(model=args.model)
  warm = SamplingParams(max_tokens=min(args.count, 8), temperature=args.temperature)
  full = SamplingParams(max_tokens=args.count, temperature=args.temperature)

  for _ in range(args.warmup):
    llm.generate([prompt], warm)

  t0 = time.time()
  outputs = llm.generate([prompt], full)
  elapsed = time.time() - t0

  text = outputs[0].outputs[0].text if outputs and outputs[0].outputs else ""
  n_out = args.count
  rate = n_out / elapsed if elapsed > 0 else 0.0
  print(text, end="" if text.endswith("\n") else "\n")
  print(
    f"\nGeneration stats:\n"
    f"  {n_out} tokens\n"
    f"  throughput: {rate:.5f}tok/s\n"
    f"  latency: {elapsed / n_out:.5f}s/tok\n"
    f"  total: {elapsed:.5f}s\n"
  )


if __name__ == "__main__":
  main()
