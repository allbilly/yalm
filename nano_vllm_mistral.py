#!/usr/bin/env python3
"""Mistral-7B decode via nano-vllm (ref/nano-vllm)."""

import argparse
import os
import sys
import time
from pathlib import Path

DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
DEFAULT_PROMPT = "Q: What is the meaning of life?"


def resolve_model(path: str) -> str:
  p = Path(path).expanduser()
  if p.is_dir():
    return str(p.resolve())
  from huggingface_hub import snapshot_download
  try:
    return snapshot_download(path, local_files_only=True)
  except Exception:
    return snapshot_download(path)


def main() -> None:
  p = argparse.ArgumentParser(description="Mistral decode benchmark with nano-vllm")
  p.add_argument("--count", type=int, default=120)
  p.add_argument("--runs", type=int, default=1, help="Timed decode runs (model loaded once).")
  p.add_argument("--temperature", type=float, default=0.7)
  p.add_argument("--prompt", default=DEFAULT_PROMPT)
  p.add_argument("--prompt-file", default=None)
  p.add_argument("--model", default=DEFAULT_MODEL)
  p.add_argument("--warmup", type=int, default=1)
  p.add_argument("--enforce-eager", action="store_true", help="Disable CUDA graphs")
  args = p.parse_args()

  prompt = Path(args.prompt_file).read_text() if args.prompt_file else args.prompt
  model_path = resolve_model(os.environ.get("MISTRAL_PATH", args.model))

  try:
    from nanovllm import LLM, SamplingParams
  except ImportError:
    print("nanovllm not installed. Run: pip install -e ref/nano-vllm", file=sys.stderr)
    sys.exit(1)

  llm = LLM(
    model_path,
    enforce_eager=args.enforce_eager,
    tensor_parallel_size=1,
    max_model_len=4096,
    gpu_memory_utilization=0.90,
  )
  warm = SamplingParams(max_tokens=min(args.count, 8), temperature=args.temperature)
  full = SamplingParams(
    max_tokens=args.count,
    temperature=args.temperature,
    ignore_eos=True,
  )

  for _ in range(args.warmup):
    llm.generate([prompt], warm)

  rates: list[float] = []
  text = ""
  for run in range(args.runs):
    t0 = time.time()
    outputs = llm.generate([prompt], full)
    elapsed = time.time() - t0
    n_tok = len(outputs[0].get("token_ids", [])) or args.count
    rate = n_tok / elapsed
    rates.append(rate)
    print(f"  run {run + 1}: {rate:.2f} tok/s", file=sys.stderr)
    text = outputs[0]["text"]

  avg = sum(rates) / len(rates)
  n_tok = len(outputs[0].get("token_ids", [])) or args.count
  elapsed = n_tok / avg
  print(text)
  print(f"throughput: {avg:.2f} tok/s ({elapsed:.2f}s for {n_tok} tokens)")


if __name__ == "__main__":
  main()
