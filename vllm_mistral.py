#!/usr/bin/env python3
"""Mistral-7B decode via vLLM (offline LLM.generate)."""

import argparse
import sys
import time
from pathlib import Path

DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
DEFAULT_PROMPT = "Q: What is the meaning of life?"


def main() -> None:
  p = argparse.ArgumentParser(description="Mistral decode benchmark with vLLM")
  p.add_argument("--count", type=int, default=120)
  p.add_argument("--runs", type=int, default=1, help="Timed decode runs (model loaded once).")
  p.add_argument("--temperature", type=float, default=0.7)
  p.add_argument("--prompt", default=DEFAULT_PROMPT)
  p.add_argument("--prompt-file", default=None)
  p.add_argument("--model", default=DEFAULT_MODEL)
  p.add_argument("--warmup", type=int, default=1)
  p.add_argument("--ignore-eos", dest="ignore_eos", action="store_true", default=True,
                 help="Generate exactly --count tokens (bench parity with yalm -n; default on).")
  p.add_argument("--no-ignore-eos", dest="ignore_eos", action="store_false")
  p.add_argument("--spec-decode", action="store_true",
                 help="Enable speculative decoding (draft model required; separate bench mode).")
  args = p.parse_args()

  prompt = args.prompt
  if args.prompt_file:
    prompt = Path(args.prompt_file).read_text()

  try:
    from vllm import LLM, SamplingParams
  except ImportError:
    print("vllm is not installed in this Python.", file=sys.stderr)
    sys.exit(1)

  llm_kw: dict = dict(
    model=args.model,
    dtype="float16",
    trust_remote_code=True,
    max_model_len=4096,
  )
  if args.spec_decode:
    print("spec-decode: enabled (pass draft model via --spec-model when supported)", file=sys.stderr)
    # ponytail: caller must set speculative_config when benchmarking spec decode
  samp_kw = dict(max_tokens=args.count, temperature=args.temperature)
  if args.ignore_eos:
    samp_kw["ignore_eos"] = True
    samp_kw["min_tokens"] = args.count
  llm = LLM(**llm_kw)
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
  mode = "spec-decode" if args.spec_decode else "no-spec-decode"
  print(text, end="" if text.endswith("\n") else "\n")
  print(
    f"\nGeneration stats:\n"
    f"  mode: {mode}\n"
    f"  {n_out} tokens (requested {args.count})\n"
    f"  throughput: {avg:.5f}tok/s\n"
    f"  latency: {elapsed / n_out:.5f}s/tok\n"
    f"  total: {elapsed:.5f}s\n"
  )


if __name__ == "__main__":
  main()
