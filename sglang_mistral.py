#!/usr/bin/env python3
"""Mistral-7B decode via SGLang Engine."""

import argparse
import sys
import time
from pathlib import Path

DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
DEFAULT_PROMPT = "Q: What is the meaning of life?"


def main() -> None:
  p = argparse.ArgumentParser(description="Mistral decode benchmark with SGLang")
  p.add_argument("--count", type=int, default=120)
  p.add_argument("--runs", type=int, default=1, help="Timed decode runs (model loaded once).")
  p.add_argument("--temperature", type=float, default=0.7)
  p.add_argument("--prompt", default=DEFAULT_PROMPT)
  p.add_argument("--prompt-file", default=None)
  p.add_argument("--model", default=DEFAULT_MODEL)
  p.add_argument("--warmup", type=int, default=1,
                 help="Short decode warmups before CUDA graph capture.")
  p.add_argument("--ignore-eos", dest="ignore_eos", action="store_true", default=True,
                 help="Generate exactly --count tokens (bench parity; default on).")
  p.add_argument("--no-ignore-eos", dest="ignore_eos", action="store_false")
  p.add_argument("--graph-warmup", dest="graph_warmup", action="store_true", default=True,
                 help="One full-length decode before timing (CUDA graph capture; default on).")
  p.add_argument("--no-graph-warmup", dest="graph_warmup", action="store_false")
  args = p.parse_args()

  prompt = args.prompt
  if args.prompt_file:
    prompt = Path(args.prompt_file).read_text()

  try:
    import sglang as sgl
  except ImportError:
    print("sglang is not installed in this Python.", file=sys.stderr)
    sys.exit(1)

  llm = sgl.Engine(model_path=args.model, dtype="float16", context_length=4096)
  samp_kw = {"max_new_tokens": args.count, "temperature": args.temperature}
  if args.ignore_eos:
    samp_kw["ignore_eos"] = True
  samp = samp_kw
  warm = {"max_new_tokens": min(args.count, 8), "temperature": args.temperature}

  for _ in range(args.warmup):
    llm.generate(prompt, warm)
  if args.graph_warmup:
    # ponytail: first full decode captures CUDA graphs (~40s on 3080); keep out of timed run
    llm.generate(prompt, samp)

  rates: list[float] = []
  text = ""
  for run in range(args.runs):
    t0 = time.time()
    out = llm.generate(prompt, samp)
    elapsed = time.time() - t0
    meta = out.get("meta_info", {}) if isinstance(out, dict) else {}
    n_out = meta.get("completion_tokens") or meta.get("output_token_logprobs_len")
    if not n_out and isinstance(out, dict) and "output_ids" in out:
      n_out = len(out["output_ids"])
    if not n_out:
      n_out = args.count
    rate = n_out / elapsed if elapsed > 0 else 0.0
    rates.append(rate)
    print(f"  run {run + 1}: {rate:.2f} tok/s", file=sys.stderr)
    text = out.get("text", "") if isinstance(out, dict) else str(out)

  avg = sum(rates) / len(rates)
  n_out = meta.get("completion_tokens") if isinstance(out, dict) else args.count
  if not n_out:
    n_out = args.count
  elapsed = n_out / avg if avg > 0 else 0.0
  print(text, end="" if text.endswith("\n") else "\n")
  print(
    f"\nGeneration stats:\n"
    f"  {n_out} tokens (requested {args.count})\n"
    f"  throughput: {avg:.5f}tok/s\n"
    f"  latency: {elapsed / n_out:.5f}s/tok\n"
    f"  total: {elapsed:.5f}s\n"
  )
  llm.shutdown()


if __name__ == "__main__":
  main()
