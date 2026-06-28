#!/usr/bin/env python3
"""Benchmark Mistral-7B-Instruct-v0.2 decode across engines; print comparison table."""

import argparse
import os
import re
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

PROMPT = "Q: What is the meaning of life?"
MODEL_GB = 14.48
PEAK_BW_4090 = 1008
PEAK_BW_3080 = 760

BLOG_4090: dict[str, tuple[str, str]] = {
  "huggingface transformers": ("25.9", "37.2%"),
  "llama.cpp": ("61.0", "87.6%"),
  "calm": ("66.0", "94.8%"),
  "yalm": ("63.8", "91.6%"),
  "tinygrad": ("—", "—"),
}

ENGINE_ORDER = list(BLOG_4090)


@dataclass
class BenchResult:
  name: str
  rates: list[float]

  @property
  def avg(self) -> float:
    return statistics.mean(self.rates)

  @property
  def stdev(self) -> float:
    return statistics.stdev(self.rates) if len(self.rates) > 1 else 0.0

  def bw_pct(self) -> float:
    return self.avg * MODEL_GB / PEAK_BW_3080 * 100

  def fmt_3080(self) -> str:
    n = len(self.rates)
    if n == 1:
      return f"{self.avg:.1f}"
    if self.stdev < 0.05:
      return f"{self.avg:.1f} ({n}-run avg, stdev ~{self.stdev:.2f})"
    return f"{self.avg:.1f} (avg, stdev ~{self.stdev:.2f}, n={n})"


def parse_rate(engine: str, text: str) -> float:
  if engine == "yalm":
    m = re.search(r"Generation stats:.*?throughput:\s*([\d.]+)\s*tok/s", text, re.DOTALL)
  else:
    patterns = {
      "huggingface transformers": r"throughput:\s*([\d.]+)\s*tok/s",
      "llama.cpp": r"Generation:\s*([\d.]+)\s*t/s",
      "calm": r"tok/s=([\d.]+)",
      "tinygrad": r"([\d.]+)\s*tok/s\s*\(",
    }
    m = re.search(patterns[engine], text)
  if not m:
    raise RuntimeError(f"could not parse {engine} output:\n{text[-2000:]}")
  return float(m.group(1))


def run_cmd(cmd: list[str], env: dict[str, str] | None = None, cwd: str | None = None) -> str:
  print(f"$ {' '.join(cmd)}", file=sys.stderr)
  p = subprocess.run(cmd, check=True, capture_output=True, text=True, env=env, cwd=cwd)
  return p.stdout + p.stderr


def bench_transformers(model_id: str, count: int, runs: int) -> BenchResult:
  import torch
  from transformers import AutoModelForCausalLM, AutoTokenizer

  tokenizer = AutoTokenizer.from_pretrained(model_id)
  model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
  inputs = tokenizer([PROMPT], return_tensors="pt").to("cuda")
  _ = model.generate(**inputs, max_new_tokens=1, do_sample=True)

  rates = []
  for i in range(runs):
    t0 = time.time()
    out = model.generate(**inputs, max_new_tokens=count, do_sample=True)[0].tolist()
    elapsed = time.time() - t0
    n_gen = len(out) - inputs["input_ids"].shape[-1]
    rates.append(n_gen / elapsed)
    print(f"transformers run {i+1}: {rates[-1]:.2f} tok/s", file=sys.stderr)
  return BenchResult("huggingface transformers", rates)


def bench_yalm(bin_path: Path, ckpt: Path, count: int, runs: int) -> BenchResult:
  rates = []
  for i in range(runs):
    out = run_cmd([str(bin_path), str(ckpt), "-d", "cuda", "-m", "completion", "-i", PROMPT, "-n", str(count)])
    rates.append(parse_rate("yalm", out))
    print(f"yalm run {i+1}: {rates[-1]:.2f} tok/s", file=sys.stderr)
  return BenchResult("yalm", rates)


def bench_llamacpp(bin_path: Path, gguf: Path, count: int, runs: int) -> BenchResult:
  rates = []
  for i in range(runs):
    out = run_cmd([
      str(bin_path), "-m", str(gguf), "-c", "4096", "-n", str(count),
      "-p", PROMPT, "--no-display-prompt", "--single-turn",
    ])
    rates.append(parse_rate("llama.cpp", out))
    print(f"llama.cpp run {i+1}: {rates[-1]:.2f} tok/s", file=sys.stderr)
  return BenchResult("llama.cpp", rates)


def bench_calm(bin_path: Path, calm_file: Path, count: int, runs: int) -> BenchResult:
  rates = []
  for i in range(runs):
    out = run_cmd([str(bin_path), str(calm_file), "-c", "4096", "-n", str(count), "-i", PROMPT])
    rates.append(parse_rate("calm", out))
    print(f"calm run {i+1}: {rates[-1]:.2f} tok/s", file=sys.stderr)
  return BenchResult("calm", rates)


def bench_tinygrad(script: Path, python: Path, model: str, count: int, runs: int, beam: int,
                  weights: Path | None = None) -> BenchResult:
  env = os.environ.copy()
  env["BEAM"] = str(beam)
  rates = []
  for i in range(runs + 1):
    cmd = [str(python), str(script), "--model", model, "--count", str(count)]
    if weights is not None:
      cmd += ["--weights", str(weights)]
    proc = subprocess.run(
      cmd, check=True, capture_output=True, text=True, env=env, cwd=str(script.parent),
    )
    text = proc.stdout + proc.stderr
    if i == 0:
      print("tinygrad warmup done", file=sys.stderr)
      continue
    rates.append(parse_rate("tinygrad", text))
    print(f"tinygrad run {i}: {rates[-1]:.2f} tok/s", file=sys.stderr)
  return BenchResult("tinygrad", rates)


def print_table(results: list[BenchResult], count: int) -> None:
  by_name = {r.name: r for r in results}
  print()
  print(f'Comparison (Mistral-7B-Instruct-v0.2 fp16, 4k context, prompt "{PROMPT}", {count} generated tokens):')
  print()
  print(f"Card peak BW: RTX 4090 = {PEAK_BW_4090} GB/s, RTX 3080 = {PEAK_BW_3080} GB/s.")
  print(f'BW used = model size ({MODEL_GB} GB fp16) × tok/s.')
  print()
  print("| Engine                       | RTX 4090 tok/s (blog) | 4090 % peak BW | RTX 3080 tok/s (this box) | 3080 % peak BW |")
  print("| ---------------------------- | --------------------- | -------------- | ------------------------- | -------------- |")
  for name in ENGINE_ORDER:
    blog_tok, blog_bw = BLOG_4090[name]
    if name in by_name:
      r = by_name[name]
      tok_3080 = r.fmt_3080()
      bw_3080 = f"{r.bw_pct():.1f}%"
    else:
      tok_3080, bw_3080 = "—", "—"
    print(f"| {name:<28} | {blog_tok:<21} | {blog_bw:<14} | {tok_3080:<25} | {bw_3080:<14} |")
  print()
  ranked = sorted((r for r in results), key=lambda r: r.avg, reverse=True)
  ranking = " > ".join(f"{r.name.split()[-1] if ' ' in r.name else r.name} {r.avg:.1f}" for r in ranked)
  print(f"Ranking on this box: {ranking}")


def main() -> None:
  root = Path(__file__).resolve().parent
  home = Path.home()
  parser = argparse.ArgumentParser(description="Benchmark Mistral decode and print comparison table")
  parser.add_argument("--count", type=int, default=120, help="Generated tokens per run")
  parser.add_argument("--runs", type=int, default=10, help="Timed runs per engine (transformers defaults to 1)")
  parser.add_argument("--beam", type=int, default=8, help="BEAM for tinygrad")
  parser.add_argument("--engines", default="all",
                      help="Comma-separated: transformers,yalm,llama.cpp,calm,tinygrad or all")
  parser.add_argument("--python", default=sys.executable, help="Python for tinygrad subprocess")
  parser.add_argument("--model", default=DEFAULT_MODEL, help="HuggingFace repo id")
  parser.add_argument("--weights", default=None, help="Local weights dir (overrides --model / hub cache)")
  parser.add_argument("--yalm-bin", default=str(root / "build/main"))
  parser.add_argument("--yalm-ckpt", default=str(root / "mistral-7b-instruct-fp16.yalm"))
  parser.add_argument("--llama-cli", default=str(home / "llama.cpp/build/bin/llama-cli"))
  parser.add_argument("--gguf", default=str(home / "mistral-7b-instruct-v0.2.fp16.gguf"))
  parser.add_argument("--calm-bin", default=str(home / "calm/build/run"))
  parser.add_argument("--calm-file", default=str(home / ".cache/mistral-7b-instruct.fp16.calm"))
  parser.add_argument("--tinygrad-script", default=str(root / "tinygrad_mistral.py"))
  args = parser.parse_args()
  weights_dir = Path(args.weights).expanduser() if args.weights else None
  model_id = str(weights_dir) if weights_dir else args.model

  want = set(ENGINE_ORDER) if args.engines == "all" else {e.strip() for e in args.engines.split(",")}
  unknown = want - set(ENGINE_ORDER)
  if unknown:
    parser.error(f"unknown engines: {', '.join(sorted(unknown))}")

  results: list[BenchResult] = []

  if "huggingface transformers" in want:
    results.append(bench_transformers(model_id, args.count, runs=1))

  if "yalm" in want:
    for p in (args.yalm_bin, args.yalm_ckpt):
      if not Path(p).exists():
        print(f"skip yalm: missing {p}", file=sys.stderr)
        break
    else:
      results.append(bench_yalm(Path(args.yalm_bin), Path(args.yalm_ckpt), args.count, args.runs))

  if "llama.cpp" in want:
    for p in (args.llama_cli, args.gguf):
      if not Path(p).exists():
        print(f"skip llama.cpp: missing {p}", file=sys.stderr)
        break
    else:
      results.append(bench_llamacpp(Path(args.llama_cli), Path(args.gguf), args.count, args.runs))

  if "calm" in want:
    for p in (args.calm_bin, args.calm_file):
      if not Path(p).exists():
        print(f"skip calm: missing {p}", file=sys.stderr)
        break
    else:
      results.append(bench_calm(Path(args.calm_bin), Path(args.calm_file), args.count, args.runs))

  if "tinygrad" in want:
    if not Path(args.tinygrad_script).exists():
      print(f"skip tinygrad: missing {args.tinygrad_script}", file=sys.stderr)
    else:
      try:
        import tinygrad  # noqa: F401
      except ImportError:
        print("skip tinygrad: pip install tinygrad", file=sys.stderr)
      else:
        results.append(bench_tinygrad(
          Path(args.tinygrad_script), Path(args.python), args.model,
          args.count, args.runs, args.beam, weights_dir,
        ))

  if not results:
    parser.error("no engines ran (check paths and --engines)")

  print_table(results, args.count)


if __name__ == "__main__":
  main()
