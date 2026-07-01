#!/usr/bin/env python3
"""Benchmark Mistral-7B-Instruct-v0.2 decode across engines; print comparison table."""

import argparse
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

from bench_engines import find_python_with

DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
PROMPT = "Q: What is the meaning of life?"
MODEL_GB = 14.48
PEAK_BW = (4090, 1008), (3080, 760)

BLOG_4090 = {
  "huggingface transformers": ("25.9", "37.2%"),
  "transformers (torch.compile)": ("—", "—"),
  "llama.cpp": ("61.0", "87.6%"),
  "calm": ("66.0", "94.8%"),
  "yalm": ("63.8", "91.6%"),
  "vllm": ("—", "—"),
  "sglang": ("—", "—"),
  "tensorrt-llm": ("—", "—"),
  "nano-vllm": ("—", "—"),
  "tinygrad": ("—", "—"),
}
ENGINE_ORDER = list(BLOG_4090)

PATTERNS = {
  "huggingface transformers": r"throughput:\s*([\d.]+)\s*tok/s",
  "transformers (torch.compile)": r"throughput:\s*([\d.]+)\s*tok/s",
  "llama.cpp": r"Generation:\s*([\d.]+)\s*t/s",
  "calm": r"tok/s=([\d.]+)",
  "tinygrad": r"([\d.]+)\s*tok/s\s*\(",
  "yalm": r"Generation stats:.*?throughput:\s*([\d.]+)\s*tok/s",
  "vllm": r"throughput:\s*([\d.]+)\s*tok/s",
  "sglang": r"throughput:\s*([\d.]+)\s*tok/s",
  "tensorrt-llm": r"throughput:\s*([\d.]+)\s*tok/s",
  "nano-vllm": r"throughput:\s*([\d.]+)\s*tok/s",
}


def parse_rate(engine: str, text: str) -> float:
  flags = re.DOTALL if engine == "yalm" else 0
  if not (m := re.search(PATTERNS[engine], text, flags)):
    raise RuntimeError(f"could not parse {engine} output:\n{text[-2000:]}")
  return float(m.group(1))


def parse_rates(engine: str, text: str) -> list[float]:
  if runs := [float(m.group(2)) for m in re.finditer(r"run (\d+): ([\d.]+) tok/s", text)]:
    return runs
  return [parse_rate(engine, text)]


def run_cmd(cmd: list[str], env: dict[str, str] | None = None, cwd: str | None = None) -> str:
  print(f"$ {' '.join(cmd)}", file=sys.stderr)
  p = subprocess.run(cmd, check=True, capture_output=True, text=True, env=env, cwd=cwd)
  return p.stdout + p.stderr


def engine_env(python: Path) -> dict[str, str]:
  bindir = str(python.parent)
  path = os.environ.get("PATH", "")
  return {**os.environ, "PATH": f"{bindir}:{path}" if bindir not in path.split(":") else path}


def bench_external_py(engine: str, script: Path, python: Path, model_id: str, count: int,
                      runs: int) -> tuple[str, list[float]]:
  cmd = [str(python), str(script), "--model", model_id, "--count", str(count), "--runs", str(runs)]
  rates = parse_rates(engine, run_cmd(cmd, env=engine_env(python)))
  for i, r in enumerate(rates, 1):
    print(f"{engine} run {i}: {r:.2f} tok/s", file=sys.stderr)
  return engine, rates


def bench_subprocess(engine: str, cmd: list[str], runs: int, env: dict[str, str] | None = None) -> tuple[str, list[float]]:
  if "--runs" in cmd:
    rates = parse_rates(engine, run_cmd(cmd, env=env))
  else:
    rates = [parse_rate(engine, run_cmd(cmd, env=env)) for _ in range(runs)]
  for i, r in enumerate(rates, 1):
    print(f"{engine} run {i}: {r:.2f} tok/s", file=sys.stderr)
  return engine, rates


def bench_transformers(model_id: str, count: int, runs: int, *, torch_compile: bool = True) -> tuple[str, list[float]]:
  import torch
  from transformers import AutoModelForCausalLM, AutoTokenizer

  engine = "transformers (torch.compile)" if torch_compile else "huggingface transformers"
  tok = AutoTokenizer.from_pretrained(model_id)
  model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
  model.eval()
  if torch_compile:
    model = torch.compile(model, mode="reduce-overhead")
  inputs = tok([PROMPT], return_tensors="pt").to("cuda")
  gen_kw = dict(max_new_tokens=count, do_sample=True)

  with torch.inference_mode():
    warmups = 3 if torch_compile else 1
    for _ in range(warmups):
      model.generate(**inputs, max_new_tokens=min(count, 8 if torch_compile else 1), do_sample=True)
    if torch_compile:
      torch.cuda.synchronize()
    rates = []
    for run in range(runs):
      t0 = time.time()
      out = model.generate(**inputs, **gen_kw)[0].tolist()
      if torch_compile:
        torch.cuda.synchronize()
      rate = (len(out) - inputs["input_ids"].shape[-1]) / (time.time() - t0)
      rates.append(rate)
      print(f"{engine} run {run + 1}: {rate:.2f} tok/s", file=sys.stderr)

  del model, tok, inputs, out
  torch.cuda.empty_cache()
  return engine, rates


def trtllm_docker_cmd(root: Path, script: Path, trtllm_args: list[str]) -> list[str]:
  """Run trtllm_mistral.py in the official TRT-LLM container (docker or podman)."""
  image = os.environ.get("TRTLLM_IMAGE", "nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc19")
  runtime = os.environ.get("CONTAINER_RUNTIME", "/usr/bin/nvidia-container-runtime")
  ctr, is_podman = None, False
  if shutil.which("docker") and subprocess.run(["docker", "info"], capture_output=True).returncode == 0:
    ctr, is_podman = "docker", False
  elif shutil.which("podman") and subprocess.run(["podman", "info"], capture_output=True).returncode == 0:
    ctr, is_podman = "podman", True
  if not ctr:
    raise RuntimeError("no container engine (docker/podman)")
  cmd = [ctr, "run", "--rm", "--ipc=host"]
  cmd += ["--runtime", runtime] if is_podman else ["--gpus", "all"]
  hf = Path.home() / ".cache/huggingface"
  cmd += [
    "--ulimit", "memlock=-1", "--ulimit", "stack=67108864",
    "-v", f"{hf}:/root/.cache/huggingface",
    "-v", f"{root}:{root}", "-w", str(root),
    image, "python3", str(script), *trtllm_args,
  ]
  return cmd


def bench_trtllm(root: Path, script: Path, trtllm_args: list[str], *, docker: bool) -> list[float]:
  if docker:
    cmd = trtllm_docker_cmd(root, script, trtllm_args)
  else:
    cmd = [sys.executable, str(script), *trtllm_args]
  rates = parse_rates("tensorrt-llm", run_cmd(cmd))
  for i, r in enumerate(rates, 1):
    print(f"tensorrt-llm run {i}: {r:.2f} tok/s", file=sys.stderr)
  return rates


def bench_tinygrad(script: Path, python: Path, model: str, count: int, runs: int, beam: int,
                   weights: Path | None) -> tuple[str, list[float]]:
  env = {**os.environ, "BEAM": str(beam)}
  cmd = [str(python), str(script), "--model", model, "--count", str(count)]
  if weights:
    cmd += ["--weights", str(weights)]
  rates = []
  for i in range(runs + 1):
    out = run_cmd(cmd, env=env, cwd=str(script.parent))
    if i:
      rates.append(parse_rate("tinygrad", out))
      print(f"tinygrad run {i}: {rates[-1]:.2f} tok/s", file=sys.stderr)
  return "tinygrad", rates


def fmt_tok(rates: list[float]) -> str:
  avg = statistics.mean(rates)
  if len(rates) == 1:
    return f"{avg:.1f}"
  sd = statistics.stdev(rates)
  extra = f" ({len(rates)}-run avg, stdev ~{sd:.2f})" if sd < 0.05 else f" (avg, stdev ~{sd:.2f}, n={len(rates)})"
  return f"{avg:.1f}{extra}"


def print_table(results: dict[str, list[float]], count: int) -> None:
  print(f'\nComparison (Mistral-7B-Instruct-v0.2 fp16, 4k context, prompt "{PROMPT}", {count} tokens):\n')
  print(f"Card peak BW: RTX 4090 = {PEAK_BW[0][1]} GB/s, RTX 3080 = {PEAK_BW[1][1]} GB/s.")
  print(f"BW used = model size ({MODEL_GB} GB fp16) × tok/s.\n")
  print("| Engine                       | RTX 4090 tok/s (blog) | 4090 % peak BW | RTX 3080 tok/s (this box) | 3080 % peak BW |")
  print("| ---------------------------- | --------------------- | -------------- | ------------------------- | -------------- |")
  for name in ENGINE_ORDER:
    blog_tok, blog_bw = BLOG_4090[name]
    if name in results:
      avg = statistics.mean(results[name])
      tok_3080, bw_3080 = fmt_tok(results[name]), f"{avg * MODEL_GB / PEAK_BW[1][1] * 100:.1f}%"
    else:
      tok_3080, bw_3080 = "—", "—"
    print(f"| {name:<28} | {blog_tok:<21} | {blog_bw:<14} | {tok_3080:<25} | {bw_3080:<14} |")
  ranked = sorted(((n, statistics.mean(r)) for n, r in results.items()), key=lambda x: x[1], reverse=True)
  print(f"\nRanking: {' > '.join(f'{n.split()[-1] if ' ' in n else n} {v:.1f}' for n, v in ranked)}")


def main() -> None:
  root = Path(__file__).resolve().parent
  home = Path.home()
  p = argparse.ArgumentParser(description="Benchmark Mistral decode and print comparison table")
  p.add_argument("--count", type=int, default=120)
  p.add_argument("--runs", type=int, default=10)
  p.add_argument("--beam", type=int, default=8)
  p.add_argument("--engines", default="all")
  p.add_argument("--python", default=sys.executable)
  p.add_argument("--model", default=DEFAULT_MODEL)
  p.add_argument("--weights", default=None)
  p.add_argument("--yalm-bin", default=str(root / "build/main"))
  p.add_argument("--yalm-ckpt", default=str(root / "mistral-7b-instruct-fp16.yalm"))
  p.add_argument("--llama-cli", default=str(home / "llama.cpp/build/bin/llama-cli"))
  p.add_argument("--gguf", default=str(home / "mistral-7b-instruct-v0.2.fp16.gguf"))
  p.add_argument("--calm-bin", default=str(home / "calm/build/run"))
  p.add_argument("--calm-file", default=str(home / ".cache/mistral-7b-instruct.fp16.calm"))
  p.add_argument("--tinygrad-script", default=str(root / "tinygrad_mistral.py"))
  p.add_argument("--trtllm-script", default=str(root / "trtllm_mistral.py"))
  p.add_argument("--trtllm-docker", action="store_true",
                 help="Run tensorrt-llm via trtllm_docker_bench.sh (needs nvidia-container-toolkit)")
  p.add_argument("--vllm-script", default=str(root / "vllm_mistral.py"))
  p.add_argument("--sglang-script", default=str(root / "sglang_mistral.py"))
  p.add_argument("--vllm-python", default=None, help="Python with vllm (auto-detect ~/.cache/uv venvs)")
  p.add_argument("--sglang-python", default=None, help="Python with sglang (auto-detect ~/.cache/uv venvs)")
  p.add_argument("--nano-vllm-script", default=str(root / "nano_vllm_mistral.py"))
  p.add_argument("--nano-vllm-python", default=None, help="Python with nanovllm (default ref/nano-vllm/.venv)")
  p.add_argument("--torch-compile", dest="torch_compile", action="store_true", default=True,
                 help="Use torch.compile(mode='reduce-overhead') for the transformers engine (default: on).")
  p.add_argument("--no-torch-compile", dest="torch_compile", action="store_false",
                 help="Run transformers in eager mode.")
  args, trtllm_extra = p.parse_known_args()

  weights = Path(args.weights).expanduser() if args.weights else None
  model_id = str(weights) if weights else args.model
  want = set(ENGINE_ORDER) if args.engines == "all" else {e.strip() for e in args.engines.split(",")}
  if bad := want - set(ENGINE_ORDER):
    p.error(f"unknown engines: {', '.join(sorted(bad))}")

  results: dict[str, list[float]] = {}

  if "huggingface transformers" in want:
    label = "transformers (torch.compile)" if args.torch_compile else "huggingface transformers"
    if label in results:
      print(f"skip {label}: already ran with the other torch_compile setting", file=sys.stderr)
    else:
      results[label] = bench_transformers(model_id, args.count, args.runs, torch_compile=args.torch_compile)[1]


  for engine, paths, cmd in [
    ("yalm", (args.yalm_bin, args.yalm_ckpt),
     [args.yalm_bin, args.yalm_ckpt, "-d", "cuda", "-m", "completion", "-i", PROMPT, "-n", str(args.count)]),
    ("llama.cpp", (args.llama_cli, args.gguf),
     [args.llama_cli, "-m", args.gguf, "-c", "4096", "-n", str(args.count), "-p", PROMPT, "--no-display-prompt", "--single-turn"]),
    ("calm", (args.calm_bin, args.calm_file),
     [args.calm_bin, args.calm_file, "-c", "4096", "-n", str(args.count), "-i", PROMPT]),
  ]:
    if engine not in want:
      continue
    if any(not Path(x).exists() for x in paths):
      print(f"skip {engine}: missing path", file=sys.stderr)
      continue
    results[engine] = bench_subprocess(engine, cmd, args.runs)[1]

  if "tinygrad" in want:
    if not Path(args.tinygrad_script).exists():
      print(f"skip tinygrad: missing {args.tinygrad_script}", file=sys.stderr)
    else:
      try:
        import tinygrad  # noqa: F401
        results["tinygrad"] = bench_tinygrad(
          Path(args.tinygrad_script), Path(args.python), args.model, args.count, args.runs, args.beam, weights)[1]
      except ImportError:
        print("skip tinygrad: pip install tinygrad", file=sys.stderr)

  if "tensorrt-llm" in want:
    script = Path(args.trtllm_script)
    if not script.exists():
      print(f"skip tensorrt-llm: missing {script}", file=sys.stderr)
    else:
      trtllm_args = ["--model", model_id, "--count", str(args.count), "--runs", str(args.runs), *trtllm_extra]
      use_docker = args.trtllm_docker
      if not use_docker:
        try:
          import tensorrt_llm  # noqa: F401
        except ImportError:
          print("tensorrt-llm: no native install; using Docker", file=sys.stderr)
          use_docker = True
      try:
        results["tensorrt-llm"] = bench_trtllm(root, script, trtllm_args, docker=use_docker)
      except RuntimeError as e:
        print(f"skip tensorrt-llm: {e}", file=sys.stderr)

  for engine, script_arg, pkg, py_arg in [
    ("vllm", args.vllm_script, "vllm", args.vllm_python),
    ("sglang", args.sglang_script, "sglang", args.sglang_python),
    ("nano-vllm", args.nano_vllm_script, "nanovllm", args.nano_vllm_python),
  ]:
    if engine not in want:
      continue
    script = Path(script_arg)
    if not script.exists():
      print(f"skip {engine}: missing {script}", file=sys.stderr)
      continue
    python = Path(py_arg) if py_arg else find_python_with(pkg)
    if not python:
      print(f"skip {engine}: no Python with {pkg} (see requirements-{engine}.txt)", file=sys.stderr)
      continue
    print(f"{engine}: using {python}", file=sys.stderr)
    results[engine] = bench_external_py(engine, script, python, model_id, args.count, args.runs)[1]

  if not results:
    p.error("no engines ran (check paths and --engines)")
  print_table(results, args.count)


if __name__ == "__main__":
  main()
