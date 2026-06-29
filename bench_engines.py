#!/usr/bin/env python3
"""Find ~/.cache/uv venvs with optional packages installed."""

import subprocess
import sys
from pathlib import Path


def find_python_with(pkg: str) -> Path | None:
  roots = [
    Path.home() / ".cache/uv/environments-v2",
    Path.home() / ".cache/uv/environments-v2/vllm-bench",
    Path.home() / ".cache/uv/environments-v2/sglang-bench",
  ]
  seen: set[Path] = set()
  candidates: list[Path] = []
  for root in roots:
    if not root.is_dir():
      continue
    for pat in ("*/*/bin/python", "*/.venv/bin/python", ".venv/bin/python"):
      candidates.extend(sorted(root.glob(pat)))
  for py in candidates:
    if not py.is_file() or py in seen:
      continue
    seen.add(py)
    r = subprocess.run([str(py), "-c", f"import {pkg}"], capture_output=True)
    if r.returncode == 0:
      return py
  return None


if __name__ == "__main__":
  pkg = sys.argv[1] if len(sys.argv) > 1 else "vllm"
  py = find_python_with(pkg)
  if py:
    print(py)
    sys.exit(0)
  sys.exit(1)
