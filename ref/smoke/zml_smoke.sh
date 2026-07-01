#!/usr/bin/env bash
# zml (the source of the `llmd` Docker image) smoke.
#
# zml is a Bazel + Zig + MLIR stack. The smallest end-to-end example
# is `bazel run //examples/mnist`; the LLM example is `bazel run
# //examples/llm`. We do NOT have Bazel installed and the LLM example
# (which is the only Mistral-relevant path) only supports Llama 3.1,
# Qwen 3.5, and LFM 2.5 — no Mistral module.
#
# Real smoke at this machine's capability = static:
#   1. The repo is laid out as a Bazel workspace (BUILD/MODULE.bazel exist)
#   2. The MNIST and LLM examples exist
#   3. The LLM example's README lists the supported model families
#   4. No "Mistral" string appears as a model name in the LLM example
#
# What this does NOT do (would need bazelisk + first-time MLIR toolchain
# download of ~1 GB, and a CUDA-13.x host for the actual run):
#   - bazel run //examples/mnist  (the official 30-second smoke)
#   - bazel run //examples/llm   (the LLM path)
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ZML="$HERE/../zml"

# 1. Bazel workspace files
[[ -f "$ZML/MODULE.bazel" ]] || { echo "FAIL: MODULE.bazel missing"; exit 1; }
[[ -f "$ZML/BUILD.bazel" ]] || { echo "FAIL: BUILD.bazel missing"; exit 1; }
echo "Bazel workspace: OK (MODULE.bazel + BUILD.bazel)"

# 2. Examples layout
[[ -d "$ZML/examples/mnist" ]] || { echo "FAIL: examples/mnist missing"; exit 1; }
[[ -d "$ZML/examples/llm" ]] || { echo "FAIL: examples/llm missing"; exit 1; }
echo "examples/mnist: OK"
echo "examples/llm: OK"

# 3. LLM example supported models (parse the README)
LLM_README="$ZML/examples/llm/README.md"
if [[ -f "$LLM_README" ]]; then
  echo
  echo "examples/llm supported models (from README):"
  grep -iE "(Llama|Qwen|LFM|Mistral)" "$LLM_README" \
    | grep -viE "bazel run|^\s*\$|^\s*#" \
    | head -10 | sed 's/^/  /'
fi

# 4. Search for any Mistral mentions
mistral_hits=$(grep -riE "\bmistral\b" "$ZML/examples/llm" "$ZML/zml" 2>/dev/null | head -5 || true)
if [[ -n "$mistral_hits" ]]; then
  echo
  echo "Mistral: NO model file; incidental mentions:"
  echo "$mistral_hits" | sed 's/^/  /'
else
  echo
  echo "Mistral: NO (no mistral references in examples/llm or zml/)"
fi

# 5. bazelisk availability — needed for the real smoke
if which bazel bazelisk 2>/dev/null | head -1 > /dev/null; then
  echo
  echo "bazel: present (real smoke possible; not run in this script)"
else
  echo
  echo "bazel: NOT INSTALLED (real smoke needs bazelisk; would download ~1 GB on first run)"
fi

echo
echo "PASS: zml smoke (static; real bazel run needs bazelisk + ~1 GB MLIR toolchain)"
