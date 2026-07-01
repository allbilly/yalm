#!/usr/bin/env bash
# Run all 4 ref/ smoke tests in sequence. Each script is self-contained and
# honest about what it can and cannot do — see ref/smoke/NOTES.md and each
# individual <repo>_smoke.sh for the rationale.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE/../.."   # yalm/ — so relative ref/smoke paths work

PASS=0
FAIL=0
for smoke in flashinfer_smoke.py openinfer_smoke.sh flashqwen_smoke.sh zml_smoke.sh; do
  echo
  echo "================================================================"
  echo "  $smoke"
  echo "================================================================"
  if [[ "$smoke" == *.py ]]; then
    if /home/a/yalm/.venv/bin/python "ref/smoke/$smoke"; then PASS=$((PASS+1)); else FAIL=$((FAIL+1)); fi
  else
    if bash "ref/smoke/$smoke"; then PASS=$((PASS+1)); else FAIL=$((FAIL+1)); fi
  fi
done

echo
echo "================================================================"
echo "  SUMMARY: $PASS passed, $FAIL failed"
echo "================================================================"
exit $FAIL
