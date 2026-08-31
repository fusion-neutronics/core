#!/usr/bin/env bash
# Run the yamc-gpu test suite with each test in its own process.
#
# Running all of yamc-gpu's GPU tests in a single test-binary process
# degrades the GPU device state as tests accumulate, so a different
# subset spuriously fails each run (kernels return zero tallies). This
# is purely in-process exhaustion -- every test passes in isolation.
# Giving each test a fresh process (hence a fresh GPU context) makes
# the suite deterministic.
set -euo pipefail

cd "$(dirname "$0")/.."

# Build the test binary without running, then locate the freshest one.
cargo test -p yamc-gpu --lib --no-run
BIN=$(ls -t target/debug/deps/yamc_gpu-* 2>/dev/null | grep -v '\.d$' | head -1)
if [ -z "${BIN:-}" ] || [ ! -x "$BIN" ]; then
  echo "test-gpu.sh: could not locate the yamc-gpu test binary" >&2
  exit 1
fi

mapfile -t TESTS < <("$BIN" --list 2>/dev/null | grep ': test$' | sed 's/: test$//')
if [ "${#TESTS[@]}" -eq 0 ]; then
  echo "test-gpu.sh: no tests found in $BIN" >&2
  exit 1
fi

echo "Running ${#TESTS[@]} yamc-gpu tests, one process each..."
fail=0
failed=()
for t in "${TESTS[@]}"; do
  if ! "$BIN" --exact "$t" --test-threads=1 >/dev/null 2>&1; then
    fail=$((fail + 1))
    failed+=("$t")
    echo "FAIL: $t"
  fi
done

if [ "$fail" -gt 0 ]; then
  echo "test-gpu.sh: $fail/${#TESTS[@]} yamc-gpu tests failed:" >&2
  printf '  %s\n' "${failed[@]}" >&2
  exit 1
fi

echo "test-gpu.sh: all ${#TESTS[@]} yamc-gpu tests passed (per-process)"
