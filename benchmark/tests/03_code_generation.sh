#!/usr/bin/env bash
# 03_code_generation.sh — Code generation task.
# Measures performance on code output (typically higher token count).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/../lib.sh"

TEST_NAME="code_generation"

payload=$(printf '{"model":"%s","stream":true,"temperature":0.3,"max_tokens":1024,"messages":[{"role":"user","content":"Write a Python function that implements a thread-safe LRU (Least Recently Used) cache with a configurable maximum size. The cache should support get and put operations with O(1) time complexity. Include type hints and comprehensive docstrings. Handle edge cases like setting a key to None and accessing a non-existent key."}]}' "${MODEL_NAME:-auto}")

echo "▶ ${TEST_NAME}: Code generation (Python LRU cache with TTL)"

for i in $(seq 1 ${ITERATIONS:-3}); do
    run_chat_stream "$payload"
    print_result_line "$TEST_NAME" "$i"
done
