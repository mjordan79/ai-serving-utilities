#!/usr/bin/env bash
# 01_simple_chat.sh — Short single-turn chat, no reasoning, no tools.
# Measures baseline latency and throughput for simple Q&A.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/../lib.sh"

TEST_NAME="simple_chat"

payload=$(printf '{"model":"%s","stream":true,"temperature":0.7,"max_tokens":256,"messages":[{"role":"user","content":"What is the capital of France and what is its population?"}]}' "${MODEL_NAME:-auto}")

echo "▶ ${TEST_NAME}: Short single-turn Q&A"

for i in $(seq 1 ${ITERATIONS:-3}); do
    run_chat_stream "$payload"
    print_result_line "$TEST_NAME" "$i"
done
