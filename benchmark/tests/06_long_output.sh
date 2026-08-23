#!/usr/bin/env bash
# 06_long_output.sh — Prompt that generates a long response.
# Measures sustained generation throughput (decoding performance).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/../lib.sh"

TEST_NAME="long_output"

payload=$(printf '{"model":"%s","stream":true,"temperature":0.7,"max_tokens":4096,"messages":[{"role":"user","content":"Write a comprehensive blog post about the evolution of distributed consensus algorithms. Start with the two-phase commit, move through Paxos and Raft, and discuss modern approaches like Narwhal and Bullshark. Include a comparison table and discuss the trade-offs between latency, throughput, and fault tolerance. Write in a technical but accessible style."}]}' "${MODEL_NAME:-auto}")

echo "▶ ${TEST_NAME}: Long output generation (up to 4096 tokens)"

for i in $(seq 1 ${ITERATIONS:-3}); do
    run_chat_stream "$payload"
    print_result_line "$TEST_NAME" "$i"
done
