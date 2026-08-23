#!/usr/bin/env bash
# 08_mixed_workload.sh — Sequential mix of different task types.
# Simulates a realistic workload pattern: short Q&A → code → reasoning → tool call.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/../lib.sh"

TEST_NAME="mixed_workload"

# Task A: Short Q&A
payload_a=$(printf '{"model":"%s","stream":true,"temperature":0.7,"max_tokens":128,"messages":[{"role":"user","content":"What is quantum entanglement? Explain in one sentence."}]}' "${MODEL_NAME:-auto}")

# Task B: Code snippet
payload_b=$(printf '{"model":"%s","stream":true,"temperature":0.3,"max_tokens":512,"messages":[{"role":"user","content":"Write a Rust function that implements binary search on a sorted slice, returning an Option<usize>."}]}' "${MODEL_NAME:-auto}")

# Task C: Reasoning
payload_c=$(printf '{"model":"%s","stream":true,"temperature":0.3,"max_tokens":1024,"messages":[{"role":"user","content":"If f(x) = 3x² + 2x - 5, find the derivative and evaluate it at x = 4. Show the derivation."}]}' "${MODEL_NAME:-auto}")

# Task D: Translation
payload_d=$(printf '{"model":"%s","stream":true,"temperature":0.5,"max_tokens":256,"messages":[{"role":"user","content":"Translate the following English paragraph to Italian, preserving the formal tone: The committee has reached a consensus regarding the proposed budget allocations for the upcoming fiscal year. All members are requested to review the attached documentation before the final vote."}]}' "${MODEL_NAME:-auto}")

echo "▶ ${TEST_NAME}: Mixed workload (4 task types)"

for i in $(seq 1 ${ITERATIONS:-3}); do
    echo "  Task A: Short Q&A..."
    run_chat_stream "$payload_a"
    print_result_line "${TEST_NAME}_qna" "$i"

    echo "  Task B: Code (Rust)..."
    run_chat_stream "$payload_b"
    print_result_line "${TEST_NAME}_code" "$i"

    echo "  Task C: Reasoning (calculus)..."
    run_chat_stream "$payload_c"
    print_result_line "${TEST_NAME}_reasoning" "$i"

    echo "  Task D: Translation (EN→IT)..."
    run_chat_stream "$payload_d"
    print_result_line "${TEST_NAME}_translation" "$i"
done
