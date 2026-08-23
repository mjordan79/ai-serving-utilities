#!/usr/bin/env bash
# 04_reasoning.sh — Math/logic reasoning with thinking enabled.
# Measures performance when chain-of-thought / reasoning parser is active.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/../lib.sh"

TEST_NAME="reasoning"

# Build payload — include thinking kwargs if ENABLE_THINKING is set.
# chat_template_kwargs is the real vLLM/OpenAI body field; it overrides the
# server-side --default-chat-template-kwargs for this request.
if [[ "${ENABLE_THINKING:-true}" == "true" ]]; then
    payload=$(printf '{"model":"%s","stream":true,"temperature":0.3,"max_tokens":2048,"messages":[{"role":"user","content":"A train leaves station A at 60 km/h. Another train leaves station B at 80 km/h. The stations are 420 km apart. They leave at the same time, heading toward each other. How long until they meet, and how far from station A is the meeting point? Show your work step by step."}],"chat_template_kwargs":{"enable_thinking":true}}' "${MODEL_NAME:-auto}")
else
    payload=$(printf '{"model":"%s","stream":true,"temperature":0.3,"max_tokens":1024,"messages":[{"role":"user","content":"A train leaves station A at 60 km/h. Another train leaves station B at 80 km/h. The stations are 420 km apart. They leave at the same time, heading toward each other. How long until they meet, and how far from station A is the meeting point? Show your work step by step."}]}' "${MODEL_NAME:-auto}")
fi

echo "▶ ${TEST_NAME}: Math reasoning ($( [[ "${ENABLE_THINKING:-true}" == "true" ]] && echo 'thinking enabled' || echo 'no thinking' ))"

for i in $(seq 1 ${ITERATIONS:-3}); do
    run_chat_stream "$payload"
    print_result_line "$TEST_NAME" "$i"
done
