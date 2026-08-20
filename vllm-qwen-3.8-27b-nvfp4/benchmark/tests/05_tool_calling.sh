#!/usr/bin/env bash
# 05_tool_calling.sh — Function/tool calling scenario.
# Measures performance with tool definitions and tool call responses.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/../lib.sh"

TEST_NAME="tool_calling"

payload=$(printf '{"model":"%s","stream":true,"temperature":0.3,"max_tokens":512,"messages":[{"role":"user","content":"What is the current weather in Tokyo and Paris? Compare the temperatures."}],"tools":[{"type":"function","function":{"name":"get_weather","description":"Get the current weather for a city","parameters":{"type":"object","properties":{"city":{"type":"string","description":"The name of the city"},"unit":{"type":"string","enum":["celsius","fahrenheit"],"description":"Temperature unit"}},"required":["city"]}}}],"tool_choice":"auto"}' "${MODEL_NAME:-auto}")

echo "▶ ${TEST_NAME}: Tool calling (weather function)"

for i in $(seq 1 ${ITERATIONS:-3}); do
    run_chat_stream "$payload"
    print_result_line "$TEST_NAME" "$i"
done
