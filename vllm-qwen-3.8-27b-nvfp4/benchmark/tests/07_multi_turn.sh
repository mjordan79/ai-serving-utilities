#!/usr/bin/env bash
# 07_multi_turn.sh — Multi-turn conversation with history.
# Measures performance with growing context window across turns.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/../lib.sh"

TEST_NAME="multi_turn"

# Turn 1: Initial question
payload_t1=$(printf '{"model":"%s","stream":true,"temperature":0.7,"max_tokens":256,"messages":[{"role":"user","content":"Explain the difference between TCP and UDP in networking. Keep it concise."}]}' "${MODEL_NAME:-auto}")

# Turn 2: Follow-up with history
payload_t2=$(printf '{"model":"%s","stream":true,"temperature":0.7,"max_tokens":256,"messages":[{"role":"user","content":"Explain the difference between TCP and UDP in networking. Keep it concise."},{"role":"assistant","content":"TCP (Transmission Control Protocol) is connection-oriented, providing reliable, ordered, and error-checked delivery of data. It uses a three-way handshake to establish connections and includes flow control and congestion control mechanisms. UDP (User Datagram Protocol) is connectionless, offering best-effort delivery without guarantees of order or reliability. It has minimal overhead and is faster but less reliable."},{"role":"user","content":"Which one is better for video streaming and why?"}]}' "${MODEL_NAME:-auto}")

# Turn 3: Deeper follow-up with full history
payload_t3=$(printf '{"model":"%s","stream":true,"temperature":0.7,"max_tokens":512,"messages":[{"role":"user","content":"Explain the difference between TCP and UDP in networking. Keep it concise."},{"role":"assistant","content":"TCP (Transmission Control Protocol) is connection-oriented, providing reliable, ordered, and error-checked delivery of data. It uses a three-way handshake to establish connections and includes flow control and congestion control mechanisms. UDP (User Datagram Protocol) is connectionless, offering best-effort delivery without guarantees of order or reliability. It has minimal overhead and is faster but less reliable."},{"role":"user","content":"Which one is better for video streaming and why?"},{"role":"assistant","content":"UDP is generally preferred for video streaming. Here is why: 1) Speed: UDP has lower latency since it does not wait for acknowledgments or retransmit lost packets. 2) Tolerance for loss: In video streaming, missing a few frames is acceptable, but waiting for retransmission causes buffering. 3) Real-time requirements: Live streaming can not afford the delay TCP introduces. Protocols like RTP build on UDP to add minimal reliability features while maintaining low latency."},{"role":"user","content":"Now explain how QUIC protocol tries to get the best of both worlds. What problems does it solve?"}]}' "${MODEL_NAME:-auto}")

echo "▶ ${TEST_NAME}: Multi-turn conversation (3 turns, growing context)"

for i in $(seq 1 ${ITERATIONS:-3}); do
    echo "  Turn 1/3..."
    run_chat_stream "$payload_t1"
    print_result_line "${TEST_NAME}_turn1" "$i"

    echo "  Turn 2/3..."
    run_chat_stream "$payload_t2"
    print_result_line "${TEST_NAME}_turn2" "$i"

    echo "  Turn 3/3..."
    run_chat_stream "$payload_t3"
    print_result_line "${TEST_NAME}_turn3" "$i"
done
