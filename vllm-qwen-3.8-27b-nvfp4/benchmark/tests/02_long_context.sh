#!/usr/bin/env bash
# 02_long_context.sh — Long prefill with short answer.
# Measures how the model handles large input context (prefill performance).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/../lib.sh"

TEST_NAME="long_context"

# Generate a long context: ~4000 tokens of structured text (pure bash, no Python)
long_context=""
for i in $(seq 1 40); do
    long_context+="Section ${i}: This is a detailed technical analysis of distributed systems architecture. The system comprises multiple microservices communicating over gRPC with protobuf schemas. Each service maintains its own state in a PostgreSQL database with read replicas. The consensus protocol uses Raft with a quorum of 3 nodes. Monitoring is done via Prometheus with Grafana dashboards. Logging follows the structured JSON format with correlation IDs. The deployment pipeline uses GitOps with ArgoCD managing Kubernetes manifests. Security is enforced through mTLS between services and OAuth2 for external APIs. "
done

escaped_ctx=$(json_escape "$long_context")
payload=$(printf '{"model":"%s","stream":true,"temperature":0.3,"max_tokens":128,"messages":[{"role":"user","content":"Given the following document, provide a brief summary in 3 bullet points:\\n\\n%s"}]}' "${MODEL_NAME:-auto}" "$escaped_ctx")

echo "▶ ${TEST_NAME}: Long prefill (~4000 input tokens), short answer"

for i in $(seq 1 ${ITERATIONS:-3}); do
    run_chat_stream "$payload"
    print_result_line "$TEST_NAME" "$i"
done
