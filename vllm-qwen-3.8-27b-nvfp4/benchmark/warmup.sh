#!/usr/bin/env bash
# warmup.sh — Pre-compile Triton kernels and warm up the model before benchmarking.
#
# Sends a diverse set of prompts to trigger kernel compilation for:
# - Different sequence lengths (prefill sizes)
# - Different output lengths (decode sizes)
# - Different data paths (reasoning, tool calls, standard chat)
#
# Usage: ./warmup.sh (auto-detects config from project .env)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="${SCRIPT_DIR}/.."

# ── Auto-detect configuration from project .env ──────────────────────────────

ENV_FILE="${PROJECT_DIR}/.env"

if [[ ! -f "$ENV_FILE" ]]; then
    echo "ERROR: Project .env not found at ${ENV_FILE}"
    exit 1
fi

while IFS='=' read -r key value; do
    [[ "$key" =~ ^[[:space:]]*# ]] && continue
    [[ -z "$key" ]] && continue
    key=$(echo "$key" | xargs)
    value=$(echo "$value" | xargs | sed 's/^"//;s/"$//')
    [[ -z "$key" ]] && continue
    export "$key=$value"
done < "$ENV_FILE"

if [[ -n "${LETSENCRYPT_DOMAIN:-}" ]]; then
    if curl -sS -k --max-time 5 --output /dev/null "https://${LETSENCRYPT_DOMAIN}/health" 2>/dev/null; then
        BASE_URL="https://${LETSENCRYPT_DOMAIN}"
    else
        echo "INFO: ${LETSENCRYPT_DOMAIN} not reachable, falling back to https://localhost"
        BASE_URL="https://localhost"
    fi
else
    BASE_URL="http://localhost:1235"
fi

API_KEY=""
if command -v docker &>/dev/null; then
    API_KEY=$(docker compose -f "${PROJECT_DIR}/docker-compose.yml" \
        exec --no-deps vllm-server cat /root/.vllm-key/.api_key 2>/dev/null || true)
fi
if [[ -z "$API_KEY" ]] && command -v docker &>/dev/null; then
    API_KEY=$(docker exec vllm-server cat /root/.vllm-key/.api_key 2>/dev/null || true)
fi
if [[ -z "$API_KEY" ]]; then
    echo -n "Enter VLLM_API_KEY: " >&2
    read -r API_KEY >&2
fi
export API_KEY

source "${SCRIPT_DIR}/lib.sh"

WARMUP_LOG="${SCRIPT_DIR}/results/warmup_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "${SCRIPT_DIR}/results"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "═══════════════════════════════════════════════════════════"
log "  WARMUP — Triton kernel pre-compilation"
log "  Target: ${BASE_URL}"
log "═══════════════════════════════════════════════════════════"

# ── Phase 1: Health check ───────────────────────────────────────────────────
log "Phase 1: Health check..."
http_code=$(curl -sS -k --max-time 30 -o /dev/null -w "%{http_code}" "${BASE_URL}/health" 2>/dev/null || echo "000")
if [[ "$http_code" == "200" ]]; then
    # Verify API is functional with /v1/models (requires auth)
    model_info=$(curl -sS -k --max-time 30 \
        -H "Authorization: Bearer ${API_KEY}" \
        "${BASE_URL}/v1/models" 2>/dev/null || echo "")

    model_id=$(echo "$model_info" | gawk -F'"' '/"id"/{for(i=1;i<=NF;i++) if($i=="id") print $(i+2)}' | head -1)
    if [[ -n "$model_id" ]]; then
        log "  ✓ Model ready: ${model_id}"
    else
        log "  ⚠ Health endpoint OK (HTTP 200) but /v1/models returned unexpected response"
        log "  ✓ Proceeding with warmup anyway"
    fi
else
    log "  ✗ Model not responding (HTTP $http_code). Is the server running?"
    exit 1
fi

# ── Phase 2: Short prefill, short decode ────────────────────────────────────
log "Phase 2: Short prefill + short decode (kernel warmup)..."
for i in 1 2 3; do
    payload=$(printf '{"model":"%s","stream":false,"temperature":0.7,"max_tokens":64,"messages":[{"role":"user","content":"Warmup prompt #%d. What is 2+2?"}]}' "${MODEL_NAME:-auto}" "$i")
    curl -sS -k --max-time 60 \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer ${API_KEY}" \
        -d "$payload" "${BASE_URL}/v1/chat/completions" >/dev/null 2>&1
    log "  ✓ Short request ${i}/3"
done

# ── Phase 3: Medium prefill, medium decode ─────────────────────────────────
log "Phase 3: Medium prefill + medium decode..."
for i in 1 2; do
    payload=$(printf '{"model":"%s","stream":false,"temperature":0.5,"max_tokens":512,"messages":[{"role":"user","content":"Write a Python function that implements a binary search tree with insert, delete, and search operations. Include type hints and docstrings. The tree should handle duplicate values by storing a count. Provide the complete class implementation with proper error handling."}]}' "${MODEL_NAME:-auto}")
    curl -sS -k --max-time 120 \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer ${API_KEY}" \
        -d "$payload" "${BASE_URL}/v1/chat/completions" >/dev/null 2>&1
    log "  ✓ Medium request ${i}/2"
done

# ── Phase 4: Long prefill (context window warmup) ───────────────────────────
log "Phase 4: Long prefill (~2000 tokens)..."
# Generate a ~2000 token context (pure bash)
long_text=""
for i in $(seq 1 20); do
    long_text+="Chapter ${i}: The architecture of modern large language models relies on transformer-based attention mechanisms. The self-attention layer computes query, key, and value projections for each token in the sequence. FlashAttention optimizes this computation by minimizing HBM memory I/O through tiling and re-computation. The feed-forward network uses SwiGLU activation with gated linear units. Layer normalization is applied pre-normalization style before each sub-layer. The residual connections help gradient flow through deep networks. RoPE (Rotary Positional Embedding) provides relative position information. Grouped Query Attention reduces the number of key-value heads for inference efficiency. "
done

escaped_ctx=$(json_escape "$long_text")
    payload=$(printf '{"model":"%s","stream":false,"temperature":0.3,"max_tokens":128,"messages":[{"role":"user","content":"Summarize this technical document in 3 sentences:\n\n%s"}]}' "${MODEL_NAME:-auto}" "$escaped_ctx")
curl -sS -k --max-time 180 \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer ${API_KEY}" \
    -d "$payload" "${BASE_URL}/v1/chat/completions" >/dev/null 2>&1
log "  ✓ Long prefill done"

# ── Phase 5: Long decode (sustained generation) ─────────────────────────────
log "Phase 5: Long decode (2048 token output)..."
payload=$(printf '{"model":"%s","stream":false,"temperature":0.7,"max_tokens":2048,"messages":[{"role":"user","content":"Write a detailed technical tutorial on building a REST API with FastAPI in Python. Cover dependency injection, middleware, background tasks, and database integration with SQLAlchemy. Include code examples for each section."}]}' "${MODEL_NAME:-auto}")
curl -sS -k --max-time 300 \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer ${API_KEY}" \
    -d "$payload" "${BASE_URL}/v1/chat/completions" >/dev/null 2>&1
log "  ✓ Long decode done"

# ── Phase 6: Reasoning path warmup ──────────────────────────────────────────
log "Phase 6: Reasoning/thinking path warmup..."
if [[ "${ENABLE_THINKING:-true}" == "true" ]]; then
    payload=$(printf '{"model":"%s","stream":false,"temperature":0.3,"max_tokens":1024,"messages":[{"role":"user","content":"Solve: A water tank has two inlet pipes and one outlet pipe. Pipe A fills the tank in 3 hours, Pipe B in 5 hours, and the outlet empties it in 8 hours. If all three are open simultaneously, how long to fill an empty tank? Show step by step."}],"extra_body":{"thinking":{"type":"enabled"}}}' "${MODEL_NAME:-auto}")
    curl -sS -k --max-time 180 \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer ${API_KEY}" \
        -d "$payload" "${BASE_URL}/v1/chat/completions" >/dev/null 2>&1
    log "  ✓ Reasoning path done"
fi

# ── Phase 7: Tool calling path warmup ───────────────────────────────────────
log "Phase 7: Tool calling path warmup..."
payload=$(printf '{"model":"%s","stream":false,"temperature":0.3,"max_tokens":256,"messages":[{"role":"user","content":"What is the weather in London?"}],"tools":[{"type":"function","function":{"name":"get_weather","description":"Get weather for a city","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]}' "${MODEL_NAME:-auto}")
curl -sS -k --max-time 60 \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer ${API_KEY}" \
    -d "$payload" "${BASE_URL}/v1/chat/completions" >/dev/null 2>&1
log "  ✓ Tool calling path done"

# ── Phase 8: Streaming path warmup ──────────────────────────────────────────
log "Phase 8: Streaming path warmup..."
payload=$(printf '{"model":"%s","stream":true,"temperature":0.7,"max_tokens":256,"messages":[{"role":"user","content":"Write a haiku about artificial intelligence."}]}' "${MODEL_NAME:-auto}")
# Consume the stream fully
curl -sS -k --max-time 60 \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer ${API_KEY}" \
    -d "$payload" "${BASE_URL}/v1/chat/completions" >/dev/null 2>&1
log "  ✓ Streaming path done"

# ── Done ────────────────────────────────────────────────────────────────────
log "═══════════════════════════════════════════════════════════"
log "  WARMUP COMPLETE — Model is ready for benchmarking"
log "═══════════════════════════════════════════════════════════"
log "Log: ${WARMUP_LOG}"

# Save a copy of this output
echo "$(date): Warmup completed successfully" >> "$WARMUP_LOG"
