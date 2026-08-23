#!/usr/bin/env bash
# warmup.sh — Pre-compile Triton kernels and warm up the model before benchmarking.
#
# Sends a diverse set of prompts to trigger kernel compilation for:
# - Different sequence lengths (prefill sizes)
# - Different output lengths (decode sizes)
# - Different data paths (reasoning, tool calls, standard chat)
#
# Usage (invoke via bash — the scripts are stored in git without the exec bit):
#   bash warmup.sh              Warmup default model (qwen)
#   bash warmup.sh <model>      Warmup a specific model (qwen|muse)
#
# Auto-detects config from the selected model's .env and docker-compose.yml.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

source "${SCRIPT_DIR}/lib.sh"

# ── Parse arguments: [model] ────────────────────────────────────────────────

MODEL_SELECTOR="${1:-qwen}"
case "$MODEL_SELECTOR" in
    qwen|muse) ;;
    *)
        echo "ERROR: Unknown model '${MODEL_SELECTOR}'. Valid models: qwen, muse."
        exit 1
        ;;
esac

# ── Resolve model target, then auto-detect configuration from its .env ─────

resolve_model_target "$MODEL_SELECTOR"

ENV_FILE="${PROJECT_DIR}/.env"
parse_project_env "$ENV_FILE"
resolve_base_url
recover_api_key

WARMUP_LOG="${SCRIPT_DIR}/results/warmup_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "${SCRIPT_DIR}/results"

log() {
    local msg="[$(date '+%H:%M:%S')] $*"
    echo "$msg"
    echo "$msg" >> "$WARMUP_LOG"
}

# Send a warmup request. Phases are best-effort (no abort on failure), but HTTP
# errors are logged with code and body — a 401 must be visible, not silent.
WARMUP_FAILURES=0
post() {
    local payload="$1" max_time="$2" label="$3"
    local bodyfile codefile
    bodyfile=$(mktemp 2>/dev/null || printf '/tmp/warmup_body_%s' "$$")
    codefile=$(mktemp 2>/dev/null || printf '/tmp/warmup_code_%s' "$$")
    local rc=0
    curl -sS -k --max-time "$max_time" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer ${API_KEY}" \
        -d "$payload" \
        -o "$bodyfile" -w '%{http_code}' > "$codefile" \
        "${BASE_URL}/v1/chat/completions" 2>/dev/null || rc=$?
    local http_code
    http_code=$(cat "$codefile" 2>/dev/null || echo "000")
    if [[ $rc -eq 0 && "$http_code" == "200" ]]; then
        log "  ✓ ${label}"
    else
        local body
        body=$(head -c 120 "$bodyfile" 2>/dev/null || true)
        log "  ⚠ ${label} FAILED (HTTP ${http_code}: ${body})"
        WARMUP_FAILURES=$((WARMUP_FAILURES + 1))
        # Not fatal: remaining phases may still compile useful kernels.
    fi
    rm -f "$bodyfile" "$codefile"
}

log "═══════════════════════════════════════════════════════════"
log "  WARMUP — Triton kernel pre-compilation"
log "  Deployment: ${MODEL_SELECTOR}"
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
    post "$payload" 60 "Short request ${i}/3"
done

# ── Phase 3: Medium prefill, medium decode ─────────────────────────────────
log "Phase 3: Medium prefill + medium decode..."
for i in 1 2; do
    payload=$(printf '{"model":"%s","stream":false,"temperature":0.5,"max_tokens":512,"messages":[{"role":"user","content":"Write a Python function that implements a binary search tree with insert, delete, and search operations. Include type hints and docstrings. The tree should handle duplicate values by storing a count. Provide the complete class implementation with proper error handling."}]}' "${MODEL_NAME:-auto}")
    post "$payload" 120 "Medium request ${i}/2"
done

# ── Phase 4: Long prefill (context window warmup) ───────────────────────────
log "Phase 4: Long prefill (~2000 tokens)..."
# Generate a ~2000 token context (pure bash)
long_text=""
for i in $(seq 1 20); do
    long_text+="Chapter ${i}: The architecture of modern large language models relies on transformer-based attention mechanisms. The self-attention layer computes query, key, and value projections for each token in the sequence. FlashAttention optimizes this computation by minimizing HBM memory I/O through tiling and re-computation. The feed-forward network uses SwiGLU activation with gated linear units. Layer normalization is applied pre-normalization style before each sub-layer. The residual connections help gradient flow through deep networks. RoPE (Rotary Positional Embedding) provides relative position information. Grouped Query Attention reduces the number of key-value heads for inference efficiency. "
done

escaped_ctx=$(json_escape "$long_text")
    # NOTE: \\n in the printf format → literal backslash-n in the JSON payload
    # (escaped newline). A real newline would break the JSON string → HTTP 400.
    payload=$(printf '{"model":"%s","stream":false,"temperature":0.3,"max_tokens":128,"messages":[{"role":"user","content":"Summarize this technical document in 3 sentences:\\n\\n%s"}]}' "${MODEL_NAME:-auto}" "$escaped_ctx")
    post "$payload" 180 "Long prefill"

# ── Phase 5: Long decode (sustained generation) ─────────────────────────────
log "Phase 5: Long decode (2048 token output)..."
payload=$(printf '{"model":"%s","stream":false,"temperature":0.7,"max_tokens":2048,"messages":[{"role":"user","content":"Write a detailed technical tutorial on building a REST API with FastAPI in Python. Cover dependency injection, middleware, background tasks, and database integration with SQLAlchemy. Include code examples for each section."}]}' "${MODEL_NAME:-auto}")
post "$payload" 300 "Long decode"

# ── Phase 6: Reasoning path warmup ──────────────────────────────────────────
log "Phase 6: Reasoning/thinking path warmup..."
# chat_template_kwargs is the real JSON body field (extra_body is a no-op in raw
# JSON — it is a Python-SDK concept). Overrides the server-side default.
think_flag="true"
[[ "${ENABLE_THINKING:-true}" == "true" ]] || think_flag="false"
payload=$(printf '{"model":"%s","stream":false,"temperature":0.3,"max_tokens":1024,"messages":[{"role":"user","content":"Solve: A water tank has two inlet pipes and one outlet pipe. Pipe A fills the tank in 3 hours, Pipe B in 5 hours, and the outlet empties it in 8 hours. If all three are open simultaneously, how long to fill an empty tank? Show step by step."}],"chat_template_kwargs":{"enable_thinking":%s}}' "${MODEL_NAME:-auto}" "$think_flag")
post "$payload" 180 "Reasoning path (thinking=${think_flag})"

# ── Phase 7: Tool calling path warmup ───────────────────────────────────────
log "Phase 7: Tool calling path warmup..."
payload=$(printf '{"model":"%s","stream":false,"temperature":0.3,"max_tokens":256,"messages":[{"role":"user","content":"What is the weather in London?"}],"tools":[{"type":"function","function":{"name":"get_weather","description":"Get weather for a city","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]}' "${MODEL_NAME:-auto}")
post "$payload" 60 "Tool calling path"

# ── Phase 8: Streaming path warmup ──────────────────────────────────────────
log "Phase 8: Streaming path warmup..."
payload=$(printf '{"model":"%s","stream":true,"temperature":0.7,"max_tokens":256,"messages":[{"role":"user","content":"Write a haiku about artificial intelligence."}]}' "${MODEL_NAME:-auto}")
post "$payload" 60 "Streaming path"

# ── Done ────────────────────────────────────────────────────────────────────
log "═══════════════════════════════════════════════════════════"
if [[ $WARMUP_FAILURES -eq 0 ]]; then
    log "  WARMUP COMPLETE — Model is ready for benchmarking"
    log "Log: ${WARMUP_LOG}"
else
    log "  WARMUP FAILED — ${WARMUP_FAILURES} phase(s) did not succeed (see log)."
    log "  The model is NOT confirmed ready for benchmarking."
    log "Log: ${WARMUP_LOG}"
    exit 1
fi
