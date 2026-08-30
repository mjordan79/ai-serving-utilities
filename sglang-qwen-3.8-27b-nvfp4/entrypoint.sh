#!/usr/bin/env bash
set -e

API_KEY_FILE="/root/.sglang-key/.api_key"

# Runtime defaults live here so Compose only passes explicit overrides.
ENABLE_API_KEY="${ENABLE_API_KEY:-true}"
ENABLE_REQUEST_METRICS="${ENABLE_REQUEST_METRICS:-true}"
DISABLE_LOG_STATS="${DISABLE_LOG_STATS:-false}"

if [ "$ENABLE_REQUEST_METRICS" = "true" ] && [ "$DISABLE_LOG_STATS" = "true" ]; then
  echo "ERROR: ENABLE_REQUEST_METRICS=true requires DISABLE_LOG_STATS=false." >&2
  exit 1
fi

if [ -n "$HF_TOKEN" ]; then
  export HF_TOKEN
fi

# API key / authentication (ENABLE_API_KEY, default true).
#   true : the server starts with --api-key <key>. Key resolution order:
#           1. $VLLM_API_KEY if set (e.g. from .env): used as-is and written
#              to the key file (volume: sglang-keys), overwriting any previous key.
#           2. otherwise the existing key file is reused (persists across rebuilds).
#           3. otherwise a new sk-<uuid> is generated, persisted, and printed to
#              the logs (shown only on the startup that creates it).
#   false: no authentication (local dev only). --api-key is not passed and
#          VLLM_API_KEY is unset so vLLM's native env-var fallback cannot enable it.
API_KEY_ARGS=()
if [ "$ENABLE_API_KEY" = "true" ]; then
  # Case 1: ENABLE_API_KEY=true and VLLM_API_KEY undefined or empty
  if [ -z "$VLLM_API_KEY" ]; then
    if [ -f "$API_KEY_FILE" ]; then
      # Reuse existing key
      VLLM_API_KEY="$(cat "$API_KEY_FILE")"
      echo "[API] Reusing existing API key from file."
    else
      # Generate new key
      VLLM_API_KEY="sk-$(cat /proc/sys/kernel/random/uuid 2>/dev/null || python3 -c 'import uuid; print(uuid.uuid4())')"
      mkdir -p "$(dirname "$API_KEY_FILE")"
      echo "$VLLM_API_KEY" > "$API_KEY_FILE"
      echo "[API] Generated new API key: $VLLM_API_KEY"
      echo "[API] Save this value — it will not be shown again."
    fi
  else
    # Case 2: ENABLE_API_KEY=true and VLLM_API_KEY is set
    mkdir -p "$(dirname "$API_KEY_FILE")"
    echo "$VLLM_API_KEY" > "$API_KEY_FILE"
    echo "[API] Using API key from environment and writing it to file."
  fi

  export VLLM_API_KEY
  API_KEY_ARGS=(--api-key "$VLLM_API_KEY")

else
  # Case 3: ENABLE_API_KEY=false -> fully disable auth
  unset VLLM_API_KEY
  echo "[API] Auth disabled — server will start without API key."
fi

# --- Build SGLang arguments from environment variables ---
# Every flag has a default here; override through the container environment to change.

# Model
MODEL_NAME="${MODEL_NAME:-RadixArk/Qwen3.8-27B-NVFP4}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8_e4m3}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.94}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-flashinfer}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-1}"
CUDA_GRAPH_MAX_BS_DECODE="${CUDA_GRAPH_MAX_BS_DECODE:-1}"

# Mamba (hybrid attention state)
MAMBA_FULL_MEMORY_RATIO="${MAMBA_FULL_MEMORY_RATIO:-4.59}"
MAMBA_SSM_DTYPE="${MAMBA_SSM_DTYPE:-bfloat16}"
MAMBA_RADIX_CACHE_STRATEGY="${MAMBA_RADIX_CACHE_STRATEGY:-extra_buffer}"

# Parsers
REASONING_PARSER="${REASONING_PARSER:-qwen3}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_coder}"

# Behavior
ALLOW_AUTO_TRUNCATE="${ALLOW_AUTO_TRUNCATE:-true}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-true}"

# Speculative decoding (EAGLE) — OFF by default.
# Enable by setting ENABLE_MTP=true in docker-compose.yml; the flags below
# are then passed through to SGLang.
ENABLE_MTP="${ENABLE_MTP:-false}"
SPECULATIVE_NUM_STEPS="${SPECULATIVE_NUM_STEPS:-3}"
SPECULATIVE_EAGLE_TOPK="${SPECULATIVE_EAGLE_TOPK:-1}"
SPECULATIVE_NUM_DRAFT_TOKENS="${SPECULATIVE_NUM_DRAFT_TOKENS:-4}"

# API
PORT="${PORT:-30000}"

# --- Conditional arguments ---
SPECULATIVE_ARGS=()
if [ "$ENABLE_MTP" = "true" ]; then
  SPECULATIVE_ARGS=(
    --speculative-algorithm EAGLE
    --speculative-num-steps "$SPECULATIVE_NUM_STEPS"
    --speculative-eagle-topk "$SPECULATIVE_EAGLE_TOPK"
    --speculative-num-draft-tokens "$SPECULATIVE_NUM_DRAFT_TOKENS"
    --enable-linear-replayssm-spec
  )
fi

ALLOW_AUTO_TRUNCATE_ARGS=()
if [ "$ALLOW_AUTO_TRUNCATE" = "true" ]; then
  ALLOW_AUTO_TRUNCATE_ARGS=(--allow-auto-truncate)
fi

TRUST_ARGS=()
if [ "$TRUST_REMOTE_CODE" = "true" ]; then
  TRUST_ARGS=(--trust-remote-code)
fi

exec sglang serve "$MODEL_NAME" \
  "${API_KEY_ARGS[@]}" \
  --kv-cache-dtype "$KV_CACHE_DTYPE" \
  --mem-fraction-static "$MEM_FRACTION_STATIC" \
  --attention-backend "$ATTENTION_BACKEND" \
  --max-running-requests "$MAX_RUNNING_REQUESTS" \
  --cuda-graph-max-bs-decode "$CUDA_GRAPH_MAX_BS_DECODE" \
  --reasoning-parser "$REASONING_PARSER" \
  --tool-call-parser "$TOOL_CALL_PARSER" \
  --mamba-full-memory-ratio "$MAMBA_FULL_MEMORY_RATIO" \
  --mamba-radix-cache-strategy "$MAMBA_RADIX_CACHE_STRATEGY" \
  --mamba-ssm-dtype "$MAMBA_SSM_DTYPE" \
  "${SPECULATIVE_ARGS[@]}" \
  "${ALLOW_AUTO_TRUNCATE_ARGS[@]}" \
  "${TRUST_ARGS[@]}" \
  --host 0.0.0.0 \
  --port "$PORT"
