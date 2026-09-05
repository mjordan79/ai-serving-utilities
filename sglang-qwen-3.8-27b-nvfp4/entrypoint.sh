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

# Model card defaults: Gittensor Qwen3.8 with DSpark v2.
MODEL_NAME="${MODEL_NAME:-gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090}"
TP_SIZE="${TP_SIZE:-1}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-262144}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-2048}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8_e4m3}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.90}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-flashinfer}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-2}"
MAX_MAMBA_CACHE_SIZE="${MAX_MAMBA_CACHE_SIZE:-12}"

# Mamba (hybrid attention state)
MAMBA_SSM_DTYPE="${MAMBA_SSM_DTYPE:-bfloat16}"
MAMBA_RADIX_CACHE_STRATEGY="${MAMBA_RADIX_CACHE_STRATEGY:-extra_buffer_lazy}"

# Parsers
REASONING_PARSER="${REASONING_PARSER:-qwen3}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_coder}"

# Behavior
ALLOW_AUTO_TRUNCATE="${ALLOW_AUTO_TRUNCATE:-true}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-true}"

# Vision tower OFF by default (same convention as the sibling vLLM stack):
# --json-model-override-args is always passed. The default override sets
# language_model_only=true so the engine skips the vision encoder (its
# weights are never loaded, freeing VRAM for the KV cache; multimodal
# requests are rejected). To serve images again, set
# JSON_MODEL_OVERRIDE_ARGS={} in .env (or any custom JSON string).
DEFAULT_JSON_MODEL_OVERRIDE='{"language_model_only": true}'
JSON_MODEL_OVERRIDE_ARGS="${JSON_MODEL_OVERRIDE_ARGS:-$DEFAULT_JSON_MODEL_OVERRIDE}"

# Speculative decoding: DSpark v2 is the fastest model-card profile.
SPECULATIVE_ALGORITHM="${SPECULATIVE_ALGORITHM:-DSPARK}"
SPECULATIVE_DRAFT_MODEL_PATH="${SPECULATIVE_DRAFT_MODEL_PATH:-gittensor-model-hub/Qwen3.8-27B-DSpark-NVFP4}"
SPECULATIVE_DSPARK_BLOCK_SIZE="${SPECULATIVE_DSPARK_BLOCK_SIZE:-7}"
SPECULATIVE_DRAFT_MODEL_QUANTIZATION="${SPECULATIVE_DRAFT_MODEL_QUANTIZATION:-modelopt_fp4}"

# API
PORT="${PORT:-30000}"

# --- Conditional arguments ---
SPECULATIVE_ARGS=()
if [ "$SPECULATIVE_ALGORITHM" = "DSPARK" ]; then
  SPECULATIVE_ARGS=(
    --speculative-algorithm DSPARK
    --speculative-draft-model-path "$SPECULATIVE_DRAFT_MODEL_PATH"
    --speculative-dspark-block-size "$SPECULATIVE_DSPARK_BLOCK_SIZE"
    --speculative-draft-model-quantization "$SPECULATIVE_DRAFT_MODEL_QUANTIZATION"
  )
elif [ "$SPECULATIVE_ALGORITHM" != "none" ]; then
  echo "ERROR: SPECULATIVE_ALGORITHM must be DSPARK or none." >&2
  exit 1
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
  --tp-size "$TP_SIZE" \
  --context-length "$CONTEXT_LENGTH" \
  --chunked-prefill-size "$CHUNKED_PREFILL_SIZE" \
  --kv-cache-dtype "$KV_CACHE_DTYPE" \
  --mem-fraction-static "$MEM_FRACTION_STATIC" \
  --attention-backend "$ATTENTION_BACKEND" \
  --max-running-requests "$MAX_RUNNING_REQUESTS" \
  --max-mamba-cache-size "$MAX_MAMBA_CACHE_SIZE" \
  --reasoning-parser "$REASONING_PARSER" \
  --tool-call-parser "$TOOL_CALL_PARSER" \
  --mamba-radix-cache-strategy "$MAMBA_RADIX_CACHE_STRATEGY" \
  --mamba-ssm-dtype "$MAMBA_SSM_DTYPE" \
  --json-model-override-args "$JSON_MODEL_OVERRIDE_ARGS" \
  "${SPECULATIVE_ARGS[@]}" \
  "${ALLOW_AUTO_TRUNCATE_ARGS[@]}" \
  "${TRUST_ARGS[@]}" \
  --host 0.0.0.0 \
  --port "$PORT"
