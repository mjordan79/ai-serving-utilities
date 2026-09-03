#!/usr/bin/env bash
set -e

API_KEY_FILE="/root/.vllm-key/.api_key"

# Runtime defaults live here so Compose only passes explicit overrides.
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS="${VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS:-1}"
# Point the FlashInfer autotune cache inside the vLLM cache root so a single
# mounted volume (vllm-caches -> /root/.cache/vllm) persists everything
# (torch.compile / Inductor / Triton + FlashInfer) across rebuilds/recreates.
export VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR="${VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR:-/root/.cache/vllm/flashinfer_autotune_cache}"
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
#              to the key file (volume: vllm-keys), overwriting any previous key.
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

# --- Build vLLM arguments from environment variables ---
# Every flag has a default here; override through the container environment to change.

# Security: cap the `n` parameter on /v1 completions. vLLM defaults
# VLLM_MAX_N_SEQUENCES to 16384 — a single request could ask for 16384
# parallel sequences. Docs suggest 64/128 for public deployments; 16 is
# plenty here (MAX_NUM_SEQS serializes generation anyway).
VLLM_MAX_N_SEQUENCES="${VLLM_MAX_N_SEQUENCES:-16}"
export VLLM_MAX_N_SEQUENCES

# Model
MODEL_NAME="${MODEL_NAME:-unsloth/Qwen3-8-27B-NVFP4}"
DTYPE="${DTYPE:-auto}"
TP_SIZE="${TP_SIZE:-1}"

# Memory & Context
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.94}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-116800}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8_e4m3}"

# Throughput
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-6144}"

# Attention & Performance
ATTENTION_BACKEND="${ATTENTION_BACKEND:-flashinfer}"
PERFORMANCE_MODE="${PERFORMANCE_MODE:-interactivity}"

# Features
ENABLE_MTP="${ENABLE_MTP:-true}"
MTP_NUM_SPECULATIVE_TOKENS="${MTP_NUM_SPECULATIVE_TOKENS:-3}"
ENABLE_CHUNKED_PREFILL="${ENABLE_CHUNKED_PREFILL:-true}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-true}"
ENABLE_HYBRID_KV_CACHE_MANAGER="${ENABLE_HYBRID_KV_CACHE_MANAGER:-true}"
ENABLE_AUTO_TOOL_CHOICE="${ENABLE_AUTO_TOOL_CHOICE:-true}"
ENABLE_PROMPT_TOKENS_DETAILS="${ENABLE_PROMPT_TOKENS_DETAILS:-true}"

# Parsers
REASONING_PARSER="${REASONING_PARSER:-qwen3}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_coder}"
DEFAULT_ENABLE_THINKING="${DEFAULT_ENABLE_THINKING:-true}"
# Keep historical assistant thinking in multi-turn context. Model default is true (preserve);
# false strips it to save context. Inert unless the client echoes reasoning back.
# Per-request chat_template_kwargs can still override.
DEFAULT_PRESERVE_THINKING="${DEFAULT_PRESERVE_THINKING:-true}"

# Loading
SAFETENSORS_LOAD_STRATEGY="${SAFETENSORS_LOAD_STRATEGY:-prefetch}"
SKIP_MM_PROFILING="${SKIP_MM_PROFILING:-true}"
# Default = Unsloth variant (compressed-tensors). NVIDIA variant: QUANTIZATION=modelopt in .env.
QUANTIZATION="${QUANTIZATION:-compressed-tensors}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-true}"

# API
PORT="${PORT:-8000}"

# --- Conditional arguments ---
SPECULATIVE_ARGS=()
if [ "$ENABLE_MTP" = "true" ]; then
  SPECULATIVE_ARGS=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$MTP_NUM_SPECULATIVE_TOKENS}")
fi

CHUNKED_PREFILL_ARGS=()
if [ "$ENABLE_CHUNKED_PREFILL" = "true" ]; then
  CHUNKED_PREFILL_ARGS=(--enable-chunked-prefill)
fi

PREFIX_CACHING_ARGS=()
if [ "$ENABLE_PREFIX_CACHING" = "true" ]; then
  PREFIX_CACHING_ARGS=(--enable-prefix-caching)
fi

HYBRID_KV_ARGS=()
if [ "$ENABLE_HYBRID_KV_CACHE_MANAGER" = "true" ]; then
  HYBRID_KV_ARGS=(--no-disable-hybrid-kv-cache-manager)
fi

AUTO_TOOL_CHOICE_ARGS=()
if [ "$ENABLE_AUTO_TOOL_CHOICE" = "true" ]; then
  AUTO_TOOL_CHOICE_ARGS=(--enable-auto-tool-choice)
fi

PROMPT_TOKENS_ARGS=()
if [ "$ENABLE_PROMPT_TOKENS_DETAILS" = "true" ]; then
  PROMPT_TOKENS_ARGS=(--enable-prompt-tokens-details)
fi

REQUEST_METRICS_ARGS=()
if [ "$ENABLE_REQUEST_METRICS" = "true" ]; then
  REQUEST_METRICS_ARGS=(--enable-per-request-metrics)
fi

DISABLE_LOG_STATS_ARGS=()
if [ "$DISABLE_LOG_STATS" = "true" ]; then
  DISABLE_LOG_STATS_ARGS=(--disable-log-stats)
fi

SKIP_MM_ARGS=()
if [ "$SKIP_MM_PROFILING" = "true" ]; then
  SKIP_MM_ARGS=(--skip-mm-profiling)
fi

TRUST_ARGS=()
if [ "$TRUST_REMOTE_CODE" = "true" ]; then
  TRUST_ARGS=(--trust-remote-code)
fi

LANG_MODEL_ARGS=()
if [ "${LANGUAGE_MODEL_ONLY-}" != "false" ]; then
  LANG_MODEL_ARGS=(--language-model-only)
fi

CHAT_TEMPLATE_KWARGS='{"enable_thinking": '"$DEFAULT_ENABLE_THINKING"', "preserve_thinking": '"$DEFAULT_PRESERVE_THINKING"'}'

exec vllm serve "$MODEL_NAME" \
  "${API_KEY_ARGS[@]}" \
  --dtype "$DTYPE" \
  --safetensors-load-strategy "$SAFETENSORS_LOAD_STRATEGY" \
  --tensor-parallel-size "$TP_SIZE" \
  --attention-backend "$ATTENTION_BACKEND" \
  --performance-mode "$PERFORMANCE_MODE" \
  "${LANG_MODEL_ARGS[@]}" \
  "${SKIP_MM_ARGS[@]}" \
  --kv-cache-dtype "$KV_CACHE_DTYPE" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  "${CHUNKED_PREFILL_ARGS[@]}" \
  "${PREFIX_CACHING_ARGS[@]}" \
  "${HYBRID_KV_ARGS[@]}" \
  --reasoning-parser "$REASONING_PARSER" \
  --default-chat-template-kwargs "$CHAT_TEMPLATE_KWARGS" \
  "${AUTO_TOOL_CHOICE_ARGS[@]}" \
  --tool-call-parser "$TOOL_CALL_PARSER" \
  --quantization "$QUANTIZATION" \
  "${SPECULATIVE_ARGS[@]}" \
  "${TRUST_ARGS[@]}" \
  "${PROMPT_TOKENS_ARGS[@]}" \
  "${REQUEST_METRICS_ARGS[@]}" \
  "${DISABLE_LOG_STATS_ARGS[@]}" \
  --uvicorn-log-level warning \
  --port "$PORT"
