#!/usr/bin/env bash
set -e

API_KEY_FILE="/root/.vllm-key/.api_key"

# Runtime defaults live here so Compose only passes explicit overrides.
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS="${VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS:-1}"
export VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR="${VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR:-/tmp/flashinfer_autotune_cache}"
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

# API key: ENABLE_API_KEY=true (default) → auth required.
# If enabled and it doesn't exist, generate a persistent UUID.
# If enabled and it exists, reuse the saved token.
# If disabled, no authentication (dev local).
API_KEY_ARGS=()
if [ "$ENABLE_API_KEY" = "true" ]; then
  if [ -z "$VLLM_API_KEY" ]; then
    if [ -f "$API_KEY_FILE" ]; then
      VLLM_API_KEY=$(cat "$API_KEY_FILE")
      echo "[API] Reusing existing API key."
    else
      VLLM_API_KEY="sk-$(cat /proc/sys/kernel/random/uuid 2>/dev/null || python3 -c 'import uuid; print(uuid.uuid4())')"
      mkdir -p "$(dirname "$API_KEY_FILE")"
      echo "$VLLM_API_KEY" > "$API_KEY_FILE"
      echo "[API] Generated API key: $VLLM_API_KEY"
      echo "[API] Save this value — it will not be shown again."
    fi
    export VLLM_API_KEY
  fi
  API_KEY_ARGS=(--api-key "$VLLM_API_KEY")
fi

# --- Build vLLM arguments from environment variables ---
# Every flag has a default here; override through the container environment to change.

# Model
MODEL_NAME="${MODEL_NAME:-RedHatAI/Muse-Glimmer-30B-NVFP4}"
DTYPE="${DTYPE:-auto}"
TP_SIZE="${TP_SIZE:-1}"

# Memory & Context
# Defaults follow the numbers validated by the vLLM recipe on 1x RTX 5090:
# 28.8/32.6 GB used, ~179K-token KV pool at the full 128K context.
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"

# Throughput
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-6144}"

# Attention & Performance
ATTENTION_BACKEND="${ATTENTION_BACKEND:-flashinfer}"
PERFORMANCE_MODE="${PERFORMANCE_MODE:-interactivity}"

# Features
ENABLE_CHUNKED_PREFILL="${ENABLE_CHUNKED_PREFILL:-true}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-true}"
ENABLE_HYBRID_KV_CACHE_MANAGER="${ENABLE_HYBRID_KV_CACHE_MANAGER:-true}"
ENABLE_AUTO_TOOL_CHOICE="${ENABLE_AUTO_TOOL_CHOICE:-true}"
ENABLE_PROMPT_TOKENS_DETAILS="${ENABLE_PROMPT_TOKENS_DETAILS:-true}"

# Parsers — muse_glimmer (native since vLLM 0.27.0). They decode the model's
# channel-scoped output format (`to=self` / `to=user` / ATEM XML tool calls).
# The reasoning parser forces skip_special_tokens=False — keep both enabled
# together with --enable-auto-tool-choice.
REASONING_PARSER="${REASONING_PARSER:-muse_glimmer}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-muse_glimmer}"

# Sampling: load the checkpoint's published generation config (temp 1.0 /
# top_p 0.95 / top_k 64). The model is tuned for it — never run it greedy.
GENERATION_CONFIG="${GENERATION_CONFIG:-auto}"

# Loading
SAFETENSORS_LOAD_STRATEGY="${SAFETENSORS_LOAD_STRATEGY:-prefetch}"
QUANTIZATION="${QUANTIZATION:-compressed-tensors}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-true}"

# Speculative decoding — DFlash draft head (per the recipe). OFF by default:
# the 5.1 GB draft head OOMs on 1x RTX 5090 (~400 MiB headroom). Enable only
# on a 2x setup with TP_SIZE=2, where the recipe measured ~3.5x decode.
ENABLE_SPEC_DECODING="${ENABLE_SPEC_DECODING:-false}"
SPEC_METHOD="${SPEC_METHOD:-dflash}"
SPEC_MODEL="${SPEC_MODEL:-meta-models/Muse-Glimmer-30B-assistant}"
SPEC_NUM_TOKENS="${SPEC_NUM_TOKENS:-15}"

# API
PORT="${PORT:-8000}"

# --- Conditional arguments ---
SPECULATIVE_ARGS=()
if [ "$ENABLE_SPEC_DECODING" = "true" ]; then
  SPECULATIVE_ARGS=(--speculative-config "{\"method\":\"$SPEC_METHOD\",\"model\":\"$SPEC_MODEL\",\"num_speculative_tokens\":$SPEC_NUM_TOKENS}")
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

TRUST_ARGS=()
if [ "$TRUST_REMOTE_CODE" = "true" ]; then
  TRUST_ARGS=(--trust-remote-code)
fi

# Escape hatch: raw extra flags appended verbatim to `vllm serve`.
EXTRA_ARGS=()
if [ -n "$EXTRA_ARGS_STR" ]; then
  read -r -a EXTRA_ARGS <<< "$EXTRA_ARGS_STR"
fi

exec vllm serve "$MODEL_NAME" \
  "${API_KEY_ARGS[@]}" \
  --dtype "$DTYPE" \
  --safetensors-load-strategy "$SAFETENSORS_LOAD_STRATEGY" \
  --tensor-parallel-size "$TP_SIZE" \
  --attention-backend "$ATTENTION_BACKEND" \
  --performance-mode "$PERFORMANCE_MODE" \
  --kv-cache-dtype "$KV_CACHE_DTYPE" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  "${CHUNKED_PREFILL_ARGS[@]}" \
  "${PREFIX_CACHING_ARGS[@]}" \
  "${HYBRID_KV_ARGS[@]}" \
  --reasoning-parser "$REASONING_PARSER" \
  "${AUTO_TOOL_CHOICE_ARGS[@]}" \
  --tool-call-parser "$TOOL_CALL_PARSER" \
  --generation-config "$GENERATION_CONFIG" \
  --quantization "$QUANTIZATION" \
  "${SPECULATIVE_ARGS[@]}" \
  "${TRUST_ARGS[@]}" \
  "${PROMPT_TOKENS_ARGS[@]}" \
  "${REQUEST_METRICS_ARGS[@]}" \
  "${DISABLE_LOG_STATS_ARGS[@]}" \
  --uvicorn-log-level warning \
  --port "$PORT" \
  "${EXTRA_ARGS[@]}"
