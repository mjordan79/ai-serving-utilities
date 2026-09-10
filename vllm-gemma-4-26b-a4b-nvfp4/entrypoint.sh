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
MODEL_NAME="${MODEL_NAME:-nvidia/Gemma-4-26B-A4B-NVFP4}"
# Served name exposed via /v1/models. [engine]/[org]/[model-HF].
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-vllm/nvidia/gemma-4-26b-a4b-nvfp4}"
DTYPE="${DTYPE:-auto}"
TP_SIZE="${TP_SIZE:-1}"

# Memory & Context
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.88}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
# FP8 KV is not validated for Gemma 4 on the target platform; leave to the engine.
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"

# Throughput
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-6144}"

# Admission control (server-side backstop complementing the nginx rate
# limiting): hard cap on in-flight requests (waiting + running) and on
# total queued prefill tokens; overflow -> HTTP 503 from the engine.
# MAX_NUM_QUEUED_TOKENS accepts human-readable integers (4K, 64K, 128K, 256K).
# REQS 4 is 4x MAX_NUM_SEQS; 128K spans the full single-prompt context
# (MAX_MODEL_LEN) so a long prompt is admitted, and still sits far above the
# heaviest benchmark prefill (~4K) so normal use never trips the backstop.
MAX_NUM_QUEUED_REQS="${MAX_NUM_QUEUED_REQS:-4}"
MAX_NUM_QUEUED_TOKENS="${MAX_NUM_QUEUED_TOKENS:-128K}"

# Attention & Performance
# Gemma 4 = hybrid attention (SWA-128 + global, per-layer head dims) + multimodal.
# The mm-prefix path needs a backend that supports_mm_prefix(); FlashInfer does
# not. Leave empty for vLLM auto-select; set a value only with a validated backend.
ATTENTION_BACKEND="${ATTENTION_BACKEND:-}"
PERFORMANCE_MODE="${PERFORMANCE_MODE:-interactivity}"

# NVIDIA ModelOpt Gemma 4 NVFP4: vLLM CUTLASS is the preferred Blackwell
# MoE backend in the model card; set empty to let vLLM auto-select.
MOE_BACKEND="${MOE_BACKEND:-cutlass}"

# Features
ENABLE_MTP="${ENABLE_MTP:-false}"
# MTP is kept disabled: the Gemma 4 checkpoint has no MTP layers — the model
# config declares none, so there is no draft layer to speculatively sample
# from on this model.
MTP_NUM_SPECULATIVE_TOKENS="${MTP_NUM_SPECULATIVE_TOKENS:-1}"
# Per-request speculative-decoding acceptance metrics in the response body
# (metrics.speculative_decoding): none | summary | detailed. vLLM refuses to
# start if set to a non-none value while speculative decoding is disabled;
# reported only for single-sequence requests (n=1); independent of
# --disable-log-stats.
PER_REQUEST_SPEC_DECODE_METRICS="${PER_REQUEST_SPEC_DECODE_METRICS:-none}"
ENABLE_CHUNKED_PREFILL="${ENABLE_CHUNKED_PREFILL:-true}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-true}"
ENABLE_HYBRID_KV_CACHE_MANAGER="${ENABLE_HYBRID_KV_CACHE_MANAGER:-true}"
ENABLE_AUTO_TOOL_CHOICE="${ENABLE_AUTO_TOOL_CHOICE:-true}"
ENABLE_PROMPT_TOKENS_DETAILS="${ENABLE_PROMPT_TOKENS_DETAILS:-true}"

# Parsers (native vLLM parsers for this arch)
REASONING_PARSER="${REASONING_PARSER:-gemma4}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-gemma4}"
# Gemma 4 model card: thinking is not carried in multi-turn history.
DEFAULT_ENABLE_THINKING="${DEFAULT_ENABLE_THINKING:-false}"
DEFAULT_PRESERVE_THINKING="${DEFAULT_PRESERVE_THINKING:-false}"

# Loading. Gemma 4 is multimodal by default (language-model-only OFF).
SAFETENSORS_LOAD_STRATEGY="${SAFETENSORS_LOAD_STRATEGY:-prefetch}"
SKIP_MM_PROFILING="${SKIP_MM_PROFILING:-false}"
# NVIDIA ModelOpt NVFP4 (W4A16). The checkpoint also declares its quantization
# config; keep the explicit value overridable for compatibility testing.
QUANTIZATION="${QUANTIZATION:-modelopt_fp4}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-false}"

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

# Attention backend: passed ONLY when explicitly set (auto-select otherwise).
ATTENTION_ARGS=()
if [ -n "$ATTENTION_BACKEND" ]; then
  ATTENTION_ARGS=(--attention-backend "$ATTENTION_BACKEND")
fi

MOE_BACKEND_ARGS=()
if [ -n "$MOE_BACKEND" ]; then
  MOE_BACKEND_ARGS=(--moe-backend "$MOE_BACKEND")
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

SPEC_DECODE_METRICS_ARGS=()
if [ "$PER_REQUEST_SPEC_DECODE_METRICS" != "none" ]; then
  SPEC_DECODE_METRICS_ARGS=(--per-request-spec-decode-metrics "$PER_REQUEST_SPEC_DECODE_METRICS")
fi

ADMISSION_ARGS=()
if [ -n "$MAX_NUM_QUEUED_REQS" ]; then
  ADMISSION_ARGS+=(--max-num-queued-reqs "$MAX_NUM_QUEUED_REQS")
fi
if [ -n "$MAX_NUM_QUEUED_TOKENS" ]; then
  ADMISSION_ARGS+=(--max-num-queued-tokens "$MAX_NUM_QUEUED_TOKENS")
fi

SKIP_MM_ARGS=()
if [ "$SKIP_MM_PROFILING" = "true" ]; then
  SKIP_MM_ARGS=(--skip-mm-profiling)
fi

TRUST_ARGS=()
if [ "$TRUST_REMOTE_CODE" = "true" ]; then
  TRUST_ARGS=(--trust-remote-code)
fi

# Multimodal by default: add --language-model-only only when explicitly requested.
LANG_MODEL_ARGS=()
if [ "${LANGUAGE_MODEL_ONLY-}" = "true" ]; then
  LANG_MODEL_ARGS=(--language-model-only)
fi

CHAT_TEMPLATE_KWARGS='{"enable_thinking": '"$DEFAULT_ENABLE_THINKING"', "preserve_thinking": '"$DEFAULT_PRESERVE_THINKING"'}'

exec vllm serve "$MODEL_NAME" \
  --served-model-name "$SERVED_MODEL_NAME" \
  "${API_KEY_ARGS[@]}" \
  --dtype "$DTYPE" \
  --safetensors-load-strategy "$SAFETENSORS_LOAD_STRATEGY" \
  --tensor-parallel-size "$TP_SIZE" \
  "${ATTENTION_ARGS[@]}" \
  "${MOE_BACKEND_ARGS[@]}" \
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
  "${SPEC_DECODE_METRICS_ARGS[@]}" \
  "${ADMISSION_ARGS[@]}" \
  --uvicorn-log-level warning \
  --port "$PORT"
