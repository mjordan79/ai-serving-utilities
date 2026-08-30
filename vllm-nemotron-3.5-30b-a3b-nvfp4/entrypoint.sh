#!/usr/bin/env bash
set -e

API_KEY_FILE="/root/.vllm-key/.api_key"

# Runtime defaults live here so Compose only passes explicit overrides.
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS="${VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS:-1}"
# FlashInfer autotune cache — backed by the vllm-flashinfer-cache volume in
# docker-compose.yml. vLLM's default (/tmp) is wiped on container recreation,
# so every boot re-autotunes ~42 kernel configs (+15-30 s at startup).
export VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR="${VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR:-/vllm-cache/flashinfer_autotune}"
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
# Defaults follow the vLLM recipe for nvidia/NVIDIA-Nemotron-3.5-Lightning-
# 30B-A3B-NVFP4, NVFP4 variant: base args only. The Hopper-only overrides
# (humming backends, horizontal SSU, async scheduling, 256 seqs / 32K batched)
# stay OFF by default — the RTX 5090 (sm_120) is not in the recipe's validated
# hardware matrix. Every override is still exposed as a variable.

# Security: cap the `n` parameter on /v1 completions. vLLM defaults
# VLLM_MAX_N_SEQUENCES to 16384 — a single request could ask for 16384
# parallel sequences. Docs suggest 64/128 for public deployments; 16 is
# plenty here (MAX_NUM_SEQS=8 already bounds concurrent sequences).
VLLM_MAX_N_SEQUENCES="${VLLM_MAX_N_SEQUENCES:-16}"
export VLLM_MAX_N_SEQUENCES

# Model
MODEL_NAME="${MODEL_NAME:-nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4}"
DTYPE="${DTYPE:-auto}"
TP_SIZE="${TP_SIZE:-1}"

# Memory & Context
# 0.94: the RTX 5090 is dedicated to this stack — raising utilization buys
# materially more KV headroom.
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.94}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8_e4m3}"

# Throughput — conservative first-boot values.
# Recipe base: 16384 batched tokens; Hopper escalation: 32768.
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"

# Attention & Performance
# Empty by default — never force an attention backend on a hybrid
# Mamba/attention model: flashinfer is rejected for some hybrid configs
# (validated Gemma post-mortem). Set only if a specific backend is required.
ATTENTION_BACKEND="${ATTENTION_BACKEND:-}"
PERFORMANCE_MODE="${PERFORMANCE_MODE:-interactivity}"

# MoE / Mamba backends — recipe base args.
# MOE_BACKEND: Marlin FP4 MoE kernels (recipe base on non-Hopper hardware).
# LINEAR_BACKEND / MAMBA_SSU_ALGORITHM / ASYNC_SCHEDULING are Hopper-only
# overrides (humming / horizontal / async) — empty by default.
MOE_BACKEND="${MOE_BACKEND:-marlin}"
LINEAR_BACKEND="${LINEAR_BACKEND:-}"
ASYNC_SCHEDULING="${ASYNC_SCHEDULING:-}"
MAMBA_BACKEND="${MAMBA_BACKEND:-flashinfer}"
MAMBA_CACHE_MODE="${MAMBA_CACHE_MODE:-align}"
# Fast SSM Cache — recipe feature: float16 SSM state + stochastic rounding.
MAMBA_SSM_CACHE_DTYPE="${MAMBA_SSM_CACHE_DTYPE:-float16}"
ENABLE_MAMBA_CACHE_STOCHASTIC_ROUNDING="${ENABLE_MAMBA_CACHE_STOCHASTIC_ROUNDING:-true}"
MAMBA_CACHE_PHILOX_ROUNDS="${MAMBA_CACHE_PHILOX_ROUNDS:-5}"

# Features
ENABLE_MTP="${ENABLE_MTP:-true}"
# Recipe Blackwell value: 3 speculative tokens with the built-in MTP layers.
MTP_NUM_SPECULATIVE_TOKENS="${MTP_NUM_SPECULATIVE_TOKENS:-3}"
# MoE backend inside the MTP speculative config (recipe: triton).
# Empty string omits the key from the JSON entirely.
MTP_MOE_BACKEND="${MTP_MOE_BACKEND:-triton}"
ENABLE_CHUNKED_PREFILL="${ENABLE_CHUNKED_PREFILL:-true}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-true}"
ENABLE_HYBRID_KV_CACHE_MANAGER="${ENABLE_HYBRID_KV_CACHE_MANAGER:-true}"
ENABLE_AUTO_TOOL_CHOICE="${ENABLE_AUTO_TOOL_CHOICE:-true}"
ENABLE_PROMPT_TOKENS_DETAILS="${ENABLE_PROMPT_TOKENS_DETAILS:-true}"

# Parsers (per the NVIDIA model card: qwen3_coder appears in every official
# vLLM / TRT-LLM / SGLang snippet for this checkpoint — it supersedes the
# vLLM recipe's qwen3_xml).
REASONING_PARSER="${REASONING_PARSER:-nemotron_v3}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_coder}"

# Loading
SAFETENSORS_LOAD_STRATEGY="${SAFETENSORS_LOAD_STRATEGY:-prefetch}"
SKIP_MM_PROFILING="${SKIP_MM_PROFILING:-true}"
# ModelOpt NVFP4 (W4A16). Empty string omits the flag — vLLM auto-detects
# from the checkpoint's quantization_config if the flag is ever rejected.
QUANTIZATION="${QUANTIZATION:-modelopt_fp4}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-true}"

# API
PORT="${PORT:-8000}"

# --- Conditional arguments ---
SPECULATIVE_ARGS=()
if [ "$ENABLE_MTP" = "true" ]; then
  MTP_CONFIG="{\"method\":\"mtp\",\"num_speculative_tokens\":$MTP_NUM_SPECULATIVE_TOKENS"
  if [ -n "$MTP_MOE_BACKEND" ]; then
    MTP_CONFIG="$MTP_CONFIG,\"moe_backend\":\"$MTP_MOE_BACKEND\""
  fi
  MTP_CONFIG="$MTP_CONFIG}"
  SPECULATIVE_ARGS=(--speculative-config "$MTP_CONFIG")
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

ATTENTION_BACKEND_ARGS=()
if [ -n "$ATTENTION_BACKEND" ]; then
  ATTENTION_BACKEND_ARGS=(--attention-backend "$ATTENTION_BACKEND")
fi

MOE_BACKEND_ARGS=()
if [ -n "$MOE_BACKEND" ]; then
  MOE_BACKEND_ARGS=(--moe-backend "$MOE_BACKEND")
fi

LINEAR_BACKEND_ARGS=()
if [ -n "$LINEAR_BACKEND" ]; then
  LINEAR_BACKEND_ARGS=(--linear-backend "$LINEAR_BACKEND")
fi

MAMBA_ARGS=()
if [ -n "$MAMBA_BACKEND" ]; then
  MAMBA_ARGS+=(--mamba-backend "$MAMBA_BACKEND")
fi
if [ -n "$MAMBA_CACHE_MODE" ]; then
  MAMBA_ARGS+=(--mamba-cache-mode "$MAMBA_CACHE_MODE")
fi
if [ -n "$MAMBA_SSU_ALGORITHM" ]; then
  MAMBA_ARGS+=(--mamba-ssu-algorithm "$MAMBA_SSU_ALGORITHM")
fi
if [ -n "$MAMBA_SSM_CACHE_DTYPE" ]; then
  MAMBA_ARGS+=(--mamba-ssm-cache-dtype "$MAMBA_SSM_CACHE_DTYPE")
fi
if [ "$ENABLE_MAMBA_CACHE_STOCHASTIC_ROUNDING" = "true" ]; then
  MAMBA_ARGS+=(--enable-mamba-cache-stochastic-rounding)
fi
if [ -n "$MAMBA_CACHE_PHILOX_ROUNDS" ]; then
  MAMBA_ARGS+=(--mamba-cache-philox-rounds "$MAMBA_CACHE_PHILOX_ROUNDS")
fi

ASYNC_SCHEDULING_ARGS=()
if [ -n "$ASYNC_SCHEDULING" ]; then
  ASYNC_SCHEDULING_ARGS=(--async-scheduling)
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

QUANTIZATION_ARGS=()
if [ -n "$QUANTIZATION" ]; then
  QUANTIZATION_ARGS=(--quantization "$QUANTIZATION")
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
  "${ATTENTION_BACKEND_ARGS[@]}" \
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
  "${MOE_BACKEND_ARGS[@]}" \
  "${LINEAR_BACKEND_ARGS[@]}" \
  "${MAMBA_ARGS[@]}" \
  "${ASYNC_SCHEDULING_ARGS[@]}" \
  --reasoning-parser "$REASONING_PARSER" \
  "${AUTO_TOOL_CHOICE_ARGS[@]}" \
  --tool-call-parser "$TOOL_CALL_PARSER" \
  "${QUANTIZATION_ARGS[@]}" \
  "${SPECULATIVE_ARGS[@]}" \
  "${TRUST_ARGS[@]}" \
  "${PROMPT_TOKENS_ARGS[@]}" \
  "${REQUEST_METRICS_ARGS[@]}" \
  "${DISABLE_LOG_STATS_ARGS[@]}" \
  --uvicorn-log-level warning \
  --port "$PORT" \
  "${EXTRA_ARGS[@]}"
