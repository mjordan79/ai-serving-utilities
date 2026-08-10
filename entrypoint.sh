#!/bin/bash
set -e

API_KEY_FILE="/root/.vllm-key/.api_key"

if [ -n "$HF_TOKEN" ]; then
  export HF_TOKEN
fi

# API key: abilita con ENABLE_API_KEY=true.
# Se abilitata e non esiste, genera un UUID persistente.
# Se abilitata e esiste, riutilizza il token salvato.
# Se disabilitata, nessuna autenticazione (dev locale).
API_KEY_ARGS=()
if [ "${ENABLE_API_KEY:-false}" = "true" ]; then
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

SPECULATIVE_ARGS=()
if [ "${ENABLE_MTP:-true}" = "true" ]; then
  SPECULATIVE_ARGS=(--speculative-config '{"method":"mtp","num_speculative_tokens":2}')
fi

REQUEST_METRICS_ARGS=()
if [ "${ENABLE_REQUEST_METRICS:-false}" = "true" ]; then
  REQUEST_METRICS_ARGS=(--enable-per-request-metrics)
fi

exec vllm serve "$MODEL_NAME" \
  "${API_KEY_ARGS[@]}" \
  --dtype auto \
  --safetensors-load-strategy=prefetch \
  --tensor-parallel-size "${TP_SIZE:-1}" \
  --attention-backend triton_attn \
  --performance-mode interactivity \
  --language-model-only \
  --skip-mm-profiling \
  --kv-cache-dtype fp8_e4m3 \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.94}" \
  --max-model-len "${MAX_MODEL_LEN:-180800}" \
  --max-num-seqs "${MAX_NUM_SEQS:-1}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-6144}" \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --no-disable-hybrid-kv-cache-manager \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking": true}' \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --quantization modelopt \
  "${SPECULATIVE_ARGS[@]}" \
  --trust-remote-code \
  --enable-prompt-tokens-details \
  "${REQUEST_METRICS_ARGS[@]}" \
  --port 8000
