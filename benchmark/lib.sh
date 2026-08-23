#!/usr/bin/env bash
# lib.sh — Shared library for all benchmark scripts.
# Sourced by individual tests, run.sh and warmup.sh. Handles curl, timing,
# metrics, and shared configuration discovery.
# Dependencies: bash, curl, gawk, date. No jq required.

set -euo pipefail

# ── Shared configuration discovery (used by run.sh and warmup.sh) ──────────
# Callers invoke resolve_model_target <model> first; it defines PROJECT_DIR
# (plus DIRECT_PORT / CONTAINER_NAME) for the selected model deployment.

# Model selector → deployment directory (relative to the workspace root, the
# parent of this suite). Single point of maintenance: renaming a model
# directory only requires updating this table.
MODEL_QWEN_DIR="vllm-qwen-3.8-27b-nvfp4"
MODEL_MUSE_DIR="vllm-muse-glimmer-30b-nvfp4"

# Resolve the target model deployment.
#   $1 = model selector (qwen|muse), default qwen.
# Exports:
#   PROJECT_DIR    — absolute path to the model deployment directory
#   DIRECT_PORT    — host port published by the model's docker-compose.yml
#   CONTAINER_NAME — container_name from the model's docker-compose.yml
resolve_model_target() {
    local selector="${1:-qwen}"
    local model_dir
    case "$selector" in
        qwen) model_dir="$MODEL_QWEN_DIR" ;;
        muse) model_dir="$MODEL_MUSE_DIR" ;;
        *) echo "ERROR: Unknown model '${selector}'. Valid models: qwen, muse." >&2; return 1 ;;
    esac
    local suite_dir
    suite_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_DIR="$(cd "${suite_dir}/.." && pwd)/${model_dir}"
    [[ -d "$PROJECT_DIR" ]] || { echo "ERROR: Model directory not found: ${PROJECT_DIR}" >&2; return 1; }
    local compose="${PROJECT_DIR}/docker-compose.yml"
    [[ -f "$compose" ]] || { echo "ERROR: Compose file not found: ${compose}" >&2; return 1; }
    # First published port mapping "<host>:8000" (vLLM listens on 8000 inside).
    DIRECT_PORT=$(grep -oE '"[0-9]+:8000"' "$compose" | head -1 | tr -d '"' | cut -d: -f1 || true)
    [[ -n "$DIRECT_PORT" ]] || { echo "ERROR: No published port <host>:8000 found in ${compose}" >&2; return 1; }
    CONTAINER_NAME=$(grep -E '^[[:space:]]*container_name:' "$compose" | head -1 \
        | sed 's/.*container_name:[[:space:]]*//' | tr -d '"\r' || true)
    [[ -n "$CONTAINER_NAME" ]] || { echo "ERROR: No container_name found in ${compose}" >&2; return 1; }
    export PROJECT_DIR DIRECT_PORT CONTAINER_NAME
    echo "INFO: Target model '${selector}' → ${PROJECT_DIR} (port ${DIRECT_PORT}, container ${CONTAINER_NAME})" >&2
}

# Trim leading/trailing whitespace (pure bash — no xargs, no subshell).
trim() {
    local s="$1"
    s="${s#"${s%%[![:space:]]*}"}"
    s="${s%"${s##*[![:space:]]}"}"
    printf '%s' "$s"
}

# Parse KEY=VALUE pairs from the project's .env and export them.
# Skips comments and empty lines; strips surrounding double quotes from values.
parse_project_env() {
    local env_file="$1" key value
    [[ -f "$env_file" ]] || { echo "ERROR: Project .env not found at ${env_file}" >&2; return 1; }
    while IFS='=' read -r key value; do
        key=$(trim "$key")
        [[ "$key" == \#* ]] && continue
        [[ -z "$key" ]] && continue
        value=$(trim "$value")
        value="${value%$'\r'}"
        value="${value#\"}"
        value="${value%\"}"
        export "$key=$value"
    done < "$env_file"
}

# Resolve BASE_URL: external domain if reachable, otherwise the direct
# HTTP endpoint published by docker-compose.yml (localhost:${DIRECT_PORT}).
resolve_base_url() {
    if [[ -n "${LETSENCRYPT_DOMAIN:-}" ]]; then
        if curl -sS -k --max-time 5 --output /dev/null "https://${LETSENCRYPT_DOMAIN}/health" 2>/dev/null; then
            BASE_URL="https://${LETSENCRYPT_DOMAIN}"
        else
            echo "INFO: ${LETSENCRYPT_DOMAIN} not reachable, falling back to http://localhost:${DIRECT_PORT}"
            BASE_URL="http://localhost:${DIRECT_PORT}"
        fi
    else
        BASE_URL="http://localhost:${DIRECT_PORT}"
    fi
    export BASE_URL
}

# Recover the vLLM API key. Resolution order:
#   1. Live from the running container (docker exec → docker compose exec).
#   2. VLLM_API_KEY from the project .env (fixed-key convention).
#   3. No-auth probe: if the server answers /v1/models without credentials,
#      authentication is disabled and an empty key is a valid state.
#   4. Interactive prompt as last resort.
# NOTE: docker output is captured via a temp file, not command substitution —
# under WSL interop (docker.exe), substitution/pipe capture can come back empty
# while file redirect works reliably.
recover_api_key() {
    API_KEY=""
    local probe_code=""
    if command -v docker >/dev/null 2>&1; then
        local keyfile
        keyfile=$(mktemp 2>/dev/null || printf '/tmp/vllm_key_%s' "$$")
        docker exec "${CONTAINER_NAME}" cat /root/.vllm-key/.api_key > "$keyfile" 2>/dev/null \
            && API_KEY=$(<"$keyfile")
        if [[ -z "$API_KEY" ]]; then
            # Fallback: docker compose (service name via compose file)
            docker compose -f "${PROJECT_DIR}/docker-compose.yml" \
                exec -T vllm cat /root/.vllm-key/.api_key > "$keyfile" 2>/dev/null \
                && API_KEY=$(<"$keyfile")
        fi
        rm -f "$keyfile"
    fi
    if [[ -z "$API_KEY" ]] && [[ -n "${VLLM_API_KEY:-}" ]]; then
        API_KEY="$VLLM_API_KEY"
        echo "INFO: Using VLLM_API_KEY from .env (docker recovery unavailable)." >&2
    fi
    if [[ -z "$API_KEY" ]]; then
        # No-auth probe: a server running with ENABLE_API_KEY=false ignores the
        # Authorization header and answers /v1/models without credentials (200).
        # An authenticated server answers 401. Only treat 200 as "auth off".
        probe_code=$(curl -sS -k --max-time 5 -o /dev/null -w '%{http_code}' \
            "${BASE_URL%/}/v1/models" 2>/dev/null || echo 000)
        if [[ "$probe_code" == "200" ]]; then
            echo "INFO: API key disabled on server (no-auth probe OK) — continuing without a key." >&2
        else
            echo "WARNING: Could not auto-detect API_KEY." >&2
            echo "         Docker may not be running or the container is not available." >&2
            echo "         Set VLLM_API_KEY in the project .env to skip this prompt." >&2
            echo -n "         Enter VLLM_API_KEY: " >&2
            read -r API_KEY || API_KEY=""
        fi
    fi
    if [[ -z "$API_KEY" && "$probe_code" != "200" ]]; then
        echo "ERROR: API_KEY is required." >&2
        return 1
    fi
    export API_KEY
}

# ── JSON helpers (replaces jq) ──────────────────────────────────────────────

# Escape a string for JSON embedding
json_escape() {
    local s="$1"
    s="${s//\\/\\\\}"
    s="${s//\"/\\\"}"
    s="${s//$'\n'/\\n}"
    s="${s//$'\t'/\\t}"
    printf '%s' "$s"
}

# Extract a simple JSON value by key from a JSON string (flat or 1-level nested)
# Usage: json_get_value "$json" "prompt_tokens"
json_get_value() {
    local json="$1"
    local key="$2"
    echo "$json" | gawk -v key="\"$key\"" '
    {
        # Find "key": value pattern
        n = split($0, parts, key)
        if (n >= 2) {
            rest = parts[2]
            # Skip leading : and whitespace
            gsub(/^[[:space:]]*:[[:space:]]*/, "", rest)
            # Extract value (number or quoted string)
            if (match(rest, /^"[^"]*"/)) {
                val = substr(rest, RSTART+1, RLENGTH-2)
                print val
            } else if (match(rest, /^[0-9.eE+-]+/)) {
                val = substr(rest, RSTART, RLENGTH)
                print val
            }
        }
    }'
}

# Extract delta.content OR delta.reasoning from SSE chunk JSON
# vLLM with reasoning models puts content in "reasoning" field during thinking
json_get_delta_content() {
    local json="$1"
    # Try "content" first, then "reasoning"
    local result
    result=$(echo "$json" | gawk '
    /"content"/ {
        if (match($0, /"content"[[:space:]]*:[[:space:]]*"/)) {
            rest = substr($0, RSTART + RLENGTH)
            content = ""
            i = 1
            while (i <= length(rest)) {
                c = substr(rest, i, 1)
                if (c == "\\") {
                    content = content substr(rest, i, 2)
                    i += 2
                    continue
                }
                if (c == "\"") break
                content = content c
                i++
            }
            if (content != "") {
                gsub(/\\n/, "\n", content)
                gsub(/\\t/, "\t", content)
                print content
                exit
            }
        }
    }
    /"reasoning"/ {
        if (match($0, /"reasoning"[[:space:]]*:[[:space:]]*"/)) {
            rest = substr($0, RSTART + RLENGTH)
            content = ""
            i = 1
            while (i <= length(rest)) {
                c = substr(rest, i, 1)
                if (c == "\\") {
                    content = content substr(rest, i, 2)
                    i += 2
                    continue
                }
                if (c == "\"") break
                content = content c
                i++
            }
            if (content != "") {
                gsub(/\\n/, "\n", content)
                gsub(/\\t/, "\t", content)
                print content
                exit
            }
        }
    }')
    printf '%s' "$result"
}

# ── Helpers ──────────────────────────────────────────────────────────────────

# Nanosecond timestamp
now_ns() {
    date +%s%N
}

# Millisecond timestamp from nanoseconds
ns_to_ms() {
    echo $(( $1 / 1000000 ))
}

# ── Curl wrapper: chat completion (non-streaming for accurate metrics) ──────
# vLLM v0.27.1 no longer sends usage in SSE chunks. Use stream=false for
# authoritative token counts and metrics.time_to_first_token_ms for TTFT.
#
# Returns metrics via global variables:
#   METRIC_TTFT_MS        — Time to first token (from vLLM metrics)
#   METRIC_TOTAL_MS       — Total request duration (wall clock)
#   METRIC_OUTPUT_TOKENS  — Completion tokens (from usage)
#   METRIC_INPUT_TOKENS   — Prompt tokens (from usage)
#   METRIC_PROMPT_TOKENS  — Prompt tokens (from usage)
#   METRIC_GEN_TOKENS     — Completion tokens (from usage)
#   METRIC_TPS            — Tokens per second
#   METRIC_RAW_JSON       — Raw JSON response

declare -g METRIC_TTFT_MS=0
declare -g METRIC_TOTAL_MS=0
declare -g METRIC_OUTPUT_TOKENS=0
declare -g METRIC_INPUT_TOKENS=0
declare -g METRIC_PROMPT_TOKENS=0
declare -g METRIC_GEN_TOKENS=0
declare -g METRIC_TPS="0.0"
declare -g METRIC_RAW_JSON=""

run_chat_stream() {
    local payload="$1"

    METRIC_TTFT_MS=0
    METRIC_TOTAL_MS=0
    METRIC_OUTPUT_TOKENS=0
    METRIC_INPUT_TOKENS=0
    METRIC_PROMPT_TOKENS=0
    METRIC_GEN_TOKENS=0
    METRIC_TPS="0.0"
    METRIC_RAW_JSON=""

    # Force non-streaming mode for accurate token counts
    # vLLM v0.27.1 does not send usage in SSE chunks
    local nostream_payload
    nostream_payload=$(printf '%s' "$payload" | sed 's/"stream":true/"stream":false/')

    local start_ns
    start_ns=$(now_ns)
    local tmpfile codefile
    tmpfile=$(mktemp)
    codefile=$(mktemp)

    local rc=0
    curl -sS -k --max-time "${CURL_TIMEOUT:-600}" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer ${API_KEY}" \
        -d "$nostream_payload" \
        -o "$tmpfile" \
        -w '%{http_code}' > "$codefile" \
        "${BASE_URL}/v1/chat/completions" 2>/dev/null || rc=$?
    local status
    status=$(cat "$codefile" 2>/dev/null || echo "")
    rm -f "$codefile"
    if [[ $rc -ne 0 || "${status:-000}" != "200" ]]; then
        local error_msg
        error_msg=$(head -c 200 "$tmpfile" 2>/dev/null || true)
        rm -f "$tmpfile"
        echo "ERROR: HTTP ${status:-curl-failed} (rc=$rc): ${error_msg}" >&2
        return 1
    fi

    local end_ns
    end_ns=$(now_ns)

    local response
    response=$(cat "$tmpfile")
    rm -f "$tmpfile"

    METRIC_RAW_JSON="$response"

    # Extract token counts from usage
    local usage_prompt usage_completion
    usage_prompt=$(json_get_value "$response" "prompt_tokens")
    usage_completion=$(json_get_value "$response" "completion_tokens")

    if [[ -n "$usage_prompt" ]]; then
        METRIC_PROMPT_TOKENS=$usage_prompt
        METRIC_INPUT_TOKENS=$usage_prompt
    fi
    if [[ -n "$usage_completion" ]]; then
        METRIC_GEN_TOKENS=$usage_completion
        METRIC_OUTPUT_TOKENS=$usage_completion
    fi

    # Extract TTFT from vLLM metrics (more accurate than wall clock)
    local ttft_from_metrics
    ttft_from_metrics=$(json_get_value "$response" "time_to_first_token_ms")
    if [[ -n "$ttft_from_metrics" ]]; then
        # vLLM returns float, truncate to integer
        METRIC_TTFT_MS=$(printf '%.0f' "$ttft_from_metrics" 2>/dev/null || echo "0")
    fi

    # Wall clock total time
    METRIC_TOTAL_MS=$(( (end_ns - start_ns) / 1000000 ))

    # Fallback: if no metrics TTFT, use wall clock
    if [[ $METRIC_TTFT_MS -eq 0 ]]; then
        METRIC_TTFT_MS=$METRIC_TOTAL_MS
    fi

    # TPS calculation
    if [[ $METRIC_TOTAL_MS -gt 0 ]]; then
        METRIC_TPS=$(awk "BEGIN { printf \"%.2f\", ${METRIC_OUTPUT_TOKENS} / (${METRIC_TOTAL_MS} / 1000.0) }")
    fi
}

# ── Report helpers ───────────────────────────────────────────────────────────

# Print a single test result line (TSV format for easy parsing)
print_result_line() {
    local test_name="$1"
    local iteration="$2"
    echo -e "${test_name}\t${iteration}\t${METRIC_INPUT_TOKENS}\t${METRIC_OUTPUT_TOKENS}\t${METRIC_TTFT_MS}\t${METRIC_TOTAL_MS}\t${METRIC_TPS}"
}

# Print a formatted summary table header
print_table_header() {
    printf "%-40s %6s %8s %8s %10s %12s %10s\n" \
        "Test" "Iter" "InTok" "OutTok" "TTFT(ms)" "Total(ms)" "TPS"
    printf "%-40s %6s %8s %8s %10s %12s %10s\n" \
        "----------------------------------------" "------" "--------" "--------" "----------" "------------" "----------"
}

# Print a formatted summary table row
print_table_row() {
    local test_name="$1"
    local iteration="$2"
    printf "%-40s %6s %8s %8s %10s %12s %10s\n" \
        "$test_name" "$iteration" "$METRIC_INPUT_TOKENS" "$METRIC_OUTPUT_TOKENS" \
        "$METRIC_TTFT_MS" "$METRIC_TOTAL_MS" "$METRIC_TPS"
}

# Write a markdown summary to a file
write_markdown_summary() {
    local output_file="$1"
    local model_label="$2"
    local timestamp
    timestamp=$(date '+%Y-%m-%d %H:%M:%S %Z')

    cat > "$output_file" <<EOF
# Benchmark Report: ${model_label}

- **Date:** ${timestamp}
- **Target:** ${BASE_URL}
- **Model:** ${model_label}

## Results

| Test | Iter | Input Tokens | Output Tokens | TTFT (ms) | Total (ms) | TPS |
|------|------|-------------|---------------|-----------|------------|-----|
EOF

    # Append data rows (expects TSV on stdin)
    while IFS=$'\t' read -r test_name iter in_tok out_tok ttft total tps; do
        echo "| ${test_name} | ${iter} | ${in_tok} | ${out_tok} | ${ttft} | ${total} | ${tps} |" >> "$output_file"
    done

    cat >> "$output_file" <<'EOF'

## Notes

- TTFT = Time To First Token (ms)
- TPS = Tokens Per Second (output tokens / total duration)
- Total = End-to-end request duration (ms)
- Each test runs multiple iterations; results are averaged in the comparison report.
EOF
}


