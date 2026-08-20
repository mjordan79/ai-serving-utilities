#!/usr/bin/env bash
# lib.sh — Shared library for all benchmark scripts.
# Sourced by individual tests and run.sh. Handles curl, timing, metrics.
# Dependencies: bash, curl, gawk, date. No jq required.

set -euo pipefail

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
    local tmpfile
    tmpfile=$(mktemp)

    curl -sS -k --max-time "${CURL_TIMEOUT:-600}" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer ${API_KEY}" \
        -d "$nostream_payload" \
        "${BASE_URL}/v1/chat/completions" \
        > "$tmpfile" 2>/dev/null || {
            local rc=$?
            local error_msg
            error_msg=$(cat "$tmpfile" 2>/dev/null || echo "unknown error")
            rm -f "$tmpfile"
            echo "ERROR: curl failed with exit code $rc" >&2
            echo "ERROR: $error_msg" >&2
            return 1
        }

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

# ── Curl wrapper: non-streaming chat completion ─────────────────────────────
# For simpler single-response tests where streaming isn't needed.

run_chat_nostream() {
    local payload="$1"
    local tmpfile
    tmpfile=$(mktemp)

    local start_ns
    start_ns=$(now_ns)

    curl -sS -k --max-time "${CURL_TIMEOUT:-600}" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer ${API_KEY}" \
        -d "$payload" \
        "${BASE_URL}/v1/chat/completions" \
        > "$tmpfile" 2>/dev/null || {
            local rc=$?
            rm -f "$tmpfile"
            echo "ERROR: curl failed with exit code $rc" >&2
            return 1
        }

    local end_ns
    end_ns=$(now_ns)

    METRIC_TOTAL_MS=$(( (end_ns - start_ns) / 1000000 ))
    METRIC_TTFT_MS=$METRIC_TOTAL_MS  # Non-streaming: TTFT ≈ total
    METRIC_PROMPT_TOKENS=$(json_get_value "$(cat "$tmpfile")" "prompt_tokens")
    METRIC_GEN_TOKENS=$(json_get_value "$(cat "$tmpfile")" "completion_tokens")
    METRIC_PROMPT_TOKENS=${METRIC_PROMPT_TOKENS:-0}
    METRIC_GEN_TOKENS=${METRIC_GEN_TOKENS:-0}
    METRIC_INPUT_TOKENS=$METRIC_PROMPT_TOKENS
    METRIC_OUTPUT_TOKENS=$METRIC_GEN_TOKENS

    if [[ $METRIC_TOTAL_MS -gt 0 ]]; then
        METRIC_TPS=$(awk "BEGIN { printf \"%.2f\", ${METRIC_OUTPUT_TOKENS} / (${METRIC_TOTAL_MS} / 1000.0) }")
    fi

    rm -f "$tmpfile"
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

# Compute average and stddev for a column from TSV data
compute_avg() {
    local col="$1"
    local data="$2"
    echo "$data" | awk -F'\t' -v c="$col" '{ sum += $c; n++ } END { if(n>0) printf "%.2f", sum/n; else print "N/A" }'
}

compute_min() {
    local col="$1"
    local data="$2"
    echo "$data" | awk -F'\t' -v c="$col" 'BEGIN{m=999999999} { if($c+0 < m) m=$c+0 } END { printf "%d", m }'
}

compute_max() {
    local col="$1"
    local data="$2"
    echo "$data" | awk -F'\t' -v c="$col" 'BEGIN{m=0} { if($c+0 > m) m=$c+0 } END { printf "%d", m }'
}
