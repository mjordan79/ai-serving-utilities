#!/usr/bin/env bash
# run.sh — Master benchmark runner.
#
# Usage:
#   ./run.sh                     Run all benchmarks
#   ./run.sh <test>              Run a single test (e.g., 01_simple_chat)
#
# Reads configuration from the project's .env file (parent directory).
# Recovers API_KEY automatically via docker compose.
# Results are written to results/<model_label>/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="${SCRIPT_DIR}/.."
TESTS_DIR="${SCRIPT_DIR}/tests"
RESULTS_DIR="${SCRIPT_DIR}/results"

# Source shared library for report helpers
source "${SCRIPT_DIR}/lib.sh"

# ── Auto-detect configuration from project .env ─────────────────────────────

ENV_FILE="${PROJECT_DIR}/.env"

if [[ ! -f "$ENV_FILE" ]]; then
    echo "ERROR: Project .env not found at ${ENV_FILE}"
    echo "       Make sure you're running this from the benchmark/ directory."
    exit 1
fi

# Parse .env (skip comments and empty lines, handle quoted values)
while IFS='=' read -r key value; do
    # Skip comments and empty lines
    [[ "$key" =~ ^[[:space:]]*# ]] && continue
    [[ -z "$key" ]] && continue
    # Trim whitespace
    key=$(echo "$key" | xargs)
    value=$(echo "$value" | xargs | sed 's/^"//;s/"$//')
    [[ -z "$key" ]] && continue
    export "$key=$value"
done < "$ENV_FILE"

# ── Derive BASE_URL from LETSENCRYPT_DOMAIN ─────────────────────────────────

if [[ -n "${LETSENCRYPT_DOMAIN:-}" ]]; then
    # Try the external domain first; fallback to localhost if unreachable
    if curl -sS -k --max-time 5 --output /dev/null "https://${LETSENCRYPT_DOMAIN}/health" 2>/dev/null; then
        BASE_URL="https://${LETSENCRYPT_DOMAIN}"
    else
        echo "INFO: ${LETSENCRYPT_DOMAIN} not reachable, falling back to https://localhost"
        BASE_URL="https://localhost"
    fi
else
    BASE_URL="http://localhost:1235"
fi
export BASE_URL

# ── Recover API_KEY automatically ───────────────────────────────────────────

# Try docker compose exec first (reads from /root/.vllm-key/.api_key)
API_KEY=""
if command -v docker &>/dev/null; then
    API_KEY=$(docker compose -f "${PROJECT_DIR}/docker-compose.yml" \
        exec --no-deps vllm-server cat /root/.vllm-key/.api_key 2>/dev/null || true)
fi

# Fallback: try docker exec directly (older compose v1)
if [[ -z "$API_KEY" ]] && command -v docker &>/dev/null; then
    API_KEY=$(docker exec vllm-server cat /root/.vllm-key/.api_key 2>/dev/null || true)
fi

# Fallback: ask user
if [[ -z "$API_KEY" ]]; then
    echo "WARNING: Could not auto-detect API_KEY."
    echo "         Docker may not be running or the container is not available."
    echo -n "         Enter VLLM_API_KEY: " >&2
    read -r API_KEY >&2
    if [[ -z "$API_KEY" ]]; then
        echo "ERROR: API_KEY is required. Set it via the running container or enter it manually."
        exit 1
    fi
fi

export API_KEY

# ── Build MODEL_LABEL from MODEL_NAME + QUANTIZATION ────────────────────────

if [[ -n "${QUANTIZATION:-}" ]]; then
    MODEL_LABEL="${MODEL_NAME:-unknown} (${QUANTIZATION})"
else
    MODEL_LABEL="${MODEL_NAME:-unknown}"
fi

# ── Defaults ────────────────────────────────────────────────────────────────

ITERATIONS="${ITERATIONS:-3}"
ENABLE_THINKING="${ENABLE_THINKING:-true}"
CURL_TIMEOUT="${CURL_TIMEOUT:-600}"
export ITERATIONS ENABLE_THINKING CURL_TIMEOUT

# ── Parse arguments ─────────────────────────────────────────────────────────

SINGLE_TEST="${1:-}"

if [[ -n "$SINGLE_TEST" ]]; then
    test_path="${TESTS_DIR}/${SINGLE_TEST}.sh"
    if [[ ! -f "$test_path" ]]; then
        echo "ERROR: Test not found: $test_path"
        echo "Available tests:"
        ls -1 "${TESTS_DIR}/"[0-9]*.sh 2>/dev/null | xargs -I{} basename {} .sh
        exit 1
    fi
fi

# ── Setup output directory ──────────────────────────────────────────────────

MODEL_SAFE=$(echo "${MODEL_LABEL:-unknown}" | tr '/' '_' | tr '.' '-' | tr ' ' '-')
RUN_DIR="${RESULTS_DIR}/${MODEL_SAFE}"
mkdir -p "$RUN_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
TSV_FILE="${RUN_DIR}/results_${TIMESTAMP}.tsv"
MD_FILE="${RUN_DIR}/report_${TIMESTAMP}.md"

# ── Header ──────────────────────────────────────────────────────────────────

echo ""
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║  BENCHMARK SUITE                                        ║"
echo "╠═══════════════════════════════════════════════════════════╣"
echo "║  Target : ${BASE_URL}"
echo "║  Model  : ${MODEL_LABEL:-unknown}"
echo "║  Iters  : ${ITERATIONS}"
echo "║  Output : ${TSV_FILE}"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""

# Write TSV header
echo -e "Test\tIteration\tInputTokens\tOutputTokens\tTTFT_ms\tTotal_ms\tTPS" > "$TSV_FILE"

# ── Collect results ─────────────────────────────────────────────────────────

ALL_RESULTS=""

run_test() {
    local test_script="$1"
    local test_name
    test_name=$(basename "$test_script" .sh)

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "▶ Running: ${test_name}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    local test_output
    test_output=$("${test_script}" 2>&1) || {
        echo "  ✗ FAILED (exit code $?)"
        echo "$test_output" | sed 's/^/    /'
        return 1
    }

    # The test outputs TSV lines to stdout
    echo "$test_output" | grep -E '^[a-z]' || true

    # Append to TSV
    echo "$test_output" | grep -E '^[a-z]' >> "$TSV_FILE" 2>/dev/null || true

    echo ""
}

# ── Execute tests ───────────────────────────────────────────────────────────

if [[ -n "$SINGLE_TEST" ]]; then
    test_path="${TESTS_DIR}/${SINGLE_TEST}.sh"
    if [[ -f "$test_path" ]]; then
        run_test "$test_path"
    else
        echo "ERROR: Test not found: $test_path"
        exit 1
    fi
else
    # Run all tests in order
    for test_script in "${TESTS_DIR}"/[0-9]*.sh; do
        [[ -f "$test_script" ]] || continue
        run_test "$test_script"
    done
fi

# ── Generate report ─────────────────────────────────────────────────────────

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║  GENERATING REPORT                                      ║"
echo "╠═══════════════════════════════════════════════════════════╣"
echo "║  TSV : ${TSV_FILE}"
echo "║  MD  : ${MD_FILE}"
echo "╚═══════════════════════════════════════════════════════════╝"

# Generate markdown report
{
    # Read TSV data (skip header)
    tail -n +2 "$TSV_FILE"
} | write_markdown_summary "$MD_FILE" "${MODEL_LABEL:-unknown}"

# ── Summary table ───────────────────────────────────────────────────────────

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  SUMMARY (per-test averages)"
echo "═══════════════════════════════════════════════════════════"

print_table_header

# Aggregate by test name
tail -n +2 "$TSV_FILE" | awk -F'\t' '
{
    test = $1
    tests[test] = 1
    count[test]++
    sum_ttft[test] += $5
    sum_total[test] += $6
    sum_tps[test] += $7
    sum_in[test] += $3
    sum_out[test] += $4
}
END {
    n = asorti(tests, sorted)
    for (i = 1; i <= n; i++) {
        t = sorted[i]
        c = count[t]
        printf "%-40s %6s %8.0f %8.0f %10.0f %12.0f %10.2f\n", \
            t, "avg", sum_in[t]/c, sum_out[t]/c, sum_ttft[t]/c, sum_total[t]/c, sum_tps[t]/c
    }
}
'

echo ""
echo "Results saved to: ${RUN_DIR}/"
echo ""
