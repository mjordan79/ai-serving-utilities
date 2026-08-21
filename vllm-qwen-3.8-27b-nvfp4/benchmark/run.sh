#!/usr/bin/env bash
# run.sh — Master benchmark runner.
#
# Usage (invoke via bash — the scripts are stored in git without the exec bit):
#   bash run.sh                  Run all benchmarks
#   bash run.sh <test>           Run a single test (e.g., 01_simple_chat)
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

# Parse .env, resolve BASE_URL and recover the API key (shared with warmup.sh — lib.sh)
parse_project_env "$ENV_FILE"
resolve_base_url
recover_api_key

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
FAILED_TESTS=0
FAILED_TEST_NAMES=""

run_test() {
    local test_script="$1"
    local test_name
    test_name=$(basename "$test_script" .sh)

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "▶ Running: ${test_name}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # Invoked via `bash` (git stores the tests without the exec bit).
    # A failing test is recorded but does NOT abort the suite (set -e would otherwise
    # exit here and skip report generation); the final exit code reflects failures.
    local test_output rc=0
    test_output=$(bash "${test_script}" 2>&1) || rc=$?
    if [[ $rc -ne 0 ]]; then
        echo "  ✗ FAILED (exit code ${rc}) — continuing with remaining tests"
        echo "$test_output" | sed 's/^/    /'
        FAILED_TESTS=$((FAILED_TESTS + 1))
        FAILED_TEST_NAMES="${FAILED_TEST_NAMES}${FAILED_TEST_NAMES:+, }${test_name}"
        return 0
    fi

    # The test outputs TSV lines to stdout. Filter on the TSV shape (7 tab-separated
    # columns, numeric iteration) rather than the first character — stderr merged
    # into $test_output can contain lowercase lines (e.g. "curl: (6) ...") that
    # would otherwise be written as spurious TSV rows.
    echo "$test_output" | awk -F'\t' 'NF == 7 && $2 ~ /^[0-9]+$/ && $5 ~ /^[0-9]+$/ && $6 ~ /^[0-9]+$/ && $7 ~ /^[0-9.]+$/ { print }' | tee -a "$TSV_FILE" >/dev/null

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

# Exit non-zero if any test failed (the suite completed and the report was written).
if [[ $FAILED_TESTS -gt 0 ]]; then
    echo "NOTE: ${FAILED_TESTS} test(s) failed: ${FAILED_TEST_NAMES}"
    exit 1
fi
