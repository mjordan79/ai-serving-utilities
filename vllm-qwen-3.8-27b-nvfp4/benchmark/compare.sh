#!/usr/bin/env bash
# compare.sh — Side-by-side comparison of two benchmark runs.
#
# Usage:
#   ./compare.sh <results_dir_a> <results_dir_b>
#
# Finds the latest TSV in each directory and generates a comparison report.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/lib.sh"

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <results_dir_a> <results_dir_b>"
    echo ""
    echo "Finds the latest TSV file in each directory and generates"
    echo "a side-by-side comparison report."
    echo ""
    echo "Example (dir names are produced by run.sh — MODEL_NAME + ' (<quantization>)'"
    echo "sanitized via tr: / → _, . → -, space → -):"
    echo "  $0 'results/unsloth_Qwen3-8-27B-NVFP4-(compressed-tensors)' 'results/nvidia_Qwen3-6-27B-NVFP4-(modelopt)'"
    exit 1
fi

DIR_A="$1"
DIR_B="$2"

# Find latest TSV in each dir
TSV_A=$(ls -t "${DIR_A}"/results_*.tsv 2>/dev/null | head -1)
TSV_B=$(ls -t "${DIR_B}"/results_*.tsv 2>/dev/null | head -1)

if [[ -z "${TSV_A:-}" ]]; then
    echo "ERROR: No TSV found in $DIR_A"
    exit 1
fi
if [[ -z "${TSV_B:-}" ]]; then
    echo "ERROR: No TSV found in $DIR_B"
    exit 1
fi

LABEL_A=$(basename "$DIR_A")
LABEL_B=$(basename "$DIR_B")

OUTPUT_DIR="${SCRIPT_DIR}/results/comparison"
mkdir -p "$OUTPUT_DIR"
REPORT="${OUTPUT_DIR}/comparison_$(date +%Y%m%d_%H%M%S).md"

echo "═══════════════════════════════════════════════════════════"
echo "  COMPARISON: ${LABEL_A} vs ${LABEL_B}"
echo "  A: $(basename "$TSV_A")"
echo "  B: $(basename "$TSV_B")"
echo "═══════════════════════════════════════════════════════════"
echo ""

# ── Aggregate each dataset ──────────────────────────────────────────────────

aggregate_tsv() {
    local tsv="$1"
    tail -n +2 "$tsv" | awk -F'\t' '
    {
        test = $1
        tests[test] = 1
        count[test]++
        sum_ttft[test] += $5
        sum_total[test] += $6
        sum_tps[test] += $7
        sum_in[test] += $3
        sum_out[test] += $4
        if (!(test in min_ttft) || $5 < min_ttft[test]) min_ttft[test] = $5
        if (!(test in max_ttft) || $5 > max_ttft[test]) max_ttft[test] = $5
        if (!(test in min_total) || $6 < min_total[test]) min_total[test] = $6
        if (!(test in max_total) || $6 > max_total[test]) max_total[test] = $6
        if (!(test in max_tps) || $7 > max_tps[test]) max_tps[test] = $7
        if (!(test in min_tps) || $7 < min_tps[test]) min_tps[test] = $7
    }
    END {
        n = asorti(tests, sorted)
        for (i = 1; i <= n; i++) {
            t = sorted[i]
            c = count[t]
            printf "%s\t%d\t%.0f\t%.0f\t%.0f\t%.0f\t%.0f\t%.0f\t%.2f\t%.2f\t%.0f\t%.0f\n", \
                t, c, sum_in[t]/c, sum_out[t]/c, \
                sum_ttft[t]/c, min_ttft[t], max_ttft[t], \
                sum_total[t]/c, sum_tps[t], max_tps[t], min_tps[t], \
                min_total[t], max_total[t]
        }
    }'
}

DATA_A=$(aggregate_tsv "$TSV_A")
DATA_B=$(aggregate_tsv "$TSV_B")

# ── Generate markdown report ────────────────────────────────────────────────

cat > "$REPORT" <<EOF
# Benchmark Comparison: ${LABEL_A} vs ${LABEL_B}

- **Date:** $(date '+%Y-%m-%d %H:%M:%S %Z')
- **Run A:** ${TSV_A}
- **Run B:** ${TSV_B}

## Overview

| Test | Iters | Metric | ${LABEL_A} | ${LABEL_B} | Winner | Delta |
|------|-------|--------|-----------|-----------|--------|-------|
EOF

# Join and compare
echo "$DATA_A" | while IFS=$'\t' read -r test cnt_a in_a out_a ttft_a ttft_min_a ttft_max_a total_a tps_a tps_max_a tps_min_a total_min_a total_max_a; do
    # Find matching row in B
    row_b=$(echo "$DATA_B" | grep "^${test}	" || echo "")
    if [[ -z "$row_b" ]]; then
        echo "| ${test} | ${cnt_a} | — | — | No matching data in B | — | — |" >> "$REPORT"
        continue
    fi

    IFS=$'\t' read -r _ cnt_b in_b out_b ttft_b ttft_min_b ttft_max_b total_b tps_b tps_max_b tps_min_b total_min_b total_max_b <<< "$row_b"

    # Determine winners (lower is better for latency, higher for TPS)
    ttft_winner=$(awk "BEGIN { print (${ttft_a} <= ${ttft_b}) ? \"A\" : \"B\" }")
    total_winner=$(awk "BEGIN { print (${total_a} <= ${total_b}) ? \"A\" : \"B\" }")
    tps_winner=$(awk "BEGIN { print (${tps_a} >= ${tps_b}) ? \"A\" : \"B\" }")

    ttft_delta=$(awk "BEGIN { d = (${ttft_a} - ${ttft_b}) / ${ttft_b} * 100; printf \"%+.1f%%\", d }")
    total_delta=$(awk "BEGIN { d = (${total_a} - ${total_b}) / ${total_b} * 100; printf \"%+.1f%%\", d }")
    tps_delta=$(awk "BEGIN { d = (${tps_a} - ${tps_b}) / ${tps_b} * 100; printf \"%+.1f%%\", d }")

    cat >> "$REPORT" <<EOF
| ${test} | ${cnt_a}/${cnt_b} | TTFT (ms) | ${ttft_a} | ${ttft_b} | **${ttft_winner}** | ${ttft_delta} |
| ${test} | ${cnt_a}/${cnt_b} | Total (ms) | ${total_a} | ${total_b} | **${total_winner}** | ${total_delta} |
| ${test} | ${cnt_a}/${cnt_b} | TPS | ${tps_a} | ${tps_b} | **${tps_winner}** | ${tps_delta} |
| ${test} | ${cnt_a}/${cnt_b} | TTFT range | [${ttft_min_a}-${ttft_max_a}] | [${ttft_min_b}-${ttft_max_b}] | — | — |
| ${test} | ${cnt_a}/${cnt_b} | Total range | [${total_min_a}-${total_max_a}] | [${total_min_b}-${total_max_b}] | — | — |
| | | | | | | |
EOF
done

cat >> "$REPORT" <<'EOF'

## Legend

- **TTFT** = Time To First Token (lower is better)
- **Total** = End-to-end latency (lower is better)
- **TPS** = Tokens Per Second (higher is better)
- **Winner** = Which model performed better for that metric
- **Delta** = Percentage difference (A relative to B)

## Notes

- Each test runs multiple iterations; values shown are averages.
- Range shows [min, max] across iterations.
- Results depend on system load, GPU temperature, and other factors.
- Run both benchmarks on the same hardware for a fair comparison.
EOF

# ── Console summary ─────────────────────────────────────────────────────────

echo ""
echo "Comparison report: ${REPORT}"
echo ""

# Print console table
printf "%-30s %12s %12s %8s\n" "Test" "${LABEL_A} TTFT" "${LABEL_B} TTFT" "Winner"
printf "%-30s %12s %12s %8s\n" "------------------------------" "------------" "------------" "--------"

echo "$DATA_A" | while IFS=$'\t' read -r test _ _ _ ttft_a _ _ _ _ _ _ _ _; do
    row_b=$(echo "$DATA_B" | grep "^${test}	" || echo "")
    if [[ -z "$row_b" ]]; then
        printf "%-30s %12s %12s %8s\n" "$test" "$ttft_a" "N/A" "—"
        continue
    fi
    IFS=$'\t' read -r _ _ _ _ ttft_b _ _ _ _ _ _ _ _ <<< "$row_b"
    winner=$(awk "BEGIN { print (${ttft_a} <= ${ttft_b}) ? \"A\" : \"B\" }")
    printf "%-30s %12s %12s %8s\n" "$test" "$ttft_a" "$ttft_b" "$winner"
done

echo ""
