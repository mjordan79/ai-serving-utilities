# Benchmark Suite

by Renato Perini (mjordan79)

Curl-based benchmark suite for vLLM deployments. No Python, no external dependencies beyond `bash`, `curl`, `gawk`, `date`.

## Structure

```
benchmark/
├── run.sh              # Master runner — auto-detects config from project .env
├── compare.sh          # Side-by-side comparison of two runs
├── warmup.sh           # Triton kernel pre-compilation warmup
├── lib.sh              # Shared library (curl wrapper, metrics, reporting)
├── tests/
│   ├── 01_simple_chat.sh        # Short single-turn Q&A
│   ├── 02_long_context.sh       # Long prefill (~4000 input tokens)
│   ├── 03_code_generation.sh    # Code generation (Python)
│   ├── 04_reasoning.sh          # Math/logic with thinking enabled
│   ├── 05_tool_calling.sh       # Function calling
│   ├── 06_long_output.sh        # Long output (up to 4096 tokens)
│   ├── 07_multi_turn.sh         # Multi-turn conversation (3 turns)
│   └── 08_mixed_workload.sh     # Mixed task types (Q&A, code, reasoning, translation)
└── results/                # gitignored — raw data and generated reports
    ├── <model_label>/
    │   ├── results_YYYYMMDD_HHMMSS.tsv   # Raw TSV data
    │   └── report_YYYYMMDD_HHMMSS.md     # Markdown report
    └── comparison/
        └── comparison_YYYYMMDD_HHMMSS.md # Side-by-side report (compare.sh)
```

## Configuration

All configuration is auto-detected from the project's `.env` file (`../.env`):

| .env Variable | Auto-Derived | Used For |
|---------------|--------------|----------|
| `LETSENCRYPT_DOMAIN` | → `BASE_URL` | `https://<domain>` if the domain resolves, else `http://localhost:1235` |
| *(not from .env)* | → `API_KEY` | Recovered live from the running container via `docker compose exec` (falls back to `docker exec`, then an interactive prompt). The `.env` `VLLM_API_KEY` is **never read** by the scripts |
| `MODEL_NAME` | → `MODEL_LABEL` | Result directory naming |
| `QUANTIZATION` | → `MODEL_LABEL` | Appended to model label (e.g., `nvidia/Qwen3.6-27B-NVFP4 (modelopt)`) |

No manual config files needed. The scripts read the project state directly.

## Prerequisites

- `bash` (5.0+)
- `curl` (with `-k` support for SSL)
- `gawk` (uses the gawk-only `asorti` in `run.sh` and `compare.sh`; plain `mawk`/`awk` will not work)
- `date` with `%N` support (nanoseconds)

No `jq` required — all JSON parsing is handled via `gawk`/`sed`.

## Usage

> Scripts are committed without the executable bit (Windows convention) — invoke with `bash`.

### 1. Warmup (required before benchmarking)

```bash
cd benchmark
bash warmup.sh
```

Auto-detects the project `.env`, recovers the API key via Docker, and sends diverse prompts to pre-compile Triton kernels. **Skip this and your first iteration will be artificially slow.**

### 2. Run benchmarks

```bash
# All tests
bash run.sh

# Single test
bash run.sh 01_simple_chat
```

### 3. Compare results

```bash
bash compare.sh 'results/unsloth_Qwen3-8-27B-NVFP4-(compressed-tensors)' 'results/nvidia_Qwen3-6-27B-NVFP4-(modelopt)'
```

Directory names are produced by `run.sh` from `MODEL_NAME + " (${QUANTIZATION})"` sanitized via `tr` (`/` → `_`, `.` → `-`, space → `-`). The markdown report is written to `results/comparison/`.

## Metrics

| Metric | Description | Better |
|--------|-------------|--------|
| **TTFT** | Time To First Token (ms) | Lower |
| **Total** | End-to-end latency (ms) | Lower |
| **TPS** | Tokens Per Second | Higher |
| **InTok** | Input/prompt tokens | — |
| **OutTok** | Output/completion tokens | — |

## Test Matrix

| Test | Input | Output | Path | Purpose |
|------|-------|--------|------|---------|
| `simple_chat` | ~15 tok | ~50 tok | Standard | Baseline latency |
| `long_context` | ~4000 tok | ~50 tok | Long prefill | Prefill throughput |
| `code_generation` | ~50 tok | ~300 tok | Code | Code output quality + speed |
| `reasoning` | ~40 tok | ~500 tok | Thinking | CoT/reasoning path |
| `tool_calling` | ~100 tok | ~30 tok | Tools | Function calling overhead |
| `long_output` | ~80 tok | ~2000 tok | Long decode | Sustained generation TPS |
| `multi_turn` | 15→100→200 tok | ~100 tok/turn | Growing context | Context window scaling |
| `mixed_workload` | varies | varies | All | Realistic mixed pattern |

## Output Format

Raw results are TSV (tab-separated):

```
Test    Iteration    InputTokens    OutputTokens    TTFT_ms    Total_ms    TPS
simple_chat    1    15    52    245    1890    27.51
simple_chat    2    15    48    238    1750    27.43
simple_chat    3    15    51    241    1820    28.02
```

## Notes

- `MAX_NUM_SEQS=1` in the deployment means only one concurrent request. These benchmarks are sequential by design.
- Run both NVIDIA and Unsloth on the **same hardware, same load** for a fair comparison.
- GPU temperature affects performance. Let the system stabilize between runs.
- The warmup script is not optional — Triton kernel compilation on first use adds significant latency.

### vLLM v0.27.1 Specifics

- **No usage in SSE chunks**: vLLM v0.27.1 does NOT send `usage` in streaming chunks. Token counts are extracted from non-streaming responses (`stream=false`) for authoritative counts.
- **TTFT from metrics**: Non-streaming responses include `metrics.time_to_first_token_ms` for accurate TTFT measurement.
- **Reasoning/Thinking**: Model uses `"reasoning"` field during thinking phase, `"content"` for final response. `json_get_delta_content()` handles both.
- **Health endpoint**: `/health` returns HTTP 200 with **empty body** (no JSON). Use `/v1/models` for model verification (requires auth).

### JSON Parsing (No jq)

All JSON parsing uses `gawk`/`sed` — no external tools required. When constructing payloads with `printf`:

```bash
# CORRECT: \\n produces escaped newline in JSON
printf '{"content":"line1\\n\\nline2"}'

# WRONG: \n produces literal newline (invalid JSON)
printf '{"content":"line1\n\nline2"}'
```

