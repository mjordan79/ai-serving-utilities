# NVIDIA Nemotron 3.5 Lightning 30B A3B (NVFP4) — vLLM Server

Self-contained vLLM deployment for the **NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4**
checkpoint (ModelOpt NVFP4 W4A16) from HuggingFace.

- **Producer:** NVIDIA — **Publisher:** NVIDIA
- **Architecture:** hybrid Mamba (SSM) + MoE attention — 30B total parameters, ~3B active per token. Because of the hybrid layout vLLM routes it through the V1 model runner on WDDM (see *Known issues*) and applies Mamba-cache flags that a plain dense model would not use.
- **Model ID:** `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4` (~18 GB download)
- **Base image:** `vllm/vllm-openai:v0.27.1` (pinned to the minimum version of the official vLLM deployment recipe for this model)
- **vLLM flags:** base-recipe args (flashinfer Mamba backend, `align` cache mode, prefix caching) + NVFP4 variant (fp8 KV cache, Marlin MoE kernel) + built-in MTP speculative decoding (3 tokens, Triton MoE backend). Hopper-only overrides are exposed but **off by default** (see *Hopper-only overrides*): this repo targets a RTX 5090 (sm_120), which the recipe does not list.
- **Local endpoint:** `http://localhost:1237` — container `vllm-nemotron-server`
- **Reverse proxy (optional):** `docker-compose.proxy.yml` on DuckDNS (`your-domain.duckdns.org`) — see *Reverse Proxy (optional)*. Only one stack may hold ports 80/443 at a time (Qwen on 1235, Muse on 1236, this stack on 1237).

## Prerequisites

- Linux x86_64 with NVIDIA GPU, ≥24 GB VRAM recommended for the NVFP4 checkpoint. Tested with 32 GB (RTX 5090, WSL2).
- The GPU must be **dedicated to this stack** or `GPU_MEMORY_UTILIZATION` lowered accordingly (see *VRAM escalation ladder*). The default 0.88 assumes another vLLM stack (Qwen, ~2.6 GiB resident) shares the card.
- NVIDIA Container Toolkit (vLLM is started with `runtime: nvidia`).
- HuggingFace account with access to the gated checkpoint (set `HF_TOKEN`).
- **Reverse proxy (optional):** a DuckDNS subdomain pointing at the host (e.g., `your-domain.duckdns.org`). The domain must resolve before the certificate can be issued.

## Setup

### 1. Configure `.env`

```sh
cp .env.example .env
```

Edit `.env` (only the listed keys exist — all tuning flags live in `docker-compose.yml` / `entrypoint.sh` with their defaults):

- `HF_TOKEN` — your HuggingFace token (the checkpoint is gated).
- `VLLM_API_KEY` — **optional.** Leave it commented and the entrypoint generates a `sk-...` key on first boot, persists it in the `vllm-keys` volume, and prints it in the container log. Uncomment to pin a fixed key instead.
- `LETSENCRYPT_DOMAIN` / `LETSENCRYPT_EMAIL` — only used by the reverse-proxy overlay.

### 2. First run

```sh
docker compose up -d
```

This builds the image, pulls the checkpoint (~18 GB) into the `hf-cache-nemotron` volume, and starts the server. Watch the logs:

```sh
docker compose logs -f
```

First boot also prints the generated API key (if you did not pin one):

```sh
docker compose logs | grep -i "api key"
```

### 3. Wait for readiness

The container reports healthy once vLLM passes its startup health check (model load + CUDA graph capture takes several minutes):

```sh
curl http://localhost:1237/health
```

### 4. Reverse proxy (optional)

```sh
docker compose -f docker-compose.yml -f docker-compose.proxy.yml up -d --force-recreate
```

Requires the DuckDNS subdomain to be registered first (`LETSENCRYPT_DOMAIN` must resolve). Only one stack at a time may run the proxy overlay (ports 80/443): Qwen (1235), Muse Glimmer (1236), or this stack (1237).

## Usage

```sh
# Chat completion
curl http://localhost:1237/v1/chat/completions \
  -H "Authorization: Bearer $VLLM_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4",
    "messages": [
      {"role": "user", "content": "Explain quantization in one sentence."}
    ],
    "stream": true
  }'
```

With the reverse proxy, use `https://<your-domain.duckdns.org>` in place of `http://localhost:1237`.

**Coding agents:** the model card recommends forcing non-empty message content so tool calls are not dropped mid-stream. With the Python `openai` client this is a client-side parameter, not server configuration:

```python
extra_body={"chat_template_kwargs": {"force_nonempty_content": True}}
```

## Configuration

Tuning flags are set in `entrypoint.sh` (defaults) and overridden via `docker-compose.yml` (`environment:` pass-through). Setting any of the keys below in `.env` is sufficient — it propagates to the container.

| Variable | Default | Description |
|---|---|---|
| `MODEL_NAME` | `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4` | HuggingFace model ID |
| `QUANTIZATION` | `modelopt_fp4` | Quantization method. Leave empty to let vLLM auto-detect from the checkpoint config |
| `HF_CACHE_VOLUME` | `hf-cache-nemotron` | Named volume holding the HuggingFace cache (checkpoint + weights) |
| `VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR` | `/vllm-cache/flashinfer_autotune` | FlashInfer kernel autotune cache, persisted in the `vllm-flashinfer-cache` volume. vLLM's default (`/tmp`) is wiped on container recreation and costs a ~15-30 s re-autotune at every boot |
| `MAX_MODEL_LEN` | `65536` | Maximum context length (tokens). Raised 32K → 64K on 2026-08-26: GitHub Copilot + vllm-copilot prompts run ~29.5K tokens (instruction files, MCP schemas, editor context) and vLLM hard-rejects any request with `max_tokens + prompt > max_model_len`, so a 32K window left ~3K output headroom and 400'd deterministically. At 64K the full prompt + a 32K output budget fits (29.5K + 32K = 61.5K < 65.5K). Cost: the KV cache holds ~8 concurrent 64K sequences instead of ~16 — irrelevant for single-user interactive serving; drop back to `32768` if raw concurrency matters |
| `DTYPE` | `auto` | Data type (`auto`, `bfloat16`, `float16`) |
| `TP_SIZE` | `1` | Tensor parallelism size |
| `ATTENTION_BACKEND` | *(empty)* | Attention backend. Leave empty to let vLLM auto-select; forcing one is not advised on this hybrid architecture (validated failure mode on the Gemma post-mortem) |
| `PERFORMANCE_MODE` | `interactivity` | vLLM performance mode |
| `KV_CACHE_DTYPE` | `fp8_e4m3` | KV cache dtype (NVFP4 variant of the recipe) |
| `GPU_MEMORY_UTILIZATION` | `0.92` | Fraction of GPU memory for weights + KV cache. Raised 0.88 → 0.92 on 2026-08-26 (dedicated-card profile): at 0.88 the first-boot log showed a 4.05 GiB KV cache vs ~5.55 GiB at full utilization. With the Qwen stack co-running on the same 5090 (~2.6 GiB held), drop to `0.88` — the 0.92 target (29.3 of 31.84 GiB) leaves only ~0.9 GiB of headroom against free memory |
| `MAX_NUM_SEQS` | `8` | Max concurrent sequences (conservative start; Hopper-only target is 256) |
| `MAX_NUM_BATCHED_TOKENS` | `8192` | Max batched tokens per step (recipe base is 16384; Hopper-only target is 32768) |
| `ENABLE_CHUNKED_PREFILL` | `true` | Chunk long prefill to protect TTFT under load |
| `ENABLE_PREFIX_CACHING` | `true` | Cache shared prompt prefixes |
| `ENABLE_HYBRID_KV_CACHE_MANAGER` | `true` | Hybrid KV cache manager (Mamba + attention) |
| `MOE_BACKEND` | `marlin` | MoE kernel (NVFP4 variant of the recipe) |
| `LINEAR_BACKEND` | *(empty)* | **Hopper-only.** Set to `humming` on H100/H200 only |
| `MAMBA_BACKEND` | `flashinfer` | SSM backend. If flashinfer fails on sm_120, fall back to `triton` |
| `MAMBA_CACHE_MODE` | `align` | Mamba cache alignment mode (recipe base) |
| `MAMBA_SSU_ALGORITHM` | *(empty)* | **Hopper-only.** Set to `horizontal` on H100/H200 only |
| `MAMBA_SSM_CACHE_DTYPE` | `float16` | Fast SSM cache: half-precision SSM state cache (recipe feature) |
| `ENABLE_MAMBA_CACHE_STOCHASTIC_ROUNDING` | `true` | Fast SSM cache: stochastic rounding of the SSM cache (recipe feature) |
| `MAMBA_CACHE_PHILOX_ROUNDS` | `5` | Fast SSM cache: Philox RNG rounds (recipe feature) |
| `ASYNC_SCHEDULING` | *(empty)* | **Hopper-only.** Async scheduling; set on H100/H200 only |
| `ENABLE_MTP` | `true` | MTP (multi-token prediction) speculative decoding — built into the checkpoint |
| `MTP_NUM_SPECULATIVE_TOKENS` | `3` | Speculative tokens per step (recipe Blackwell value) |
| `MTP_MOE_BACKEND` | `triton` | MoE backend inside the speculative config; empty omits the key from the JSON |
| `REASONING_PARSER` | `nemotron_v3` | Reasoning (thinking) block parser (recipe feature) |
| `ENABLE_AUTO_TOOL_CHOICE` | `true` | Enable automatic tool choice |
| `TOOL_CALL_PARSER` | `qwen3_coder` | Tool-call parser. `qwen3_coder` appears in every official NVIDIA snippet (vLLM/TRT-LLM/SGLang) for this checkpoint — it supersedes the vLLM recipe's `qwen3_xml` |
| `ENABLE_PROMPT_TOKENS_DETAILS` | `true` | Prompt tokens details in responses |
| `ENABLE_REQUEST_METRICS` | `false` | Per-request metrics (requires `DISABLE_LOG_STATS=false`) |
| `DISABLE_LOG_STATS` | `true` | Disable periodic log stats |
| `TRUST_REMOTE_CODE` | `true` | Trust remote code in the model repo |
| `EXTRA_ARGS_STR` | *(empty)* | Raw extra flags appended verbatim to `vllm serve` (escape hatch, e.g. `--num-gpu-workers 1`) |
| `PORT` | `8000` | Container port |
| `HF_TOKEN` | — | HuggingFace token (gated checkpoint) |

### Hopper-only overrides

The vLLM recipe defines overrides for Hopper (H100/H200) that are **not** part of the base args and are left off here because this repo targets sm_120 (RTX 5090), which the recipe does not cover. On Hopper hardware, enable them in `.env`:

```sh
LINEAR_BACKEND=humming
MAMBA_SSU_ALGORITHM=horizontal
ASYNC_SCHEDULING=true
MAX_NUM_SEQS=256
MAX_NUM_BATCHED_TOKENS=32768
```

### Speculative decoding

MTP is on by default (`ENABLE_MTP=true`, 3 tokens, Triton MoE backend) — it is built into the checkpoint and requires no separate draft model. If MTP misbehaves on this hardware, set `ENABLE_MTP=false`.

**DSpark (next-iteration option, not wired up):** the model card recommends a DSpark drafter over the built-in MTP for low-concurrency, latency-sensitive serving — which is exactly this stack's profile (single GPU, interactive). DSpark is a separate checkpoint: `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark`, and the card's recipe pairs it with `--speculative_config.model=<DSpark-checkpoint>` and `--speculative_config.num_speculative_tokens 3`. Before switching, collect the MTP acceptance rate from `/metrics` so the comparison is measured, not assumed.

### VRAM escalation ladder

If the server OOMs during startup (CUDA OOM in logs), lower in this order and retry:

1. `MAX_MODEL_LEN=16384`
2. `GPU_MEMORY_UTILIZATION=0.85`
3. `MAX_NUM_SEQS=4`

If it starts but dies under load, apply the same three dials in the same order.

## Known issues

- **WDDM / WSL2:** `VLLM_USE_V2_MODEL_RUNNER=0` is hardcoded in `docker-compose.yml`. The vLLM V2 model runner fails its UVA buffer check on WDDM, and hybrid Mamba/attention models default to the V2 runner — without this pin the container crash-loops at startup (validated Gemma post-mortem).
- **Do not force `ATTENTION_BACKEND`:** forcing a backend on this hybrid architecture is a validated crash mode (Gemma post-mortem). Leave it empty; vLLM auto-selects.
- **`MAMBA_BACKEND=flashinfer` on sm_120:** the recipe pins flashinfer for the Mamba backend; if it fails to initialize on sm_120, fall back to `MAMBA_BACKEND=triton` in `.env`.
- **GPU sharing:** the default `GPU_MEMORY_UTILIZATION=0.92` assumes the 5090 is dedicated to this stack. With the Qwen stack (1235) co-running (~2.6 GiB held), boot still succeeds but the 0.92 target sits within ~0.9 GiB of free memory — a WDDM OOM risk under load; use `GPU_MEMORY_UTILIZATION=0.88` in `.env` for the co-running profile.

## Useful commands

```sh
# Logs
docker compose logs -f

# Stop / start (image and volumes are preserved)
docker compose down
docker compose up -d

# Full teardown (removes containers + network; data volumes remain)
docker compose down

# Full teardown including the HF cache + API key + autotune cache volumes (~18 GB + key)
docker compose down -v

# Inspect volumes
docker volume ls | grep vllm-nemotron
```
