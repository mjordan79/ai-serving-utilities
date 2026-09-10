# NVIDIA Nemotron 3.5 Lightning 30B A3B (NVFP4) — vLLM Server

Self-contained vLLM deployment for the **NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4**
checkpoint (ModelOpt NVFP4 W4A16) from HuggingFace.

- **Producer:** NVIDIA — **Publisher:** NVIDIA
- **Architecture:** hybrid Mamba (SSM) + MoE attention — 30B total parameters, ~3B active per token. Because of the hybrid layout the stack pins the V2 model runner explicitly (see *Known issues*) and applies Mamba-cache flags that a plain dense model would not use.
- **Model ID:** `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4` (~18 GB download)
- **Base image:** `vllm/vllm-openai:v0.29.0` (current stable vLLM release; the official deployment recipe's minimum for this model is v0.27.1)
- **vLLM flags:** base-recipe args (flashinfer Mamba backend, `align` cache mode, prefix caching) + NVFP4 variant (fp8 KV cache, MoE kernel auto-selected per model part: MARLIN for the quantized main model, a triton-family backend for the unquantized MTP draft MoE) + built-in MTP speculative decoding (3 speculative tokens per step). Hopper-only overrides are exposed but **off by default** (see *Hopper-only overrides*): this repo targets an RTX 5090 (sm_120), which the recipe does not list.
- **Local endpoint:** `http://localhost:1237` — container `vllm-nemotron-server`
- **Reverse proxy (optional):** `docker-compose.proxy.yml` on DuckDNS (`your-domain.duckdns.org`) — see *Reverse Proxy (optional)*. Only one stack may hold ports 80/443 at a time (Qwen on 1235, Muse on 1236, this stack on 1237, Qwen-SGLang on 1238, Gemma 4 on 1239).

## Prerequisites

- Linux x86_64 with NVIDIA GPU, ≥24 GB VRAM recommended for the NVFP4 checkpoint. Tested with 32 GB (RTX 5090, WSL2).
- The default 0.94 targets the dedicated-card profile: the RTX 5090 serves this model exclusively. If another vLLM stack (Qwen, ~2.6 GiB resident) shares the card, lower `GPU_MEMORY_UTILIZATION` accordingly (see *VRAM escalation ladder*).
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

Requires the DuckDNS subdomain to be registered first (`LETSENCRYPT_DOMAIN` must resolve). Only one stack at a time may run the proxy overlay (ports 80/443): Qwen (1235), Muse Glimmer (1236), Qwen-SGLang (1238), Gemma 4 (1239), or this stack (1237).

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
| `MAX_MODEL_LEN` | `65536` | Maximum context length (tokens). Must hold prompt + output together: vLLM hard-rejects any request with `prompt + max_tokens > MAX_MODEL_LEN`. Agent IDE prompts (instruction files, MCP schemas, editor context) run ~29.5K tokens, so `29.5K prompt + 32K output = 61.5K < 65.5K` fits. Cost: at 64K the KV cache holds ~8 concurrent sequences vs ~16 at 32K — irrelevant for single-user interactive serving; drop back to `32768` if raw concurrency matters |
| `DTYPE` | `auto` | Data type (`auto`, `bfloat16`, `float16`) |
| `TP_SIZE` | `1` | Tensor parallelism size |
| `ATTENTION_BACKEND` | *(empty)* | Attention backend. Leave empty to let vLLM auto-select; forcing one is not advised on this hybrid architecture (validated failure mode on the Gemma post-mortem) |
| `PERFORMANCE_MODE` | `interactivity` | vLLM performance mode |
| `KV_CACHE_DTYPE` | `fp8_e4m3` | KV cache dtype (NVFP4 variant of the recipe) |
| `GPU_MEMORY_UTILIZATION` | `0.94` | Fraction of GPU memory for weights + KV cache. 0.94 targets the dedicated-card profile: the RTX 5090 serves this model exclusively, so higher utilization translates directly into KV cache headroom |
| `MAX_NUM_SEQS` | `8` | Max concurrent sequences (conservative start; Hopper-only target is 256) |
| `MAX_NUM_BATCHED_TOKENS` | `8192` | Max batched tokens per step (recipe base is 16384; Hopper-only target is 32768) |
| `MAX_NUM_QUEUED_REQS` | `32` | Admission cap: max in-flight requests (waiting + running); `32` = `4× MAX_NUM_SEQS` (`8`). Overflow → HTTP `503` |
| `MAX_NUM_QUEUED_TOKENS` | `256K` | Admission cap: max queued prefill tokens counted conservatively (prefix-cache hits are not subtracted); `256K` = `32 × MAX_NUM_BATCHED_TOKENS`, one full batched-token window per admitted request. Overflow → HTTP `503` |
| `ENABLE_CHUNKED_PREFILL` | `true` | Chunk long prefill to protect TTFT under load |
| `ENABLE_PREFIX_CACHING` | `true` | Cache shared prompt prefixes |
| `ENABLE_HYBRID_KV_CACHE_MANAGER` | `true` | Hybrid KV cache manager (Mamba + attention) |
| `MOE_BACKEND` | *(empty)* | MoE kernel; empty lets the vLLM oracle auto-select a supported backend per model part (the unquantized MTP draft model rejects marlin). Set `marlin` only when MTP is off |
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
| `MTP_MOE_BACKEND` | `triton` | Emits a `"moe_backend"` key into the speculative-config JSON; empty omits the key. On 0.29.0 this key is not honored on the unquantized MTP-draft path — the draft MoE backend follows the global `MOE_BACKEND` selection (empty → vLLM oracle auto-selects) |
| `PER_REQUEST_SPEC_DECODE_METRICS` | `none` | Per-request spec-decode acceptance metrics in the response (`metrics.speculative_decoding`): `none` off \| `summary` \| `detailed` |
| `REASONING_PARSER` | `nemotron_v3` | Reasoning (thinking) block parser (recipe feature) |
| `ENABLE_AUTO_TOOL_CHOICE` | `true` | Enable automatic tool choice |
| `TOOL_CALL_PARSER` | `qwen3_coder` | Tool-call parser. `qwen3_coder` appears in every official NVIDIA snippet (vLLM/TRT-LLM/SGLang) for this checkpoint — it supersedes the vLLM recipe's `qwen3_xml` |
| `ENABLE_PROMPT_TOKENS_DETAILS` | `true` | Prompt tokens details in responses |
| `ENABLE_REQUEST_METRICS` | `false` | Per-request metrics (requires `DISABLE_LOG_STATS=false`) |
| `DISABLE_LOG_STATS` | `true` | Disable periodic log stats |
| `TRUST_REMOTE_CODE` | `false` | Pass `--trust-remote-code`. `false` for this stack (Nemotron 3.5 is a built-in vLLM architecture and the NVFP4 checkpoint is handled natively via `modelopt_fp4` — no custom HF modeling code expected); set `true` only if the checkpoint fails to load with a `--trust-remote-code` error |
| `EXTRA_ARGS_STR` | *(empty)* | Raw extra flags appended verbatim to `vllm serve` (escape hatch, e.g. `--num-gpu-workers 1`) |
| `PORT` | `8000` | Container port |
| `HF_TOKEN` | — | HuggingFace token (gated checkpoint) |

### Optional: vLLM-Copilot budget

vLLM-Copilot is an **optional** VS Code client for this stack — the server needs no client-side tuning and serves any OpenAI-compatible consumer out of the box. The parameters below are the recommended entry if you use the extension: ~29.5K-token agent prompts plus a 32K output budget fit the 64K window, so vLLM never 400s the request.

Recommended model entry (`vllm/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4`):

| Parameter | Value | Rationale |
|---|---|---|
| `maxOutputTokens` | `32768` | `MAX_MODEL_LEN=65536`: agent prompts run ~29.5K tokens, so `29.5K + 32K = 61.5K < 65.5K` fits without a deterministic 400 (vLLM hard-rejects `prompt + max_tokens > MAX_MODEL_LEN`). A `max_tokens` above `MAX_MODEL_LEN` is unsatisfiable. |
| `maxInputTokens` | *(unset)* | Auto-computed as `65536 − 32768 = 32768`. |
| `defaultParams` | `{ temperature: 1, top_p: 0.95 }` | Matches the checkpoint's sampling profile (already present in the entry). |

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

MTP is on by default (`ENABLE_MTP=true`, 3 speculative tokens per step). The MTP block — one attention layer plus one MoE layer (`num_nextn_predict_layers: 1`, 270 MTP tensors in the weight index) — is built into the checkpoint, so no separate draft model is needed. The draft's MoE kernel follows the global `MOE_BACKEND` selection: empty lets the vLLM oracle auto-select a supported backend per model part (triton-family for the unquantized draft). If MTP misbehaves on the target GPU, set `ENABLE_MTP=false`.

**Per-request spec-decode metrics:** `PER_REQUEST_SPEC_DECODE_METRICS` (default `none`) controls the experimental `metrics.speculative_decoding` response field: `summary` adds mean acceptance length, draft acceptance rate and a step-by-draft-length histogram; `detailed` additionally records the ordered per-step accepted/proposed arrays. Reported only for single-sequence requests (`n=1`); independent of `DISABLE_LOG_STATS`; vLLM refuses to start if set to a non-`none` value while speculative decoding is disabled.

**DSpark (next-iteration option, not wired up):** the model card recommends a DSpark drafter over the built-in MTP for low-concurrency, latency-sensitive serving — which is exactly this stack's profile (single GPU, interactive). DSpark is a separate checkpoint: `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark`, and the card's recipe pairs it with `--speculative_config.model=<DSpark-checkpoint>` and `--speculative_config.num_speculative_tokens 3`. Before switching, collect the MTP acceptance rate from `/metrics` so the comparison is measured, not assumed.

### VRAM escalation ladder

If the server OOMs during startup (CUDA OOM in logs), lower in this order and retry:

1. `MAX_MODEL_LEN=16384`
2. `GPU_MEMORY_UTILIZATION=0.85`
3. `MAX_NUM_SEQS=4`

If it starts but dies under load, apply the same three dials in the same order.

## Known issues

- **Model runner:** `VLLM_USE_V2_MODEL_RUNNER=1` is explicitly pinned in `docker-compose.yml` — the V2 model runner is the GA default on vLLM 0.29, so the pin is redundant but kept as a one-line rollback surface; `VLLM_WSL2_ENABLE_PIN_MEMORY=1` (also in the compose) enables the V2 UVA path on the target WSL2 platform. If V2 fails to boot or on the first request, set `VLLM_USE_V2_MODEL_RUNNER=0` to fall back to the V1 runner and recreate the container.
- **Do not force `ATTENTION_BACKEND`:** forcing a backend on this hybrid architecture is a validated crash mode (Gemma post-mortem). Leave it empty; vLLM auto-selects.
- **`MAMBA_BACKEND=flashinfer` on sm_120:** the recipe pins flashinfer for the Mamba backend; if it fails to initialize on sm_120, fall back to `MAMBA_BACKEND=triton` in `.env`.
- **`MAX_MODEL_LEN`:** the live `.env` sets `MAX_MODEL_LEN=auto` (the `.env.example` ships `32768`); the effective context is whatever fits the KV pool budget at boot — confirm the resolved value in the boot log.

## Security posture

Same nginx control set as the sibling vLLM stacks. This stack runs **vLLM v0.29.0**; the unauthenticated-endpoint list below applies to v0.29.0.

**What `--api-key` does not protect:** the key only authenticates `/v1`, `/v2` and `/inference`. On v0.29.0 the following endpoints answer **without credentials**: `/invocations` (SageMaker-compatible inference — a full auth bypass), `/generative_scoring`, `/tokenize`, `/detokenize`, `/scale_elastic_ep`, `/is_scaling_elastic_ep`, `/ping`, `/version`, `/metrics`, `/load`.

**Controls (nginx, HTTPS path):**

| Control | Value | Purpose |
|---|---|---|
| Endpoint allowlist | only `/v1/*` and `/health` are proxied, everything else → `404` | blocks every unauthenticated endpoint above; endpoints added by future vLLM releases stay blocked by default |
| Rate limit | 10 req/s per IP, burst 20, then `503` | bounds abuse and queue-flooding |
| Body-size cap | `client_max_body_size 4m` | the full context window fits with margin; bounds abuse |
| TLS | Mozilla Intermediate, HSTS, OCSP stapling | transport |

**Controls (vLLM):** `VLLM_MAX_N_SEQUENCES=16` caps the `n` parameter (vLLM default 16384). Dev-mode endpoints, profilers, gRPC and endpoint plugins are off by default in this entrypoint. Admission control (`--max-num-queued-reqs` / `--max-num-queued-tokens`, defaults `32` / `256K`) caps in-flight requests and queued prefill tokens server-side; overflow returns HTTP `503` from the engine as a backstop under the nginx rate limit.

**Residual risks:**

- Port `1237` stays published on the host in proxy mode (the overlay's `ports: []` is a no-op — Compose merges lists, as documented in the qwen stack) — LAN-only exposure. The unauthenticated endpoints listed above are reachable on 1237 with no nginx in front.
- `TRUST_REMOTE_CODE` defaults to `false`: Nemotron 3.5 loads through vLLM's built-in architecture registry and its NVFP4 weights are consumed natively via `modelopt_fp4`, so no remote-code surface is exposed at load. Set `TRUST_REMOTE_CODE=true` only if the checkpoint starts failing with a custom-code / `auto_map` load error; with `true` the flag becomes a supply-chain trust in the Hugging Face repo, not a runtime API surface.

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
