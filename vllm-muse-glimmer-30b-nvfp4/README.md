# Muse Glimmer 30B NVFP4 — vLLM Server

![NVIDIA](https://img.shields.io/badge/NVIDIA-76B900?style=for-the-badge&logo=nvidia&logoColor=white) ![Red Hat AI](https://img.shields.io/badge/Red%20Hat%20AI-EE0000?style=for-the-badge) ![vLLM](https://img.shields.io/badge/vLLM-4B8BBE?style=for-the-badge&logo=python&logoColor=white) ![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)

Docker Compose deployment for [RedHatAI/Muse-Glimmer-30B-NVFP4](https://huggingface.co/RedHatAI/Muse-Glimmer-30B-NVFP4) on vLLM (OpenAI-compatible API) — a dense 29.6B multimodal model (52 layers, 128K context, Apache 2.0) with a ViT-G/14 vision encoder, NVFP4-quantized weights (group-16, Compressed-Tensors format; vision tower and embeddings stay in BF16).

by Renato Perini (mjordan79)

## How this model answers

Muse Glimmer does **not** use JSON tool calls or `think` tags. Its output is **channel-scoped**:

- `to=self` — internal reasoning channel
- `to=user` — the final answer channel
- ATEM XML blocks for tool calls

The vLLM `muse_glimmer` reasoning and tool-call parsers (native since vLLM 0.27.0) decode these channels and map them onto the standard `reasoning` / `content` / `tool_calls` response fields. Two consequences:

1. **Never run greedy.** The checkpoint is tuned for `temperature 1.0 / top_p 0.95 / top_k 64`. The server loads that generation config (`--generation-config auto`); request payloads should carry the same sampling.
2. **Size `max_tokens` generously.** A tight cap truncates the final `to=user` channel mid-stream: the response ends with `finish_reason: stop` and an **empty `content` field** while the reasoning channel looks fine.

Reasoning strength is controlled **client-side** via a line in the system prompt:

```
Reasoning strength: <low|medium|high|xhigh>
```

There is no `enable_thinking` flag and no `chat_template_kwargs` knob for this model.

## Prerequisites

- **NVIDIA GPU** with CUDA drivers (tested on GeForce RTX 5090 — 32 GB VRAM, `TP_SIZE=1`)
- **NVIDIA Container Toolkit** (natively or on WSL) installed (`nvidia-container-toolkit`)
- **Docker** + **Docker Compose**
- **HuggingFace token** for `RedHatAI/Muse-Glimmer-30B-NVFP4`

### For remote access via HTTPS (optional)

- **DuckDNS account** with a subdomain (e.g., `your-domain.duckdns.org`) — free, up to 5 subdomains
- **Ports 80 and 443** forwarded from your router to the Docker host
- DuckDNS subdomain pointing to your public IP (update at [duckdns.org](https://www.duckdns.org))

## Setup

### 1. Configure the HuggingFace token

Create a `.env` file in the same directory as `docker-compose.yml` (a filled `.env.example` is included):

```env
HF_TOKEN=hf_your_token_here
MODEL_NAME=RedHatAI/Muse-Glimmer-30B-NVFP4
QUANTIZATION=compressed-tensors
HF_CACHE_VOLUME=hf-cache-museglimmer
MAX_MODEL_LEN=131072
```

> The token must **never** be hardcoded in docker-compose or the entrypoint.

For remote HTTPS access, also add:

```env
LETSENCRYPT_DOMAIN=your-domain.duckdns.org
LETSENCRYPT_EMAIL=you@example.com
```

To pin a fixed API key (optional) instead of the auto-generated one, add `VLLM_API_KEY=sk-...` to `.env` (documented in `.env.example`). Keep the value in the gitignored `.env`, never in compose.

> `.env` is listed in `.gitignore` — never commit it.

### 2. Build and start

```bash
docker compose build
docker compose up -d
```

The first run downloads the model (~25 GB) and stores it in the HuggingFace cache mounted at `/root/.cache/huggingface`. Initialization takes ~10 minutes; the healthcheck has a 600 s start period for exactly this reason. Watch the logs:

```bash
docker compose logs -f
```

Expected at the end of init (recipe numbers, 1x RTX 5090): ~28.8 GB used of 32.6 GB available, KV pool of ~179,647 tokens at the full 128K context, no OOM.

### 2b. Remote access via HTTPS (optional)

To expose the API over HTTPS through a DuckDNS domain, start with the proxy overlay:

```bash
docker compose -f docker-compose.yml -f docker-compose.proxy.yml up -d
```

This starts additional containers:
- **muse-docker-gen** — required by acme-companion (watches Docker events, no-op template)
- **muse-nginx** — vanilla reverse proxy on ports 80/443 with SSL hardening; generates a self-signed placeholder on first boot, then symlinks to the Let's Encrypt cert once issued
- **muse-acme** — auto-provisions and renews a free Let's Encrypt certificate for `LETSENCRYPT_DOMAIN`

On the first run, certificate issuance takes ~2 minutes. Check progress:

```bash
docker compose -f docker-compose.yml -f docker-compose.proxy.yml logs -f acme-companion
```

Once the certificate is ready, access the API at `https://<your-domain.duckdns.org>`.

> **Without the proxy overlay**, the API is exposed directly on port `1236` (HTTP). Compose binds this port on `0.0.0.0` by default, so it is reachable from the LAN, not only localhost. It is Bearer-authenticated, but use the TLS proxy for anything non-local.
>
> **Port 80/443 conflict:** if the sibling `vllm-qwen-3.8-27b-nvfp4` deployment runs its proxy overlay at the same time, both stacks claim 80/443 on the same host. Run one proxy at a time. The direct-mode ports do not conflict (Qwen `1235`, Muse `1236`).

### 3. Get your API key

API authentication is **enabled by default** (`ENABLE_API_KEY=true`). On the first run, the entrypoint generates a key and prints it to the logs:

```bash
docker compose logs | grep "Generated API key"
```

Save this value — it will **not** be shown again on subsequent restarts. You can also retrieve it anytime:

```bash
docker exec vllm-museglimmer-server cat /root/.vllm-key/.api_key
```

To disable authentication, edit `docker-compose.yml` and change the pass-through entry `- ENABLE_API_KEY` to `- ENABLE_API_KEY=false` (`.env` only holds the variables in `.env.example`; tuning is done in the compose file).

### 4. Verify

**Direct mode (default):**

```bash
curl http://localhost:1236/v1/models
```

**Proxy mode (with overlay):**

```bash
curl -k https://<your-domain>/v1/models \
  -H "Authorization: Bearer YOUR_API_KEY"
```

You should see the model in the list.

## Usage (OpenAI-compatible API)

```bash
# Direct mode
curl http://localhost:1236/v1/chat/completions \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "RedHatAI/Muse-Glimmer-30B-NVFP4",
    "messages": [
      {"role": "system", "content": "Reasoning strength: medium"},
      {"role": "user", "content": "Hello, who are you?"}
    ],
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 64,
    "max_tokens": 1024
  }'
```

> Use `-k` (insecure) on first boot before Let's Encrypt issues the cert (self-signed placeholder). Remove `-k` once the cert is active.
>
> **Images:** Muse Glimmer is a VLM — you can send `image_url` parts (base64 or URL) in message content alongside text.
>
> **Tool calling:** pass the standard OpenAI `tools` array with `tool_choice: "auto"`. The `muse_glimmer` parser translates the model's ATEM XML blocks into `tool_calls`; the server must be started with `--enable-auto-tool-choice` (default here).

## Configurable Parameters

All parameters are in `docker-compose.yml` under `environment` (values marked *from `.env`* are interpolated from your `.env` file; runtime defaults come from `entrypoint.sh`):

### Model & quantization

| Variable | Default | Description |
|---|---|---|
| `MODEL_NAME` | `RedHatAI/Muse-Glimmer-30B-NVFP4` | HuggingFace model name |
| `QUANTIZATION` | `compressed-tensors` | Quantization backend (the checkpoint's Compressed-Tensors NVFP4 format) |
| `HF_CACHE_VOLUME` | `hf-cache-museglimmer` | Named volume for the HF cache |
| `MAX_MODEL_LEN` | `131072` | Maximum context length (128K) |
| `DTYPE` | `auto` | Data type for model weights |
| `TRUST_REMOTE_CODE` | `true` | Pass `--trust-remote-code` (required by some HF repos) |
| `HF_TOKEN` | *(from `.env`)* | HuggingFace token |

### Performance

| Variable | Default | Description |
|---|---|---|
| `TP_SIZE` | `1` | Tensor parallelism (1 = single GPU) |
| `GPU_MEMORY_UTILIZATION` | `0.92` | Fraction of usable VRAM (0.0–1.0) |
| `MAX_NUM_SEQS` | `1` | Maximum concurrent sequences |
| `MAX_NUM_BATCHED_TOKENS` | `6144` | Maximum tokens per prefill batch |
| `KV_CACHE_DTYPE` | `auto` | KV cache data type (`fp8_e4m3` to halve KV footprint) |
| `ATTENTION_BACKEND` | `flashinfer` | Attention backend |
| `PERFORMANCE_MODE` | `interactivity` | vLLM performance mode |
| `ENABLE_CHUNKED_PREFILL` | `true` | Split long prefills into chunks |
| `ENABLE_PREFIX_CACHING` | `true` | Cache shared prompt prefixes |
| `ENABLE_HYBRID_KV_CACHE_MANAGER` | `true` | Hybrid (CPU+GPU) KV cache manager |
| `ENABLE_SPEC_DECODING` | `false` | DFlash speculative decoding (see Notes) |
| `SPEC_METHOD` | `dflash` | Speculative decoding method |
| `SPEC_MODEL` | `meta-models/Muse-Glimmer-30B-assistant` | DFlash draft head |
| `SPEC_NUM_TOKENS` | `15` | Speculative tokens per step |

### Behavior & features

| Variable | Default | Description |
|---|---|---|
| `REASONING_PARSER` | `muse_glimmer` | Parser that splits the `to=self` channel into `reasoning` (forces `skip_special_tokens=False`) |
| `TOOL_CALL_PARSER` | `muse_glimmer` | Translates ATEM XML blocks into OpenAI `tool_calls` |
| `GENERATION_CONFIG` | `auto` | Load the checkpoint's published generation config (temp 1.0 / top_p 0.95 / top_k 64) — see Notes |
| `ENABLE_AUTO_TOOL_CHOICE` | `true` | Allow `tool_choice: "auto"` |

### API & server

| Variable | Default | Description |
|---|---|---|
| `ENABLE_API_KEY` | `true` | API key authentication (auto-generates on first run) |
| `VLLM_API_KEY` | *(empty → auto-generated)* | Pass-through: set a fixed key in `.env` (gitignored — the one secret allowed there); if empty, the entrypoint generates and persists `sk-<uuid>` |
| `ENABLE_REQUEST_METRICS` | `true` | Per-request metrics (profiling) |
| `DISABLE_LOG_STATS` | `false` | Disable periodic vLLM throughput statistics; requires `ENABLE_REQUEST_METRICS=false` |
| `ENABLE_PROMPT_TOKENS_DETAILS` | `true` | Detailed prompt-token breakdown in usage |
| `PORT` | `8000` | In-container port (host mapping `1236:8000`) |

### Runtime & low-level

| Variable | Default | Description |
|---|---|---|
| `SAFETENSORS_LOAD_STRATEGY` | `prefetch` | Weight loading strategy |
| `VLLM_USE_V2_MODEL_RUNNER` | `0` | **Hardcoded in `docker-compose.yml`** (not a pass-through): the V2 model runner UVA buffer check fails on WSL2, so the V1 runner is forced |
| `EXTRA_ARGS_STR` | `--mm-processor-cache-gb 0` | **Hardcoded in `docker-compose.yml`**: raw flags appended verbatim to `vllm serve`; disables the mm-processor cache (workaround for the "video placeholders" bug on the dev build) |
| `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS` | `1` | Estimate CUDA-graph memory in the profiler (on) |
| `VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR` | `/tmp/flashinfer_autotune_cache` | FlashInfer autotune cache location |
| `NVIDIA_VISIBLE_DEVICES` | `all` | GPU passthrough |
| `NVIDIA_DRIVER_CAPABILITIES` | `compute,utility` | GPU driver capabilities |

### Proxy overlay (Let's Encrypt)

| Variable | Default | Description |
|---|---|---|
| `LETSENCRYPT_DOMAIN` | *(none)* | Domain for Let's Encrypt certificate (e.g., `your-domain.duckdns.org`) |
| `LETSENCRYPT_EMAIL` | *(none)* | Email for Let's Encrypt certificate notifications |

## Notes

- **vLLM image:** the deployment builds on `vllm/vllm-openai:muse-glimmer` (the recipe image; `muse_glimmer` parsers are native since vLLM 0.27.0). If that tag is not pullable, build with the fallback base without editing the Dockerfile:
  ```bash
  docker compose build --build-arg VLLM_BASE=vllm/vllm-openai:v0.27.1
  ```
- **`--generation-config auto`:** if the image rejects the flag on startup, remove the `GENERATION_CONFIG` block from `entrypoint.sh` and rebuild — sampling then comes from the request payloads. Do **not** run the model greedy either way.
- **Coexistence with the Qwen deployment:** the two stacks run side by side — separate compose project names, containers (`vllm-museglimmer-server` vs `vllm-qwen-server`), host ports (`1236` vs `1235`), HF cache and API-key volumes.
- **Speculative decoding (DFlash):** off by default — the 5.1 GB draft head (`meta-models/Muse-Glimmer-30B-assistant`, 15 tokens/step) OOMs on a single RTX 5090 (~400 MiB headroom). On a 2x setup with `TP_SIZE=2` the recipe measured ~240 tok/s decode (~3.5x). Enable via `ENABLE_SPEC_DECODING=true` only in that configuration.
- **VRAM:** with `GPU_MEMORY_UTILIZATION=0.92` on 32 GB, consumption is ~28.8 GB (recipe, model + full 128K KV pool at ~179,647 tokens).
- **HuggingFace Cache:** the cache is mounted at `/root/.cache/huggingface` and persists across container restarts.
- **Port:** the API is exposed on host port `1236` (mapped from internal port 8000), bound to `0.0.0.0` by default — reachable from the LAN, not only localhost. It is Bearer-authenticated, but prefer the TLS proxy for non-local access.
- **Reverse Proxy:** two modes via overlay:
  - **Direct** (default): `docker compose up -d` → API on port `1236` (HTTP, LAN-reachable, Bearer-authenticated)
  - **Proxy**: `docker compose -f docker-compose.yml -f docker-compose.proxy.yml up -d` → API on `https://<domain>` (HTTPS, remote)
  - The proxy overlay adds 3 containers: `muse-docker-gen`, `muse-nginx`, `muse-acme`. Nginx generates a self-signed placeholder on first boot and symlinks to the Let's Encrypt cert once issued. The domain (`LETSENCRYPT_DOMAIN`) is resolved from `.env` at runtime — never hardcoded in config files.
  - When proxy mode is active, ports 80/443 are exposed and port 1236 is automatically disabled — all traffic routes through nginx.
- **DuckDNS:** register at [duckdns.org](https://www.duckdns.org), create a subdomain, and ensure it resolves to your public IP. Ports 80 and 443 must be forwarded from your router to the Docker host for Let's Encrypt validation. Set `LETSENCRYPT_DOMAIN` in `.env` to your DuckDNS subdomain.
- **Let's Encrypt:** no separate registration required. The `muse-acme` container handles certificate issuance and renewal automatically. Provide `LETSENCRYPT_EMAIL` in `.env` for renewal notifications.

## Useful Commands

```bash
# Start (direct mode — HTTP on port 1236, bound to 0.0.0.0)
docker compose up -d

# Start with HTTPS proxy (remote access via DuckDNS)
docker compose -f docker-compose.yml -f docker-compose.proxy.yml up -d

# Real-time logs
docker compose logs -f

# Proxy logs (acme-companion certificate status)
docker compose -f docker-compose.yml -f docker-compose.proxy.yml logs -f acme-companion

# Stop (direct mode)
docker compose down

# Stop (proxy mode)
docker compose -f docker-compose.yml -f docker-compose.proxy.yml down

# Stop and remove volumes (re-downloads model on next start)
docker compose down -v

# Rebuild after entrypoint/Dockerfile changes
docker compose build --no-cache
```
