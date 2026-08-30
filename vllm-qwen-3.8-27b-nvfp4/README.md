# Qwen 27B NVFP4 — vLLM Server

![NVIDIA](https://img.shields.io/badge/NVIDIA-76B900?style=for-the-badge&logo=nvidia&logoColor=white) ![Alibaba Cloud](https://img.shields.io/badge/Alibaba%20Cloud-FF6A00?style=for-the-badge&logo=alibabacloud&logoColor=white) ![vLLM](https://img.shields.io/badge/vLLM-4B8BBE?style=for-the-badge&logo=python&logoColor=white) ![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)

Docker Compose deployment for Qwen 27B NVFP4 on vLLM (OpenAI-compatible API). The same deployment serves two HuggingFace checkpoints, selected in `.env`:

- **Unsloth** — `unsloth/Qwen3.8-27B-NVFP4`, NVFP4 via Compressed-Tensors (default)
- **NVIDIA** — `nvidia/Qwen3.6-27B-NVFP4`, NVFP4 via ModelOpt

by Renato Perini (mjordan79)

## Prerequisites

- **NVIDIA GPU** with CUDA drivers (tested on Geforce RTX 5090 — 32 GB VRAM)
- **NVIDIA Container Toolkit** (natively or on WSL) installed (`nvidia-container-toolkit`)
- **Docker** + **Docker Compose**
- **HuggingFace token** for the selected checkpoint (`.env` → `MODEL_NAME`)

### For remote access via HTTPS (optional)

- **DuckDNS account** with a subdomain (e.g., `your-domain.duckdns.org`) — free, up to 5 subdomains
- **Ports 80 and 443** forwarded from your router to the Docker host
- DuckDNS subdomain pointing to your public IP (update at [duckdns.org](https://www.duckdns.org))

## Setup

### 1. Configure the HuggingFace token

Create a `.env` file in the same directory as `docker-compose.yml`:

```env
HF_TOKEN=hf_your_token_here

# Variant: Unsloth — NVFP4 via Compressed-Tensors (default)
MODEL_NAME=unsloth/Qwen3.8-27B-NVFP4
QUANTIZATION=compressed-tensors
HF_CACHE_VOLUME=hf-cache-unsloth
MAX_MODEL_LEN=116800

# Variant: NVIDIA — NVFP4 via ModelOpt (comment the block above to use this)
#MODEL_NAME=nvidia/Qwen3.6-27B-NVFP4
#QUANTIZATION=modelopt
#HF_CACHE_VOLUME=hf-cache-nvidia
#MAX_MODEL_LEN=131072
```

> The token must **never** be hardcoded in docker-compose or the entrypoint.
> Each variant uses its own HF cache volume (`HF_CACHE_VOLUME`), so switching variants does not clobber the other's cache.

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

The first run downloads the model (~15 GB) and stores it in the HuggingFace cache mounted at `/root/.cache/huggingface`.

### 2b. Remote access via HTTPS (optional)

To expose the API over HTTPS through a DuckDNS domain, start with the proxy overlay:

```bash
docker compose -f docker-compose.yml -f docker-compose.proxy.yml up -d
```

This starts additional containers:
- **docker-gen** — required by acme-companion (watches Docker events, no-op template)
- **nginx** — vanilla reverse proxy on ports 80/443 with SSL hardening; generates a self-signed placeholder on first boot, then symlinks to the Let's Encrypt cert once issued
- **acme-companion** — auto-provisions and renews a free Let's Encrypt certificate for `LETSENCRYPT_DOMAIN`

On the first run, certificate issuance takes ~2 minutes. Check progress:

```bash
docker compose -f docker-compose.yml -f docker-compose.proxy.yml logs -f acme-companion
```

Once the certificate is ready, access the API at `https://<your-domain.duckdns.org>`.

> **Without the proxy overlay**, the API is exposed directly on port `1235` (HTTP). Compose binds this port on `0.0.0.0` by default, so it is reachable from the LAN, not only localhost. It is Bearer-authenticated, but use the TLS proxy for anything non-local.
>
> **Port 80/443 conflict:** if any of the other deployments in this repo runs its proxy overlay at the same time, both stacks claim 80/443 on the same host. Run one proxy at a time. The direct-mode ports do not conflict (Qwen `1235`, Muse `1236`, Nemotron `1237`, Qwen-SGLang `1238`).

### 3. Get your API key

API authentication is **enabled by default** (`ENABLE_API_KEY=true`). On the first run, the entrypoint generates a key and prints it to the logs:

```bash
docker compose logs | grep "Generated API key"
```

Save this value — it will **not** be shown again on subsequent restarts. You can also retrieve it anytime:

```bash
docker exec vllm-qwen-server cat /root/.vllm-key/.api_key
```

To disable authentication, edit `docker-compose.yml` and change the pass-through entry `- ENABLE_API_KEY` to `- ENABLE_API_KEY=false` (`.env` only holds the variables in `.env.example`; tuning is done in the compose file).

### 4. Verify

**Direct mode (default):**

```bash
curl http://localhost:1235/v1/models \
  -H "Authorization: Bearer YOUR_API_KEY"
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
curl http://localhost:1235/v1/chat/completions \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "unsloth/Qwen3.8-27B-NVFP4",
    "messages": [
      {"role": "user", "content": "Hello, who are you?"}
    ]
  }'

# Proxy mode (HTTPS)
curl -k https://<your-domain>/v1/chat/completions \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "unsloth/Qwen3.8-27B-NVFP4",
    "messages": [
      {"role": "user", "content": "Hello, who are you?"}
    ]
  }'
```

> Use `-k` (insecure) on first boot before Let's Encrypt issues the cert (self-signed placeholder). Remove `-k` once the cert is active.

## Configurable Parameters

All parameters are in `docker-compose.yml` under `environment` (values marked *from `.env`* are interpolated from your `.env` file):

### Model & quantization

| Variable | Default | Description |
|---|---|---|
| `MODEL_NAME` | `unsloth/Qwen3.8-27B-NVFP4` | HuggingFace model name — set per variant in `.env` |
| `QUANTIZATION` | `compressed-tensors` | Quantization backend — `modelopt` for the NVIDIA variant |
| `HF_CACHE_VOLUME` | `hf-cache-unsloth` | Named volume for the HF cache — one per variant |
| `MAX_MODEL_LEN` | `116800` | Maximum context length — set per variant in `.env` (NVIDIA: `131072`, Unsloth: `116800`) |
| `DTYPE` | `auto` | Data type for model weights |
| `TRUST_REMOTE_CODE` | `true` | Pass `--trust-remote-code` (required by some HF repos) |
| `SKIP_MM_PROFILING` | `true` | Skip multimodal profiling at startup |
| `HF_TOKEN` | *(from `.env`)* | HuggingFace token |

### Performance

| Variable | Default | Description |
|---|---|---|
| `TP_SIZE` | `1` | Tensor parallelism (1 = single GPU) |
| `GPU_MEMORY_UTILIZATION` | `0.94` | Fraction of usable VRAM (0.0–1.0) |
| `MAX_NUM_SEQS` | `1` | Maximum concurrent sequences |
| `MAX_NUM_BATCHED_TOKENS` | `6144` | Maximum tokens per prefill batch |
| `KV_CACHE_DTYPE` | `fp8_e4m3` | KV cache data type |
| `ATTENTION_BACKEND` | `flashinfer` | Attention backend |
| `PERFORMANCE_MODE` | `interactivity` | vLLM performance mode |
| `ENABLE_CHUNKED_PREFILL` | `true` | Split long prefills into chunks |
| `ENABLE_PREFIX_CACHING` | `true` | Cache shared prompt prefixes |
| `ENABLE_HYBRID_KV_CACHE_MANAGER` | `true` | Hybrid (CPU+GPU) KV cache manager |
| `ENABLE_MTP` | `true` | Multi-Token Prediction (speculative decoding) |
| `MTP_NUM_SPECULATIVE_TOKENS` | `2` | Speculative tokens per step |

### Behavior & features

| Variable | Default | Description |
|---|---|---|
| `LANGUAGE_MODEL_ONLY` | `true` | Entrypoint default: when unset (or not `false`), vLLM starts with `--language-model-only` (skips the multimodal stack — this model is text-only, so this is the lighter startup). To keep the full server path, set `LANGUAGE_MODEL_ONLY=false` in `docker-compose.yml` |
| `REASONING_PARSER` | `qwen3` | Parser that splits reasoning content from the response |
| `DEFAULT_ENABLE_THINKING` | `true` | Server-side default for `enable_thinking` (per-request `chat_template_kwargs` overrides it) |
| `DEFAULT_PRESERVE_THINKING` | `true` | Keep historical assistant thinking in multi-turn context; `false` strips it to save context (inert unless the client echoes reasoning back) |
| `TOOL_CALL_PARSER` | `qwen3_coder` | Tool-call response parser |
| `ENABLE_AUTO_TOOL_CHOICE` | `true` | Allow `tool_choice: "auto"` |

### API & server

| Variable | Default | Description |
|---|---|---|
| `ENABLE_API_KEY` | `true` | API key authentication (auto-generates on first run) |
| `VLLM_MAX_N_SEQUENCES` | `16` | Cap on the `n` parameter per `/v1` request (vLLM default 16384) |
| `VLLM_API_KEY` | *(empty → auto-generated)* | Pass-through: set a fixed key in `.env` (gitignored — the one secret allowed there); if empty, the entrypoint generates and persists `sk-<uuid>` |
| `ENABLE_REQUEST_METRICS` | `true` | Per-request metrics (profiling) |
| `DISABLE_LOG_STATS` | `false` | Disable periodic vLLM throughput statistics; requires `ENABLE_REQUEST_METRICS=false` |
| `ENABLE_PROMPT_TOKENS_DETAILS` | `true` | Detailed prompt-token breakdown in usage |
| `PORT` | `8000` | In-container port (host mapping `1235:8000`) |

### Runtime & low-level

| Variable | Default | Description |
|---|---|---|
| `SAFETENSORS_LOAD_STRATEGY` | `prefetch` | Weight loading strategy |
| `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS` | `1` | Estimate CUDA-graph memory in the profiler (on) |
| `VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR` | `/tmp/flashinfer_autotune_cache` | FlashInfer autotune cache location |
| `NVIDIA_VISIBLE_DEVICES` | `all` | GPU passthrough |
| `NVIDIA_DRIVER_CAPABILITIES` | `compute,utility` | GPU driver capabilities |

### Proxy overlay (Let's Encrypt)

| Variable | Default | Description |
|---|---|---|
| `LETSENCRYPT_DOMAIN` | *(none)* | Domain for Let's Encrypt certificate (e.g., `my-domain.duckdns.org`) |
| `LETSENCRYPT_EMAIL` | *(none)* | Email for Let's Encrypt certificate notifications |

## Notes

- **Variants:** `unsloth/Qwen3.8-27B-NVFP4` (Compressed-Tensors, default) and `nvidia/Qwen3.6-27B-NVFP4` (ModelOpt). To switch, edit the `MODEL_NAME` / `QUANTIZATION` / `HF_CACHE_VOLUME` / `MAX_MODEL_LEN` block in `.env` and run `docker compose up -d`. Each variant has its own HF cache volume, so the first run after a switch downloads that variant's weights.
- **API Key:** enabled by default (`ENABLE_API_KEY=true`). An `sk-<uuid>` is auto-generated on first run and saved to the `vllm-keys` volume at `/root/.vllm-key/.api_key`. Retrieve it with `docker exec vllm-qwen-server cat /root/.vllm-key/.api_key`. To use a fixed key, set `VLLM_API_KEY` in `.env` (gitignored; compose passes it through and the entrypoint uses it instead of generating one). To disable, change `- ENABLE_API_KEY` to `- ENABLE_API_KEY=false` in `docker-compose.yml`.
- **MTP (Multi-Token Prediction):** the NVFP4 checkpoints include up to 3 MTP layers. Default is 2 speculative tokens for better compute/accuracy balance (per-position acceptance rate ~60-70% at position 1-2 vs ~30-50% at position 3). If you get missing MTP weights errors on first startup, set `ENABLE_MTP=false` and restart.
- **VRAM:** with `GPU_MEMORY_UTILIZATION=0.94` on 32 GB, consumption is ~30.1 GB.
- **HuggingFace Cache:** the cache is mounted at `/root/.cache/huggingface` and persists across container restarts.
- **Port:** the API is exposed on host port `1235` (mapped from internal port 8000), bound to `0.0.0.0` by default — reachable from the LAN, not only localhost. It is Bearer-authenticated, but prefer the TLS proxy for non-local access.
- **Reverse Proxy:** two modes via overlay:
  - **Direct** (default): `docker compose up -d` → API on port `1235` (HTTP, LAN-reachable, Bearer-authenticated)
  - **Proxy**: `docker compose -f docker-compose.yml -f docker-compose.proxy.yml up -d` → API on `https://<domain>` (HTTPS, remote)
  - The proxy overlay adds 3 containers: `docker-gen` (required by acme-companion), `nginx` (reverse proxy with SSL), and `acme-companion` (Let's Encrypt cert management). Nginx generates a self-signed placeholder on first boot and symlinks to the Let's Encrypt cert once issued. The domain (`LETSENCRYPT_DOMAIN`) is resolved from `.env` at runtime — never hardcoded in config files.
  - **Known limitation:** the overlay's `ports: []` does NOT remove the base file's `1235:8000` publish (Compose merges lists; an empty list is a no-op). Verified on the running container: port `1235` stays published on the host even in proxy mode. nginx only guards the public 80/443 path — LAN hosts can still reach vLLM directly on 1235, bypassing the nginx allowlist (see Security posture).
- **DuckDNS:** register at [duckdns.org](https://www.duckdns.org), create a subdomain, and ensure it resolves to your public IP. Ports 80 and 443 must be forwarded from your router to the Docker host for Let's Encrypt validation. Set `LETSENCRYPT_DOMAIN` in `.env` to your DuckDNS subdomain.
- **Let's Encrypt:** no separate registration required. The `acme-companion` container handles certificate issuance and renewal automatically. Provide `LETSENCRYPT_EMAIL` in `.env` for renewal notifications.

## Security posture

Hardened against the [vLLM security docs](https://docs.vllm.ai/en/latest/usage/security/); endpoint claims verified against the running server (vLLM v0.28.0).

**What `--api-key` does not protect:** the key only authenticates `/v1`, `/v2` and `/inference`. On v0.28.0 the following endpoints answer **without credentials** (probed live): `/invocations` (SageMaker-compatible inference — a full auth bypass), `/generative_scoring`, `/tokenize`, `/detokenize`, `/scale_elastic_ep`, `/is_scaling_elastic_ep`, `/ping`, `/version`, `/metrics`, `/load`. `/pause`, `/abort_requests`, the dev-mode and weight-update endpoints do not exist in this version (and dev mode is never enabled).

**Controls (nginx, HTTPS path):**

| Control | Value | Purpose |
|---|---|---|
| Endpoint allowlist | only `/v1/*` and `/health` are proxied, everything else → `404` | blocks every unauthenticated endpoint above; endpoints added by future vLLM releases stay blocked by default |
| Rate limit | 10 req/s per IP, burst 20, then `503` | bounds abuse and queue-flooding |
| Body-size cap | `client_max_body_size 4m` | the full 116k-token context fits with margin; bounds abuse |
| TLS | Mozilla Intermediate, HSTS, OCSP stapling | transport |

**Controls (vLLM):** `VLLM_MAX_N_SEQUENCES=16` caps the `n` parameter (vLLM default 16384). Dev-mode endpoints, the tokenizer-info endpoint, profilers, gRPC, LoRA runtime loading and endpoint plugins are all off by default in this entrypoint.

**Residual risks:**

- Port `1235` stays published on the host in proxy mode (see limitation above) — LAN-only exposure. `/v1/*` is Bearer-authenticated, but the unauthenticated endpoints listed above are reachable on 1235 with no nginx in front.
- `TRUST_REMOTE_CODE=true` is on by default — a supply-chain trust in the Hugging Face repo, not a runtime API surface. Set `TRUST_REMOTE_CODE=false` only after verifying the checkpoint loads without it.

## Useful Commands

```bash
# Start (direct mode — HTTP on port 1235, bound to 0.0.0.0)
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
