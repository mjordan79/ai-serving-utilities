# Qwen 27B NVFP4 — SGLang Server

![NVIDIA](https://img.shields.io/badge/NVIDIA-76B900?style=for-the-badge&logo=nvidia&logoColor=white) ![Alibaba Cloud](https://img.shields.io/badge/Alibaba%20Cloud-FF6A00?style=for-the-badge&logo=alibabacloud&logoColor=white) ![SGLang](https://img.shields.io/badge/SGLang-4B8BBE?style=for-the-badge&logo=python&logoColor=white) ![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)

Docker Compose deployment for Qwen 27B NVFP4 on SGLang (OpenAI-compatible API).

- **Model** — `RadixArk/Qwen3.8-27B-NVFP4` (NVFP4, Mamba/hybrid attention)
- **Image** — `lmsysorg/sglang:dev-cu13-qwen38-27b-dflash2` (pinned dev tag, see *Notes*)
- **Speculative decoding** — EAGLE draft model, **disabled by default**; enable via `ENABLE_MTP=true` in `docker-compose.yml`

by Renato Perini (mjordan79)

## Prerequisites

- **NVIDIA GPU** with CUDA drivers (tested on Geforce RTX 5090 — 32 GB VRAM)
- **NVIDIA Container Toolkit** (natively or on WSL) installed (`nvidia-container-toolkit`)
- **Docker** + **Docker Compose**
- **HuggingFace token** for `RadixArk/Qwen3.8-27B-NVFP4` (`.env` → `MODEL_NAME`)

### For remote access via HTTPS (optional)

- **DuckDNS account** with a subdomain (e.g., `your-domain.duckdns.org`) — free, up to 5 subdomains
- **Ports 80 and 443** forwarded from your router to the Docker host
- DuckDNS subdomain pointing to your public IP (update at [duckdns.org](https://www.duckdns.org))

## Setup

### 1. Configure the HuggingFace token

Create a `.env` file in the same directory as `docker-compose.yml`:

```env
HF_TOKEN=hf_your_token_here

# Model (SGLang)
MODEL_NAME=RadixArk/Qwen3.8-27B-NVFP4
HF_CACHE_VOLUME=hf-cache-radixark

# Reverse proxy (optional)
LETSENCRYPT_DOMAIN=your-domain.duckdns.org
LETSENCRYPT_EMAIL=you@example.com
```

> The token must **never** be hardcoded in docker-compose or the entrypoint.
> The cache lives on the named volume `hf-cache-radixark` so it persists across container restarts.

For remote HTTPS access, the `LETSENCRYPT_DOMAIN` / `LETSENCRYPT_EMAIL` lines are already in the block above.

To pin a fixed API key (optional) instead of the auto-generated one, add `VLLM_API_KEY=sk-...` to `.env` (documented in `.env.example`). Keep the value in the gitignored `.env`, never in compose.

> `.env` is listed in `.gitignore` — never commit it.
> The variable name is `VLLM_API_KEY` (not `SGLANG_API_KEY`) for **exact parity** with the sibling vLLM deployments; the `.env` convention is shared across the whole project.

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

> **Without the proxy overlay**, the API is exposed directly on port `1238` (HTTP). Compose binds this port on `0.0.0.0` by default, so it is reachable from the LAN, not only localhost. It is Bearer-authenticated, but use the TLS proxy for anything non-local.
>
> **Port 80/443 conflict:** every proxy overlay in this repo binds the same host ports 80/443 — **only one proxy can run at a time** (this proxy and the `vllm-qwen-3.8-27b-nvfp4` proxy additionally share the same domain). The direct-mode ports do not conflict (Qwen-vLLM `1235`, Muse `1236`, Nemotron `1237`, Qwen-SGLang `1238`).

### 3. Get your API key

API authentication is **enabled by default** (`ENABLE_API_KEY=true`). On the first run, the entrypoint generates a key and prints it to the logs:

```bash
docker compose logs | grep "Generated API key"
```

Save this value — it will **not** be shown again on subsequent restarts. You can also retrieve it anytime:

```bash
docker exec sglang-qwen-server cat /root/.sglang-key/.api_key
```

To disable authentication, edit `docker-compose.yml` and change the pass-through entry `- ENABLE_API_KEY` to `- ENABLE_API_KEY=false` (`.env` only holds the variables in `.env.example`; tuning is done in the compose file).

### 4. Verify

**Direct mode (default):**

```bash
curl http://localhost:1238/v1/models \
  -H "Authorization: Bearer YOUR_API_KEY"
```

**Proxy mode (with overlay):**

```bash
curl -k https://<your-domain.duckdns.org>/v1/models \
  -H "Authorization: Bearer YOUR_API_KEY"
```

You should see the model in the list.

## Usage (OpenAI-compatible API)

```bash
# Direct mode
curl http://localhost:1238/v1/chat/completions \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "RadixArk/Qwen3.8-27B-NVFP4",
    "messages": [
      {"role": "user", "content": "Hello, who are you?"}
    ]
  }'

# Proxy mode (HTTPS)
curl -k https://<your-domain.duckdns.org>/v1/chat/completions \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "RadixArk/Qwen3.8-27B-NVFP4",
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
| `MODEL_NAME` | `RadixArk/Qwen3.8-27B-NVFP4` | HuggingFace model name — set in `.env` |
| `HF_CACHE_VOLUME` | `hf-cache-radixark` | Named volume for the HF cache |
| `KV_CACHE_DTYPE` | `fp8_e4m3` | KV cache data type |
| `MEM_FRACTION_STATIC` | `0.94` | Fraction of usable VRAM (0.0–1.0) |
| `TRUST_REMOTE_CODE` | `true` | Pass `--trust-remote-code` (required by some HF repos) |
| `HF_TOKEN` | *(from `.env`)* | HuggingFace token |

### Performance

| Variable | Default | Description |
|---|---|---|
| `ATTENTION_BACKEND` | `flashinfer` | Attention backend |
| `MAX_RUNNING_REQUESTS` | `1` | Maximum concurrent requests |
| `CUDA_GRAPH_MAX_BS_DECODE` | `1` | Max batch size for CUDA graph decode |

### Mamba (hybrid attention state)

| Variable | Default | Description |
|---|---|---|
| `MAMBA_FULL_MEMORY_RATIO` | `4.59` | Mamba full-memory ratio (model-specific tuning) |
| `MAMBA_SSM_DTYPE` | `bfloat16` | Mamba SSM data type |
| `MAMBA_RADIX_CACHE_STRATEGY` | `extra_buffer` | Mamba radix cache strategy |

### Speculative decoding (EAGLE)

| Variable | Default | Description |
|---|---|---|
| `ENABLE_MTP` | `false` | **Toggle for speculative decoding.** Set to `true` in `docker-compose.yml` to enable the EAGLE draft flags. Default is off (no speculative decoding). |
| `SPECULATIVE_NUM_STEPS` | `3` | EAGLE spec-decode steps (only used when `ENABLE_MTP=true`) |
| `SPECULATIVE_EAGLE_TOPK` | `1` | EAGLE top-k (only used when `ENABLE_MTP=true`) |
| `SPECULATIVE_NUM_DRAFT_TOKENS` | `4` | EAGLE draft tokens (only used when `ENABLE_MTP=true`) |

> When `ENABLE_MTP=true`, the entrypoint passes `--speculative-algorithm EAGLE`, `--speculative-num-steps`, `--speculative-eagle-topk`, `--speculative-num-draft-tokens`, and `--enable-linear-replayssm-spec` to SGLang. When `false` (default), none of these flags are passed.

### Behavior & features

| Variable | Default | Description |
|---|---|---|
| `REASONING_PARSER` | `qwen3` | Parser that splits reasoning content from the response |
| `TOOL_CALL_PARSER` | `qwen3_coder` | Tool-call response parser |
| `ALLOW_AUTO_TRUNCATE` | `true` | Pass `--allow-auto-truncate` (auto-truncate over-long prompts) |

### API & server

| Variable | Default | Description |
|---|---|---|
| `ENABLE_API_KEY` | `true` | API key authentication (auto-generates on first run) |
| `VLLM_API_KEY` | *(empty → auto-generated)* | Pass-through: set a fixed key in `.env` (gitignored — the one secret allowed there); if empty, the entrypoint generates and persists `sk-<uuid>` |
| `PORT` | `30000` | In-container port (host mapping `1238:30000`) |

### Runtime & low-level

| Variable | Default | Description |
|---|---|---|
| `NVIDIA_VISIBLE_DEVICES` | `all` | GPU passthrough |
| `NVIDIA_DRIVER_CAPABILITIES` | `compute,utility` | GPU driver capabilities |

### Proxy overlay (Let's Encrypt)

| Variable | Default | Description |
|---|---|---|
| `LETSENCRYPT_DOMAIN` | *(none)* | Domain for Let's Encrypt certificate (e.g., `your-domain.duckdns.org`) |
| `LETSENCRYPT_EMAIL` | *(none)* | Email for Let's Encrypt certificate notifications |

## Security posture

The nginx control set is identical to the sibling vLLM stacks (allowlist, rate limit, body cap, TLS). The engine is SGLang, not vLLM:

**Why the allowlist is the main defense:** SGLang exposes a wider native surface than vLLM (`/generate`, `/get_model_info`, `/server_info`, `/v1/models`, `/metrics`, `/abort_request`, ...) and its `--api-key` key coverage is version-dependent. The allowlist keeps every non-`/v1` path blocked regardless of what the key covers — the clients use only the OpenAI-compatible `/v1/*` surface. **Probe at first activation:** with the server up, `GET /v1/models` and one non-`/v1` path (e.g. `/get_model_info`) without a key (expect 401/404) and confirm no client depends on a non-`/v1` path; if one does, extend the allowlist deliberately.

**Controls (nginx, HTTPS path):**

| Control | Value | Purpose |
|---|---|---|
| Endpoint allowlist | only `/v1/*` and `/health` are proxied, everything else → `404` | blocks the entire non-OpenAI native surface |
| Rate limit | 10 req/s per IP, burst 20, then `503` | bounds abuse and queue-flooding |
| Body-size cap | `client_max_body_size 4m` | the full context window fits with margin; bounds abuse |
| TLS | Mozilla Intermediate, HSTS, OCSP stapling | transport |

**Controls (engine):** no `VLLM_MAX_N_SEQUENCES` equivalent in this stack — the engine is SGLang. Concurrency is bounded by `--max-running-requests` (`MAX_RUNNING_REQUESTS`).

**Residual risks:**

- Port `1238` stays published on the host in proxy mode (see known limitation above) — LAN-only exposure; the non-allowlisted native surface is reachable on 1238 with no nginx in front.
- `LETSENCRYPT_DOMAIN` is shared with the qwen stack — only one of the two can hold the public 80/443 proxy at a time (see Notes).
- `TRUST_REMOTE_CODE=true` is on by default — a supply-chain trust in the Hugging Face repo, not a runtime API surface.

## Notes

- **Image (dev tag):** `lmsysorg/sglang:dev-cu13-qwen38-27b-dflash2` is a **non-versioned dev tag** from the SGLang project. It may change or disappear without notice. Before the first `docker compose build`, verify it exists on Docker Hub (`docker pull lmsysorg/sglang:dev-cu13-qwen38-27b-dflash2`). If a stable SGLang release supports this model + Mamba/EAGLE combo, pin that stable tag instead.
- **`--enable-linear-replayssm-spec`:** this is an exotic flag from the reference command. If the image does not support it, the server will fail at boot with an unknown-flag error — remove it from `SPECULATIVE_ARGS` in `entrypoint.sh` and re-run. (Only relevant when `ENABLE_MTP=true`.)
- **`--mamba-full-memory-ratio 4.59`:** model-specific tuning value. If you override `MODEL_NAME` with a different checkpoint, re-tune this value.
- **API Key:** enabled by default (`ENABLE_API_KEY=true`). An `sk-<uuid>` is auto-generated on first run and saved to the `sglang-keys` volume at `/root/.sglang-key/.api_key`. Retrieve it with `docker exec sglang-qwen-server cat /root/.sglang-key/.api_key`. To use a fixed key, set `VLLM_API_KEY` in `.env` (gitignored; compose passes it through and the entrypoint uses it instead of generating one). To disable, change `- ENABLE_API_KEY` to `- ENABLE_API_KEY=false` in `docker-compose.yml`.
- **MTP (EAGLE speculative decoding):** **disabled by default** (`ENABLE_MTP=false` in the entrypoint). Set `ENABLE_MTP=true` in `docker-compose.yml` to enable the EAGLE draft flags (3 steps, top-k 1, 4 draft tokens, linear-replay SSM spec). The reference command in `docker.txt` uses these exact values.
- **VRAM:** with `MEM_FRACTION_STATIC=0.94` on 32 GB, consumption is ~30 GB.
- **HuggingFace Cache:** the cache is mounted at `/root/.cache/huggingface` and persists across container restarts (named volume `hf-cache-radixark`).
- **Port:** the API is exposed on host port `1238` (mapped from internal port 30000), bound to `0.0.0.0` by default — reachable from the LAN, not only localhost. It is Bearer-authenticated, but prefer the TLS proxy for non-local access.
- **Reverse Proxy:** two modes via overlay:
  - **Direct** (default): `docker compose up -d` → API on port `1238` (HTTP, LAN-reachable, Bearer-authenticated)
  - **Proxy**: `docker compose -f docker-compose.yml -f docker-compose.proxy.yml up -d` → API on `https://<your-domain.duckdns.org>` (HTTPS, remote)
  - The proxy overlay adds 3 containers: `docker-gen` (required by acme-companion), `nginx` (reverse proxy with SSL), and `acme-companion` (Let's Encrypt cert management). Nginx generates a self-signed placeholder on first boot and symlinks to the Let's Encrypt cert once issued. The domain (`LETSENCRYPT_DOMAIN`) is resolved from `.env` at runtime — never hardcoded in config files.
  - **Known limitation:** the overlay's `ports: []` does NOT remove the base file's `1238:30000` publish (Compose merges lists; an empty list is a no-op). Port `1238` stays published on the host even in proxy mode — LAN hosts can reach SGLang directly, bypassing the nginx allowlist (see Security posture).
  - **80/443 exclusivity:** every proxy overlay in this repo binds the same host ports 80/443 — run **one** proxy at a time. This proxy and the `vllm-qwen-3.8-27b-nvfp4` proxy additionally share the same `LETSENCRYPT_DOMAIN` (from each `.env`).
- **DuckDNS:** register at [duckdns.org](https://www.duckdns.org), create a subdomain, and ensure it resolves to your public IP. Ports 80 and 443 must be forwarded from your router to the Docker host for Let's Encrypt validation. Set `LETSENCRYPT_DOMAIN` in `.env` to your DuckDNS subdomain.
- **Let's Encrypt:** no separate registration required. The `acme-companion` container handles certificate issuance and renewal automatically. Provide `LETSENCRYPT_EMAIL` in `.env` for renewal notifications.

## Useful Commands

```bash
# Start (direct mode — HTTP on port 1238, bound to 0.0.0.0)
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
