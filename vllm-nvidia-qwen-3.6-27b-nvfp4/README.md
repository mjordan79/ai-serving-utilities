# NVIDIA Qwen3.6-27B-NVFP4 — vLLM Server

![NVIDIA](https://img.shields.io/badge/NVIDIA-76B900?style=for-the-badge&logo=nvidia&logoColor=white) ![Alibaba Cloud](https://img.shields.io/badge/Alibaba%20Cloud-FF6A00?style=for-the-badge&logo=alibabacloud&logoColor=white) ![vLLM](https://img.shields.io/badge/vLLM-4B8BBE?style=for-the-badge&logo=python&logoColor=white) ![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)

Docker Compose deployment for `nvidia/Qwen3.6-27B-NVFP4` on vLLM (OpenAI-compatible API).

by Renato Perini (mjordan79)

## Prerequisites

- **NVIDIA GPU** with CUDA drivers (tested on Geforce RTX 5090 — 32 GB VRAM)
- **NVIDIA Container Toolkit** (natively or on WSL) installed (`nvidia-container-toolkit`)
- **Docker** + **Docker Compose**
- **HuggingFace token** with access to `nvidia/Qwen3.6-27B-NVFP4`

### For remote access via HTTPS (optional)

- **DuckDNS account** with a subdomain (e.g., `my-model.duckdns.org`) — free, up to 5 subdomains
- **Ports 80 and 443** forwarded from your router to the Docker host
- DuckDNS subdomain pointing to your public IP (update at [duckdns.org](https://www.duckdns.org))

## Setup

### 1. Configure the HuggingFace token

Create a `.env` file in the same directory as `docker-compose.yml`:

```env
HF_TOKEN=hf_your_token_here
```

> The token must **never** be hardcoded in docker-compose or the entrypoint.

For remote HTTPS access, also add:

```env
LETSENCRYPT_DOMAIN=my-domain.duckdns.org
LETSENCRYPT_EMAIL=you@example.com
```

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

> **Without the proxy overlay**, the API is exposed directly on `localhost:1235` (HTTP, local only).

### 3. Get your API key

API authentication is **enabled by default** (`ENABLE_API_KEY=true`). On the first run, the entrypoint generates a key and prints it to the logs:

```bash
docker compose logs | grep "Generated API key"
```

Save this value — it will **not** be shown again on subsequent restarts. You can also retrieve it anytime:

```bash
docker exec vllm-server cat /root/.vllm-key/.api_key
```

To disable authentication, set `ENABLE_API_KEY=false` in `docker-compose.yml`.

### 4. Verify

**Direct mode (default):**

```bash
curl http://localhost:1235/v1/models
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
    "model": "nvidia/Qwen3.6-27B-NVFP4",
    "messages": [
      {"role": "user", "content": "Hello, who are you?"}
    ]
  }'

# Proxy mode (HTTPS)
curl -k https://<your-domain>/v1/chat/completions \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nvidia/Qwen3.6-27B-NVFP4",
    "messages": [
      {"role": "user", "content": "Hello, who are you?"}
    ]
  }'
```

> Use `-k` (insecure) on first boot before Let's Encrypt issues the cert (self-signed placeholder). Remove `-k` once the cert is active.

## Configurable Parameters

All parameters are in `docker-compose.yml` under `environment`:

| Variable | Default | Description |
|---|---|---|
| `MODEL_NAME` | `nvidia/Qwen3.6-27B-NVFP4` | HuggingFace model name |
| `TP_SIZE` | `1` | Tensor parallelism (1 = single GPU) |
| `MAX_MODEL_LEN` | `180800` | Maximum context length (tokens) |
| `MAX_NUM_SEQS` | `1` | Maximum concurrent sequences |
| `MAX_NUM_BATCHED_TOKENS` | `6144` | Maximum tokens per prefill batch |
| `GPU_MEMORY_UTILIZATION` | `0.94` | Fraction of usable VRAM (0.0–1.0) |
| `ENABLE_MTP` | `true` | Multi-Token Prediction (speculative decoding, 2 tokens) |
| `ENABLE_API_KEY` | `false` | API key authentication (auto-generates on first run) |
| `ENABLE_REQUEST_METRICS` | `false` | Per-request metrics (profiling) |
| `HF_TOKEN` | *(from `.env`)* | HuggingFace token |
| `LETSENCRYPT_DOMAIN` | *(none)* | Domain for Let's Encrypt certificate (e.g., `my-domain.duckdns.org`) |
| `LETSENCRYPT_EMAIL` | *(none)* | Email for Let's Encrypt certificate notifications |

## Notes

- **API Key:** enabled by default (`ENABLE_API_KEY=true`). An `sk-<uuid>` is auto-generated on first run and saved to `/root/.vllm-key/.api_key` (bind-mounted to the host). Retrieve it with `docker exec vllm-server cat /root/.vllm-key/.api_key`. To override, set `VLLM_API_KEY` in your environment. To disable, set `ENABLE_API_KEY=false`.
- **MTP (Multi-Token Prediction):** the `nvidia/Qwen3.6-27B-NVFP4` checkpoint includes up to 2 MTP layers. If you get missing MTP weights errors on first startup, set `ENABLE_MTP=false` and restart.
- **VRAM:** with `GPU_MEMORY_UTILIZATION=0.94` on 32 GB, consumption is ~30.8 GB. Do not increase further.
- **HuggingFace Cache:** the cache is mounted at `/root/.cache/huggingface` and persists across container restarts.
- **Port:** the API is exposed on `localhost:1235` (mapped from internal port 8000).
- **Reverse Proxy:** two modes via overlay:
  - **Direct** (default): `docker compose up -d` → API on `localhost:1235` (HTTP, local only)
  - **Proxy**: `docker compose -f docker-compose.yml -f docker-compose.proxy.yml up -d` → API on `https://<domain>` (HTTPS, remote)
  - The proxy overlay adds 3 containers: `docker-gen` (required by acme-companion), `nginx` (reverse proxy with SSL), and `acme-companion` (Let's Encrypt cert management). Nginx generates a self-signed placeholder on first boot and symlinks to the Let's Encrypt cert once issued. The domain (`LETSENCRYPT_DOMAIN`) is resolved from `.env` at runtime — never hardcoded in config files.
  - When proxy mode is active, ports 80/443 are exposed. The vllm container remains reachable on port 1235 as well — to disable it, comment out the `ports` section in `docker-compose.yml`.
- **DuckDNS:** register at [duckdns.org](https://www.duckdns.org), create a subdomain, and ensure it resolves to your public IP. Ports 80 and 443 must be forwarded from your router to the Docker host for Let's Encrypt validation. Set `LETSENCRYPT_DOMAIN` in `.env` to your DuckDNS subdomain.
- **Let's Encrypt:** no separate registration required. The `acme-companion` container handles certificate issuance and renewal automatically. Provide `LETSENCRYPT_EMAIL` in `.env` for renewal notifications.

## Useful Commands

```bash
# Start (direct mode — localhost:1235 only)
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
