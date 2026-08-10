# NVIDIA Qwen3.6-27B-NVFP4 — vLLM Server

![NVIDIA](https://img.shields.io/badge/NVIDIA-76B900?style=for-the-badge&logo=nvidia&logoColor=white) ![Alibaba Cloud](https://img.shields.io/badge/Alibaba%20Cloud-FF6A00?style=for-the-badge&logo=alibabacloud&logoColor=white) ![vLLM](https://img.shields.io/badge/vLLM-4B8BBE?style=for-the-badge&logo=python&logoColor=white) ![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)

Docker Compose deployment for `nvidia/Qwen3.6-27B-NVFP4` on vLLM (OpenAI-compatible API).

by Renato Perini (mjordan79)

## Prerequisites

- **NVIDIA GPU** with CUDA drivers (tested on Geforce RTX 5090 — 32 GB VRAM)
- **NVIDIA Container Toolkit** (natively or on WSL) installed (`nvidia-container-toolkit`)
- **Docker** + **Docker Compose**
- **HuggingFace token** with access to `nvidia/Qwen3.6-27B-NVFP4`

## Setup

### 1. Configure the HuggingFace token

Create a `.env` file in the same directory as `docker-compose.yml`:

```env
HF_TOKEN=hf_your_token_here
```

> The token must **never** be hardcoded in docker-compose or the entrypoint.

### 2. Build and start

```bash
docker compose build
docker compose up -d
```

The first run downloads the model (~15 GB) and stores it in the HuggingFace cache mounted at `/root/.cache/huggingface`.

### 3. Get your API key (optional)

API authentication is **disabled by default**. To enable it, set `ENABLE_API_KEY=true` in `docker-compose.yml`.

On the first run with API key enabled, the entrypoint generates a key and prints it to the logs:

```bash
docker compose logs | grep "Generated API key"
```

Save this value — it will **not** be shown again on subsequent restarts. You can also retrieve it anytime:

```bash
cat /root/.vllm-key/.api_key
```

### 4. Verify

```bash
curl http://localhost:1235/v1/models
```

You should see the model in the list.

## Usage (OpenAI-compatible API)

```bash
curl http://localhost:1235/v1/chat/completions \
  -H "Authorization: Bearer YOUR_API_KEY" \  # required if ENABLE_API_KEY=true
  -H "Content-Type: application/json" \
  -d '{
    "model": "nvidia/Qwen3.6-27B-NVFP4",
    "messages": [
      {"role": "user", "content": "Hello, who are you?"}
    ]
  }'
```

Or use the `openai` library pointing to `http://localhost:1235`:

```python
from openai import OpenAI

# api_key="YOUR_API_KEY" if ENABLE_API_KEY=true, otherwise "not-needed"
client = OpenAI(base_url="http://localhost:1235", api_key="not-needed")
response = client.chat.completions.create(
    model="nvidia/Qwen3.6-27B-NVFP4",
    messages=[{"role": "user", "content": "Hello"}]
)
print(response.choices[0].message.content)
```

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

## Notes

- **API Key:** disabled by default. Set `ENABLE_API_KEY=true` in `docker-compose.yml` to enable. On first run, an `sk-<uuid>` is auto-generated and saved to `/root/.vllm-key/.api_key` (bind-mounted to the host). Pass it via `Authorization: Bearer <key>` on every request. To override the auto-generated key, set `VLLM_API_KEY` in your environment — the entrypoint will use that instead.
- **MTP (Multi-Token Prediction):** the `nvidia/Qwen3.6-27B-NVFP4` checkpoint includes up to 2 MTP layers. If you get missing MTP weights errors on first startup, set `ENABLE_MTP=false` and restart.
- **VRAM:** with `GPU_MEMORY_UTILIZATION=0.94` on 32 GB, consumption is ~30.8 GB. Do not increase further.
- **HuggingFace Cache:** the cache is mounted at `/root/.cache/huggingface` and persists across container restarts.
- **Port:** the API is exposed on `localhost:1235` (mapped from internal port 8000).

## Useful Commands

```bash
# Real-time logs
docker compose logs -f

# Stop
docker compose down

# Stop and remove volumes (re-downloads model on next start)
docker compose down -v

# Rebuild after entrypoint/Dockerfile changes
docker compose build --no-cache
```
