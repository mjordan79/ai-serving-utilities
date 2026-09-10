<div align="center">

<a href="https://nvidia.com">
  <img src="https://img.shields.io/badge/NVIDIA%20GPU-76A900?style=flat-square" alt="NVIDIA GPU">
</a>
<a href="https://ai.google.dev/gemma">
  <img src="https://img.shields.io/badge/Google%20Gemma-412991?style=flat-square&logo=google&logoColor=white" alt="Google Gemma">
</a>
<a href="https://github.com/vllm-project/vllm">
  <img src="https://img.shields.io/badge/vLLM%20v0.29.0-4B8BBE?style=flat-square" alt="vLLM v0.29.0">
</a>
<a href="https://github.com/docker">
  <img src="https://img.shields.io/badge/Docker-EE5A24?style=flat-square" alt="Docker">
</a>

<br/>

# Gemma 4 26B A4B NVFP4 — vLLM Server

by Renato Perini (mjordan79)

</div>

**Prerequisites**

- NVIDIA RTX 5090 32 GB (the target GPU)
- NVIDIA Container Toolkit
- Docker
- HuggingFace token

**For remote access (optional):**

- Domain name, e.g. DuckDNS (free) — see the DuckDNS section below
- Open ports **80** and **443** in your router firewall and forward them to this Docker host

## Setup

**1. Configure the environment**

Create `.env` from `.env.example`:

```
# HuggingFace token (required)
HF_TOKEN=hf_your_token_here

# Optional: fixed API key. Secret — keep it here in the gitignored .env, never in
# docker-compose.yml. If left empty, the entrypoint generates and persists
# sk-<uuid> in the vllm-keys volume.
#VLLM_API_KEY=sk-your_fixed_key_here

# Reverse proxy (optional — required for HTTPS/DuckDNS)
LETSENCRYPT_DOMAIN=your-domain.duckdns.org
LETSENCRYPT_EMAIL=you@example.com

# Gemma 4 26B A4B IT NVFP4: NVIDIA ModelOpt checkpoint for Blackwell.
MODEL_NAME=nvidia/Gemma-4-26B-A4B-NVFP4
QUANTIZATION=modelopt_fp4
MOE_BACKEND=cutlass
HF_CACHE_VOLUME=hf-cache-gemma4
MAX_MODEL_LEN=131072
```

To use a fixed API key instead of auto-generation, set `VLLM_API_KEY` in your `.env` (gitignored). The value must follow the `sk-<uuid>` format. Leave it empty to let the entrypoint generate one.

**2. Start the service**

```bash
cd vllm-gemma-4-26b-a4b-nvfp4
docker compose up -d
```

The first run downloads the model (~15 GB of weights) and saves the HF cache in a named volume. Subsequent boots reuse the cache.

**2b. HTTPS proxy (DuckDNS + Let's Encrypt)**

For remote access, use the proxy overlay. It adds 3 containers: `nginx` (reverse proxy with SSL), `acme-companion` (Let's Encrypt cert), and `docker-gen` (required by acme-companion).

```bash
docker compose -f docker-compose.yml -f docker-compose.proxy.yml up -d
```

On first boot, acme-companion requests a Let's Encrypt certificate for `LETSENCRYPT_DOMAIN`. Until the cert is issued (~2 min), nginx serves a self-signed placeholder. Check the cert progress:

```bash
docker compose logs acme-companion  # watch for "success"
```

The domain is resolved from `LETSENCRYPT_DOMAIN` at runtime — it is never hardcoded in the nginx config.

> **Port conflict note:** the 3 proxy containers bind host ports **80/443**. Only one stack's proxy may be up at a time. The direct-mode ports do not conflict (Qwen `1235`, Muse `1236`, Nemotron `1237`, Qwen-SGLang `1238`, Gemma 4 on `1239`).

**3. Get the API key**

vLLM auto-generates a key on first start. Retrieve it from the container:

```bash
docker compose logs | grep "Generated API key"
# or
docker exec vllm-gemma4-server cat /root/.vllm-key/.api_key
```

If you set `VLLM_API_KEY` in `.env`, that key is used instead.

**4. Verify**

```bash
curl http://localhost:1239/health
# or proxy mode:
curl -k https://your-domain.duckdns.org/health
```

## Usage

The vLLM OpenAI-compatible API is exposed on the host port `1239` (direct mode) or via `https://<your-domain>` (proxy mode):

```bash
# Direct mode
curl http://localhost:1239/v1/chat/completions \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "vllm/nvidia/gemma-4-26b-a4b-nvfp4",
    "messages": [
      {"role": "user", "content": "Hello, who are you?"}
    ]
  }'

# Proxy mode (HTTPS)
curl -k https://<your-domain>/v1/chat/completions \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "vllm/nvidia/gemma-4-26b-a4b-nvfp4",
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
| `MODEL_NAME` | `nvidia/Gemma-4-26B-A4B-NVFP4` | HuggingFace model name |
| `QUANTIZATION` | `modelopt_fp4` | Quantization backend — NVFP4 (W4A16) via the NVIDIA ModelOpt checkpoint format |
| `HF_CACHE_VOLUME` | `hf-cache-gemma4` | Named volume for the HF cache |
| `MAX_MODEL_LEN` | `131072` | Maximum context length |
| `DTYPE` | `auto` | Data type for model weights |
| `TRUST_REMOTE_CODE` | `false` | Pass `--trust-remote-code`. `false` for this stack (Gemma 4 is a built-in vLLM architecture and the NVFP4 checkpoint is handled natively via `modelopt_fp4` — no custom HF modeling code expected); set `true` only if the checkpoint fails to load with a `--trust-remote-code` error |
| `SKIP_MM_PROFILING` | `false` | Gemma 4 is multimodal; multimodal profiling at startup stays on. Set `true` to skip it |
| `HF_TOKEN` | *(from `.env`)* | HuggingFace token |

### Performance

| Variable | Default | Description |
|---|---|---|
| `TP_SIZE` | `1` | Tensor parallelism (1 = single GPU) |
| `MOE_BACKEND` | `cutlass` | vLLM CUTLASS MoE backend for the Blackwell NVFP4 checkpoint; set empty for auto-selection |
| `GPU_MEMORY_UTILIZATION` | `0.88` | Fraction of usable VRAM (0.0–1.0). Shared-GPU safe: this stack coexists with the live Qwen stack on the same 32 GB card. Raise to 0.92–0.94 when the GPU is exclusive |
| `MAX_NUM_SEQS` | `1` | Maximum concurrent sequences |
| `MAX_NUM_BATCHED_TOKENS` | `6144` | Maximum tokens per prefill batch |
| `MAX_NUM_QUEUED_REQS` | `4` | Admission cap: max in-flight requests (waiting + running); `4` = `4× MAX_NUM_SEQS`. Overflow → HTTP `503` |
| `MAX_NUM_QUEUED_TOKENS` | `128K` | Admission cap: max queued prefill tokens counted conservatively (prefix-cache hits are not subtracted); `128K` spans the full single-prompt context, so a long prompt up to `MAX_MODEL_LEN` is admitted. Overflow → HTTP `503` |
| `KV_CACHE_DTYPE` | `auto` | KV cache data type — FP8 is not validated for Gemma 4 on the target platform; left to the engine |
| `ATTENTION_BACKEND` | `""` (auto) | Attention backend — leave empty: vLLM auto-selects a backend that supports the multimodal prefix path (`support_mm_prefix()`); FlashInfer does not. Set a value only with a validated backend |
| `PERFORMANCE_MODE` | `interactivity` | vLLM performance mode |
| `ENABLE_CHUNKED_PREFILL` | `true` | Split long prefills into chunks |
| `ENABLE_PREFIX_CACHING` | `true` | Cache shared prompt prefixes |
| `ENABLE_HYBRID_KV_CACHE_MANAGER` | `true` | Hybrid (CPU+GPU) KV cache manager |
| `ENABLE_MTP` | `false` | Disabled by default: the Gemma 4 checkpoint has no MTP layers — the model config declares none |
| `MTP_NUM_SPECULATIVE_TOKENS` | `1` | Speculative tokens per step — inert while MTP is disabled |
| `PER_REQUEST_SPEC_DECODE_METRICS` | `none` | Per-request spec-decode acceptance metrics in the response (`metrics.speculative_decoding`): `none` off \| `summary` \| `detailed` |

### Behavior & features

| Variable | Default | Description |
|---|---|---|
| `LANGUAGE_MODEL_ONLY` | `false` | Gemma 4 is multimodal by default (text + image). The entrypoint adds `--language-model-only` only when explicitly set to `true` |
| `REASONING_PARSER` | `gemma4` | Parser that splits reasoning content from the response |
| `DEFAULT_ENABLE_THINKING` | `false` | Server-side default for `enable_thinking` — the Gemma 4 model card forbids carrying thinking in multi-turn history; per-request `chat_template_kwargs` can still override it |
| `DEFAULT_PRESERVE_THINKING` | `false` | Do not keep historical assistant thinking in multi-turn context |
| `TOOL_CALL_PARSER` | `gemma4` | Tool-call response parser |
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
| `PORT` | `8000` | In-container port (host mapping `1239:8000`) |

### Runtime & low-level

| Variable | Default | Description |
|---|---|---|
| `VLLM_WSL2_ENABLE_PIN_MEMORY` | `1` | Hardcoded, WSL2 only: non-blocking per-step host→device staging of attention metadata; also required for the UVA / V2 model runner |
| `VLLM_USE_V2_MODEL_RUNNER` | `1` | V2 model runner — the GA default on vLLM 0.29, active without a pin; `VLLM_WSL2_ENABLE_PIN_MEMORY=1` above enables its UVA path on the target WSL2 platform. Gemma 4's V2 + auto-select multimodal-prefix attention path is unvalidated end-to-end — if boot or the first request fails on V2, add `- VLLM_USE_V2_MODEL_RUNNER=0` to the compose env (one-line fallback; V1 is the validated Gemma 4 path) and recreate the container |
| `SAFETENSORS_LOAD_STRATEGY` | `prefetch` | Weight loading strategy |
| `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS` | `1` | Estimate CUDA-graph memory in the profiler (on) |
| `NVIDIA_VISIBLE_DEVICES` | `all` | GPU passthrough |
| `NVIDIA_DRIVER_CAPABILITIES` | `compute,utility` | GPU driver capabilities |

### MTP (Multi-Token Prediction)

| Variable | Default | Description |
|---|---|---|
| `ENABLE_MTP` | `false` | Disabled: the Gemma 4 checkpoint has no MTP layers; enabling MTP would have no draft model to sample from |
| `MTP_NUM_SPECULATIVE_TOKENS` | `1` | Speculative tokens per step — inert while MTP is disabled |

### Proxy overlay (Let's Encrypt)

| Variable | Default | Description |
|---|---|---|
| `LETSENCRYPT_DOMAIN` | *(none)* | Domain for Let's Encrypt certificate (e.g., `my-domain.duckdns.org`) |
| `LETSENCRYPT_EMAIL` | *(none)* | Email for Let's Encrypt certificate notifications |

## Notes

- **API Key:** enabled by default (`ENABLE_API_KEY=true`). An `sk-<uuid>` is auto-generated on first run and saved to the `vllm-keys` volume at `/root/.vllm-key/.api_key`. Retrieve it with `docker exec vllm-gemma4-server cat /root/.vllm-key/.api_key`. To use a fixed key, set `VLLM_API_KEY` in `.env` (gitignored; compose passes it through and the entrypoint uses it instead of generating one). To disable, change `- ENABLE_API_KEY` to `- ENABLE_API_KEY=false` in `docker-compose.yml`.
- **MTP (Multi-Token Prediction):** the Gemma 4 checkpoint has no MTP layers — the model config declares none (unlike the Nemotron 3.5 stack in this repo, whose checkpoint does). vLLM v0.29.0 accepts the `mtp` speculative method; on this model there is simply nothing to draft from. Keep `ENABLE_MTP=false`.
- **Per-request spec-decode metrics:** `PER_REQUEST_SPEC_DECODE_METRICS` (default `none`) controls the experimental `metrics.speculative_decoding` response field: `summary` adds mean acceptance length, draft acceptance rate and a step-by-draft-length histogram; `detailed` additionally records the ordered per-step accepted/proposed arrays. Reported only for single-sequence requests (`n=1`); independent of `DISABLE_LOG_STATS`. vLLM refuses to start if set to a non-`none` value while speculative decoding is disabled — in this stack MTP is off, so the variable must stay `none`.
- **VRAM:** with `GPU_MEMORY_UTILIZATION=0.88` on 32 GB, the engine gets ~28.2 GB; this stack coexists with the live Qwen stack on the same card.
- **HuggingFace Cache:** the cache is mounted at `/root/.cache/huggingface` and persists across container restarts.
- **Port:** the API is exposed on host port `1239` (mapped from internal port 8000), bound to `0.0.0.0` by default — reachable from the LAN, not only localhost. It is Bearer-authenticated, but prefer the TLS proxy for non-local access.
- **Reverse Proxy:** two modes via overlay:
  - **Direct** (default): `docker compose up -d` → API on port `1239` (HTTP, LAN-reachable, Bearer-authenticated)
  - **Proxy**: `docker compose -f docker-compose.yml -f docker-compose.proxy.yml up -d` → API on `https://<domain>` (HTTPS, remote)
  - The proxy overlay adds 3 containers: `docker-gen` (required by acme-companion), `nginx` (reverse proxy with SSL), and `acme-companion` (Let's Encrypt cert management). Nginx generates a self-signed placeholder on first boot and symlinks to the Let's Encrypt cert once issued. The domain (`LETSENCRYPT_DOMAIN`) is resolved from `.env` at runtime — never hardcoded in config files.
  - **Known limitation:** the overlay's `ports: []` does NOT remove the base file's `1239:8000` publish (Compose merges lists; an empty list is a no-op). Port `1239` stays published on the host even in proxy mode. nginx only guards the public 80/443 path — LAN hosts can still reach vLLM directly on 1239, bypassing the nginx allowlist (see Security posture).
- **DuckDNS:** register at [duckdns.org](https://www.duckdns.org), create a subdomain, and ensure it resolves to your public IP. Ports 80 and 443 must be forwarded from your router to the Docker host for Let's Encrypt validation. Set `LETSENCRYPT_DOMAIN` in `.env` to your DuckDNS subdomain.
- **Let's Encrypt:** no separate registration required. The `acme-companion` container handles certificate issuance and renewal automatically. Provide `LETSENCRYPT_EMAIL` in `.env` for renewal notifications.

## Known constraints

1. **Unvalidated combo:** V2 model runner + auto-select multimodal-prefix attention + sm120/WSL2 is boot-validated but not yet end-to-end validated. If boot or the first request fails on V2, set `VLLM_USE_V2_MODEL_RUNNER=0` and restart — one-line fallback, V1 is the validated Gemma 4 path.
2. **Do not force `ATTENTION_BACKEND`:** the mm-prefix path needs a backend with `support_mm_prefix()`; FlashInfer does not. Leave `ATTENTION_BACKEND` empty (auto-select) and set a value only with a backend validated for this stack.
3. **KV cache FP8 is unvalidated** for Gemma 4 on the target platform — `KV_CACHE_DTYPE=auto`.
4. **MTP:** the Gemma 4 checkpoint has no MTP layers; `ENABLE_MTP` must remain `false`.
5. **transformers:** no pin — vLLM v0.29.0 handles the Gemma 4 hybrid config natively. If a "heterogeneous config" error reproduces, add `RUN pip install "transformers<5.15"` to the Dockerfile and rebuild.

## VRAM escalation ladder

If the engine OOMs or the estimated context does not fit:

1. Read *estimated maximum model len* from the boot log and lower `MAX_MODEL_LEN` below the estimate.
2. Lower `GPU_MEMORY_UTILIZATION` to `0.85`.
3. Keep `MAX_NUM_SEQS=1` and reduce `MAX_NUM_BATCHED_TOKENS`.

If it starts but dies under load, apply the same three dials in the same order.

## Optional: vLLM-Copilot budget

In VS Code `settings.json`, under `vllm-copilot`, add:

```json
{
  "id": "gemma4",
  "displayName": "vllm/nvidia/Gemma-4-26B-A4B-NVFP4",
  "vllmModelId": "vllm/nvidia/gemma-4-26b-a4b-nvfp4",
  "maxOutputTokens": [32768, 16384, 8192]
}
```

`maxOutputTokens` is the Output Length picker: the first element is the default and wins over any `max_tokens` in the model's `generation_config`. No `maxInputTokens` pin (the server context minus the effective output pick is the input ceiling) and no `defaultParams` (the server's generation config is the authority).

## Security posture

Hardened against the [vLLM security docs](https://docs.vllm.ai/en/latest/usage/security/); endpoint claims verified against the running server (vLLM v0.29.0).

**What `--api-key` does not protect:** the key only authenticates `/v1`, `/v2` and `/inference`. On v0.29.0 the following endpoints answer **without credentials** (probed live): `/invocations` (SageMaker-compatible inference — a full auth bypass), `/generative_scoring`, `/tokenize`, `/detokenize`, `/scale_elastic_ep`, `/is_scaling_elastic_ep`, `/ping`, `/version`, `/metrics`, `/load`. `/pause`, `/abort_requests`, the dev-mode and weight-update endpoints do not exist in this version (and dev mode is never enabled).

**Controls (nginx, HTTPS path):**

| Control | Value | Purpose |
|---|---|---|
| API allowlist | only `/v1/*` proxied | every other endpoint 404s at the edge |
| Info-leak blocklist | `/version`, `/metrics`, `/load`, OpenAPI docs → 403 | no CVE fingerprinting, no internal state, no schema leakage |
| Rate limit | 10 r/s per client, burst 20 | abuse bounded |
| Body cap | `client_max_body_size 4m` | 4m covers the full ~131k-token context with margin and bounds abuse |
| HSTS | 1-year max-age | TLS enforced for one year |
| TLS | Mozilla Intermediate profile, OCSP stapling | AEAD-only, forward secrecy |

**Controls (vLLM):** `ENABLE_API_KEY=true` (Bearer auth on `/v1`), `VLLM_MAX_N_SEQUENCES=16` (caps `n` per request). Admission control (`--max-num-queued-reqs` / `--max-num-queued-tokens`, defaults `4` / `128K`) caps in-flight requests and queued prefill tokens server-side; overflow returns HTTP `503` from the engine as a backstop under the nginx rate limit.

**Residual risk:** in proxy mode, port `1239` stays published on the LAN (see the Known limitation above). LAN hosts can reach vLLM directly, bypassing the nginx allowlist — acceptable for a home network, not for public exposure.

**Supply chain:** `TRUST_REMOTE_CODE` defaults to `false`: Gemma 4 loads through vLLM's built-in architecture registry and its NVFP4 weights are consumed natively via `modelopt_fp4`, so no remote-code surface is exposed at load. Set `TRUST_REMOTE_CODE=true` only if the checkpoint starts failing with a custom-code / `auto_map` load error; with `true` the flag becomes a supply-chain trust in the Hugging Face repo, not a runtime API surface.

## Useful Commands

```bash
# Start / stop the service (direct mode)
cd vllm-gemma-4-26b-a4b-nvfp4
docker compose up -d
docker compose stop

# Proxy mode (HTTPS)
docker compose -f docker-compose.yml -f docker-compose.proxy.yml up -d

# Watch the logs
docker compose logs -f

# Get the API key
docker exec vllm-gemma4-server cat /root/.vllm-key/.api_key

# Verify
curl http://localhost:1239/health

# Benchmark
cd ..
bash warmup.sh gemma4
bash run.sh gemma4
```
