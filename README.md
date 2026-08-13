# AI Serving Utilities

Docker-based deployment utilities for running AI models across multiple serving runtimes.

Each subdirectory contains a self-contained deployment for a specific model.

by Renato Perini (mjordan79)

## Models

| Model | Producer | Publisher | Directory | HF Page | Description |
|-------|----------|-----------|-----------|---------|-------------|
| Qwen 3.6 — 27B NVFP4 | Alibaba | NVIDIA | [`vllm-nvidia-qwen-3.6-27b-nvfp4/`](vllm-nvidia-qwen-3.6-27b-nvfp4/) | [nvidia/Qwen3.6-27B-NVFP4](https://huggingface.co/nvidia/Qwen3.6-27B-NVFP4) | `nvidia/Qwen3.6-27B-NVFP4` on vLLM (ModelOpt) |
| Qwen 3.6 — 27B NVFP4 | Alibaba | Unsloth | [`vllm-nvidia-qwen-3.6-27b-nvfp4/`](vllm-nvidia-qwen-3.6-27b-nvfp4/) | [unsloth/Qwen3.6-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.6-27B-NVFP4) | `unsloth/Qwen3.6-27B-NVFP4` on vLLM (Compressed-Tensors) |

## Directory Convention

Every model deployment follows the same structure:

```
<runtime>-<provider>-<model-name>/
├── docker-compose.yml    # Service definition
├── Dockerfile            # Custom image
├── entrypoint.sh         # Startup script (HF token, API key, model download)
├── .env                  # Local environment variables (gitignored)
└── README.md             # Model-specific setup and usage
```

**Runtime** — the serving framework (e.g., `vllm`, `tgi`, `llama-cpp`, `ollama`).  
**Provider** — the model publisher (e.g., `nvidia`, `meta`, `mistral`, `google`).  
**Model name** — lowercase, hyphenated, includes size/precision (e.g., `qwen-3.6-27b-nvfp4`).

## Quick Start

1. Navigate to the desired model directory.
2. Create a `.env` file with your `HF_TOKEN`.
3. Run `docker compose build && docker compose up -d`.

See each model's `README.md` for prerequisites and configuration details.

## License

[GNU General Public License v3.0](LICENSE)
