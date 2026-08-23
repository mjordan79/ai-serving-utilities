# AI Serving Utilities

Docker-based deployment utilities for running AI models across multiple serving runtimes.

Each subdirectory contains a self-contained deployment for a specific model.

by Renato Perini (mjordan79)

## Models

| Model | Producer | Publisher | Directory | HF Page | Description |
|-------|----------|-----------|-----------|---------|-------------|
| Qwen 3.8 — 27B NVFP4 | Alibaba | Unsloth | [`vllm-qwen-3.8-27b-nvfp4/`](vllm-qwen-3.8-27b-nvfp4/) | [unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4) | `unsloth/Qwen3.8-27B-NVFP4` on vLLM (Compressed-Tensors) |
| Qwen 3.6 — 27B NVFP4 | Alibaba | NVIDIA | [`vllm-qwen-3.8-27b-nvfp4/`](vllm-qwen-3.8-27b-nvfp4/) | [nvidia/Qwen3.6-27B-NVFP4](https://huggingface.co/nvidia/Qwen3.6-27B-NVFP4) | `nvidia/Qwen3.6-27B-NVFP4` on vLLM (ModelOpt) |
| Muse Glimmer 30B — NVFP4 | Meta | RedHatAI | [`vllm-muse-glimmer-30b-nvfp4/`](vllm-muse-glimmer-30b-nvfp4/) | [RedHatAI/Muse-Glimmer-30B-NVFP4](https://huggingface.co/RedHatAI/Muse-Glimmer-30B-NVFP4) | `RedHatAI/Muse-Glimmer-30B-NVFP4` on vLLM (Compressed-Tensors) |

The two Qwen variants are served by the **same deployment** — the active checkpoint is selected in `.env` (`MODEL_NAME`, `QUANTIZATION`), not in the directory name.

## Directory Convention

Every model deployment follows the same structure:

```
<runtime>-<model-name>/
├── docker-compose.yml    # Service definition
├── Dockerfile            # Custom image
├── entrypoint.sh         # Startup script (HF token, API key, model download)
├── .env                  # Local environment variables (gitignored)
└── README.md             # Model-specific setup and usage
```

**Runtime** — the serving framework (e.g., `vllm`, `tgi`, `llama-cpp`, `ollama`).  
**Model name** — lowercase, hyphenated, includes size/precision (e.g., `qwen-3.8-27b-nvfp4`).

One directory per model deployment. Where the same deployment can serve multiple HuggingFace checkpoints (different publisher/quantization), the variant is selected in `.env` (`MODEL_NAME`, `QUANTIZATION`) rather than in the directory name.

## Quick Start

1. Navigate to the desired model directory.
2. Create a `.env` file with your `HF_TOKEN`.
3. Run `docker compose build && docker compose up -d`.

See each model's `README.md` for prerequisites and configuration details.

## License

[GNU General Public License v3.0](LICENSE)
