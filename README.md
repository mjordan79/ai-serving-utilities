# AI Serving Utilities

![vLLM](https://img.shields.io/badge/vLLM-4B8BBE?style=for-the-badge&logo=python&logoColor=white) ![SGLang](https://img.shields.io/badge/SGLang-4B8BBE?style=for-the-badge&logo=python&logoColor=white) ![NVIDIA](https://img.shields.io/badge/NVIDIA-76B900?style=for-the-badge&logo=nvidia&logoColor=white) ![GeForce RTX 5090](https://img.shields.io/badge/GeForce%20RTX%205090-76B900?style=for-the-badge&logo=nvidia&logoColor=white) ![Alibaba Cloud](https://img.shields.io/badge/Alibaba%20Cloud-FF6A00?style=for-the-badge&logo=alibabacloud&logoColor=white) ![Meta](https://img.shields.io/badge/Meta-0668E1?style=for-the-badge&logo=meta&logoColor=white) ![Red Hat AI](https://img.shields.io/badge/Red%20Hat%20AI-000000?style=for-the-badge&logo=redhat&logoColor=EE0000) ![Hugging Face](https://img.shields.io/badge/Hugging%20Face-1F2937?style=for-the-badge&logo=huggingface&logoColor=FFD23E) ![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)

Docker-based deployment utilities for running AI models across multiple serving runtimes.

Each subdirectory contains a self-contained deployment for a specific model.

by Renato Perini (mjordan79)

## Models

| Model | Producer | Publisher | Directory | HF Page | Description |
|-------|----------|-----------|-----------|---------|-------------|
| Qwen 3.8 — 27B NVFP4 | Alibaba | Unsloth | [`vllm-qwen-3.8-27b-nvfp4/`](vllm-qwen-3.8-27b-nvfp4/) | [unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4) | `unsloth/Qwen3.8-27B-NVFP4` on vLLM (Compressed-Tensors) |
| Qwen 3.6 — 27B NVFP4 | Alibaba | NVIDIA | [`vllm-qwen-3.8-27b-nvfp4/`](vllm-qwen-3.8-27b-nvfp4/) | [nvidia/Qwen3.6-27B-NVFP4](https://huggingface.co/nvidia/Qwen3.6-27B-NVFP4) | `nvidia/Qwen3.6-27B-NVFP4` on vLLM (ModelOpt) |
| Qwen 3.8 — 27B NVFP4 (SGLang) | Alibaba | RadixArk | [`sglang-qwen-3.8-27b-nvfp4/`](sglang-qwen-3.8-27b-nvfp4/) | [RadixArk/Qwen3.8-27B-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-27B-NVFP4) | `RadixArk/Qwen3.8-27B-NVFP4` on SGLang (NVFP4, Mamba/hybrid attention) |
| Muse Glimmer 30B — NVFP4 | Meta | RedHatAI | [`vllm-muse-glimmer-30b-nvfp4/`](vllm-muse-glimmer-30b-nvfp4/) | [RedHatAI/Muse-Glimmer-30B-NVFP4](https://huggingface.co/RedHatAI/Muse-Glimmer-30B-NVFP4) | `RedHatAI/Muse-Glimmer-30B-NVFP4` on vLLM (Compressed-Tensors) |
| Nemotron 3.5 Lightning — 30B A3B NVFP4 | NVIDIA | NVIDIA | [`vllm-nemotron-3.5-30b-a3b-nvfp4/`](vllm-nemotron-3.5-30b-a3b-nvfp4/) | [nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4) | Hybrid Mamba-MoE (30B total / 3B active) on vLLM (ModelOpt NVFP4 W4A16) |

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

## Target Hardware & Tuning

All deployments in this repository are **tuned for a single NVIDIA GeForce RTX 5090 (32 GB VRAM, sm_120 Blackwell)** running on Windows / WSL2 (WDDM). VRAM budgets, memory-fraction/utilization values, speculative-decoding settings, attention backends, and the "one stack holds ports 80/443 at a time" proxy constraint all assume that card and its driver stack.

If you target different hardware (more/less VRAM, a different architecture, or a datacenter GPU), re-tune the per-deployment values — see each model's `README.md` *Notes* for the knobs that are most sensitive to VRAM and architecture.

## Benchmarking

The shared benchmark suite lives in [`benchmark/`](benchmark/) and is model-agnostic — the target deployment is selected as a positional argument (default `qwen`):

```bash
cd benchmark
bash warmup.sh [qwen|muse]          # Triton kernel pre-compilation (required first)
bash run.sh [qwen|muse] [test]      # benchmark suite (8 tests × 3 iterations)
bash compare.sh <results_a> <results_b>
```

> The suite is model-agnostic by design, but currently only the `qwen` and `muse` vLLM targets are wired in — the `nemotron` and `sglang` deployments are not yet supported and are planned for a future release.

Results land in `benchmark/results/<model_label>/` (gitignored). See the [Benchmark Suite README](benchmark/README.md) for full documentation.

## License

[GNU General Public License v3.0](LICENSE)
