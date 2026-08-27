# vLLM OpenAI server for NVIDIA Nemotron 3.5 Lightning 30B A3B NVFP4.
#
# Base image: vllm/vllm-openai:v0.27.1 — the version pinned as minimum by the
# vLLM recipe for this model (Mamba backends, FP4 MoE via Marlin, MTP
# speculative decoding). Same base as the Qwen stack: no new image pull.
FROM vllm/vllm-openai:v0.27.1

# Non-interactive apt operations
ENV DEBIAN_FRONTEND=noninteractive

# Timezone
ENV TZ=Europe/Rome

# Copy and set the entrypoint
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Tuned MoE kernel configs (generated with benchmarks/kernels/benchmark_moe.py
# --tune, procedure in nemotron.issue). Copied additively into the vLLM
# configs dir: the default configs bundled in the image are preserved. The
# README.md inside moe-configs/ keeps this COPY valid before the tuned JSON
# exists.
COPY moe-configs/ /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/configs/

# Launch vLLM with the OpenAI-compatible API
ENTRYPOINT ["/entrypoint.sh"]
