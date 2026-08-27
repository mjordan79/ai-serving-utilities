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
# Launch vLLM with the OpenAI-compatible API
ENTRYPOINT ["/entrypoint.sh"]
