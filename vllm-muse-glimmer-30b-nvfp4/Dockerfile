# vLLM OpenAI server for Muse Glimmer 30B NVFP4 (RedHatAI).
#
# Base image: vllm/vllm-openai:muse-glimmer — curated for Muse Glimmer, ships
# the native `muse_glimmer` tool-call/reasoning parsers (available since
# vLLM 0.27.0+). If the tag is not pulled yet, verify with:
#   docker pull vllm/vllm-openai:muse-glimmer
# Fallback (plain release, parsers included from 0.27.0):
#   docker compose build --build-arg VLLM_BASE=vllm/vllm-openai:v0.27.1
ARG VLLM_BASE=vllm/vllm-openai:muse-glimmer
FROM ${VLLM_BASE}

# Non-interactive apt operations
ENV DEBIAN_FRONTEND=noninteractive

# Timezone
ENV TZ=Europe/Rome

# Copy and set the entrypoint
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Launch vLLM with the OpenAI-compatible API
ENTRYPOINT ["/entrypoint.sh"]
