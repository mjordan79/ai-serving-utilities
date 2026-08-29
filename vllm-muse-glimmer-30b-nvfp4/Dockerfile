# vLLM OpenAI server for Muse Glimmer 30B NVFP4 (RedHatAI).
#
# Base image: vllm/vllm-openai:v0.28.0 — pinned numeric tag, same pattern as
# the other stacks. v0.28.0 is the FIRST plain release with the native
# `muse_glimmer` tool-call parser (vllm/tool_parsers/muse_glimmer_tool_parser.py)
# and the first-class MuseGlimmer configs (MuseGlimmer{,Text,Vision,Assistant}Config);
# the v0.27.1 release contains none of them (verified: 0 matches in-image) and
# fails at boot with KeyError on --tool-call-parser muse_glimmer — that is why
# upstream also ships the curated `muse-glimmer` tag (avoided: unversioned, unpinned).

FROM vllm/vllm-openai:v0.28.0

# Non-interactive apt operations
ENV DEBIAN_FRONTEND=noninteractive

# Timezone
ENV TZ=Europe/Rome

# Copy and set the entrypoint
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Launch vLLM with the OpenAI-compatible API
ENTRYPOINT ["/entrypoint.sh"]
