FROM vllm/vllm-openai:v0.27.1

# Needed for packages that require localization.
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Rome

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
