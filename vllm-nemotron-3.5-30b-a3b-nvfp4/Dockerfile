FROM vllm/vllm-openai:v0.27.1

# Non-interactive apt operations
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Rome

COPY entrypoint.sh /entrypoint.sh

RUN apt update && apt full-upgrade -y && \
    chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
