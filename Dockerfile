FROM vllm/vllm-openai:nightly

# Serve per installare pacchetti che richiedono localizzazione.
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Rome

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
