FROM lmsysorg/sglang:dev-cu13-qwen38-27b-dflash2

# Needed for packages that require localization.
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Rome

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
