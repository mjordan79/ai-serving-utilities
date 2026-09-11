FROM lmsysorg/sglang:v0.5.19

# Needed for packages that require localization.
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Rome

COPY entrypoint.sh /entrypoint.sh

RUN apt update && apt full-upgrade -y && \
    chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
