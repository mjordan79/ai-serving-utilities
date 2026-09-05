FROM lmsysorg/sglang:nightly-dev-cu13-20260901-07c8f729

# Needed for packages that require localization.
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Rome

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
