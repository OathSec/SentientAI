# SentientAI runtime image.
#   docker build -t sentientai:latest .
# The sandbox launches sibling containers through the host Docker socket
# (mounted by deploy/docker-compose.yml). That socket is root-equivalent on the
# host: only mount it on a machine dedicated to SentientAI, or use
# sandbox.backend=disabled.

ARG PYTHON_IMAGE=python:3.12-slim
ARG DOCKER_CLI_IMAGE=docker:27-cli

FROM ${DOCKER_CLI_IMAGE} AS dockercli

FROM ${PYTHON_IMAGE} AS build
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /src
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
# Optional, build-time only: trust a CA bundle for a TLS-inspecting corporate proxy.
# The bundle must contain the full set of trusted roots; it is never stored in the image.
#   docker build --secret id=ca_bundle,src=/etc/ssl/certs/ca-certificates.crt .
RUN --mount=type=secret,id=ca_bundle,required=false \
    if [ -s /run/secrets/ca_bundle ]; then export PIP_CERT=/run/secrets/ca_bundle; fi \
 && python -m venv /opt/venv && /opt/venv/bin/pip install ".[all]"

FROM ${PYTHON_IMAGE}
LABEL org.opencontainers.image.title="SentientAI" \
      org.opencontainers.image.description="Persistent autonomous cognitive runtime" \
      org.opencontainers.image.licenses="MIT"
RUN apt-get update \
 && apt-get install -y --no-install-recommends git ca-certificates tini \
 && rm -rf /var/lib/apt/lists/* \
 && groupadd --system sentient && useradd --system --gid sentient --create-home sentient \
 && mkdir -p /data /workspace /config && chown -R sentient:sentient /data /workspace /config
COPY --from=dockercli /usr/local/bin/docker /usr/local/bin/docker
COPY --from=build /opt/venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    SENTIENT_STORAGE__DATA_DIR=/data \
    SENTIENT_SAFETY__WORKSPACE_ROOTS='["/workspace"]' \
    SENTIENT_API__HOST=0.0.0.0 \
    SENTIENT_LOGGING__FORMAT=json
USER sentient
WORKDIR /home/sentient
EXPOSE 8765
VOLUME ["/data", "/workspace"]
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8765/health', timeout=4).status == 200 else 1)"
ENTRYPOINT ["/usr/bin/tini", "--", "sentientai"]
CMD ["run"]
