# Richer execution sandbox: scientific Python + common CLI tools, unprivileged user.
#   docker build -t sentientai-sandbox:latest deploy/sandbox
# then set sandbox.image: sentientai-sandbox:latest
# The runtime still applies --network none, --read-only, --cap-drop ALL, resource limits.
FROM python:3.12-slim
RUN apt-get update \
 && apt-get install -y --no-install-recommends git jq ripgrep sqlite3 curl \
 && rm -rf /var/lib/apt/lists/* \
 && pip install --no-cache-dir numpy pandas scipy matplotlib sympy pyyaml requests beautifulsoup4 \
 && mkdir -p /work && chown 65534:65534 /work
ENV MPLBACKEND=Agg HOME=/tmp
USER 65534:65534
WORKDIR /work
