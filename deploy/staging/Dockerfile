# Image for running self-improvement staging tests in isolation (no network).
#   docker build -f deploy/staging/Dockerfile -t sentientai-staging:latest .
# then set self_improvement.staging_image: sentientai-staging:latest
# The staged worktree is mounted at /src and put first on PYTHONPATH, so tests
# exercise the *patched* code while dependencies come from this image.
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml README.md LICENSE /tmp/sentientai/
COPY src /tmp/sentientai/src
RUN --mount=type=secret,id=ca_bundle,required=false \
    if [ -s /run/secrets/ca_bundle ]; then export PIP_CERT=/run/secrets/ca_bundle; fi \
 && pip install --no-cache-dir "/tmp/sentientai[dev]" && pip uninstall -y sentientai && rm -rf /tmp/sentientai
ENV PYTHONDONTWRITEBYTECODE=1 HOME=/tmp
USER 65534:65534
WORKDIR /src
