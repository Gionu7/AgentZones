# Agent Zone image: zone runtime + bundled MCP servers (proxy doctor).
# Build behind a corporate proxy:
#   docker build --build-arg HTTPS_PROXY=$HTTPS_PROXY --build-arg HTTP_PROXY=$HTTP_PROXY \
#                -f deploy/Dockerfile -t agentzones:latest .
# Corporate TLS interception: drop the root CA(s) as *.crt into deploy/certs/ before building.
FROM python:3.12-slim

ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    PIP_CERT=/etc/ssl/certs/ca-certificates.crt

COPY deploy/certs/ /usr/local/share/ca-certificates/corp/
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl git \
 && update-ca-certificates && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY agentzones ./agentzones
COPY proxy_doctor ./proxy_doctor
COPY models ./models
COPY examples ./examples
RUN pip install --no-cache-dir . && useradd -m -u 10001 zone && mkdir -p /data && chown zone /data

USER zone
ENV ZONE_DATA=/data ZONE_CONFIG=/config/zone.yaml ZONE_HOST=0.0.0.0
EXPOSE 7700
HEALTHCHECK --interval=15s --timeout=3s CMD curl -fsS http://127.0.0.1:${ZONE_PORT:-7700}/health || exit 1
CMD ["sh", "-c", "agentzone up -c ${ZONE_CONFIG} --host ${ZONE_HOST}"]
