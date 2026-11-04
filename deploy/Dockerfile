# One image for every part of a zone: gateway, MCP server and all agents (AGENT_IMPL picks the agent).
# Behind a corporate proxy:
#   docker build --build-arg HTTPS_PROXY=$HTTPS_PROXY --build-arg HTTP_PROXY=$HTTP_PROXY -f deploy/Dockerfile .
# TLS inspection at build time: drop the root CA(s) as *.crt into deploy/certs/. At run time use HOST_CA_BUNDLE.
# Extra libraries for your own FL plugins (torch, ...): add them to the pip install line.
FROM python:3.12-slim

ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    PIP_CERT=/etc/ssl/certs/ca-certificates.crt

COPY deploy/certs/ /usr/local/share/ca-certificates/corp/
# Extra Debian packages (git lets Proxy Doctor apply `git config` fixes). Set APT_PACKAGES= to skip apt,
# e.g. when the build network cannot reach a Debian mirror. Mirrors are switched to HTTPS: many
# corporate proxies refuse plain HTTP.
ARG APT_PACKAGES="git"
RUN update-ca-certificates \
 && if [ -n "$APT_PACKAGES" ]; then \
      sed -i 's|http://deb.debian.org|https://deb.debian.org|g' /etc/apt/sources.list.d/debian.sources \
      && apt-get update && apt-get install -y --no-install-recommends $APT_PACKAGES \
      && rm -rf /var/lib/apt/lists/*; fi

WORKDIR /app
COPY pyproject.toml README.md ./
COPY agentzones ./agentzones
COPY proxy_doctor ./proxy_doctor
COPY tasks ./tasks
COPY models ./models
RUN pip install --no-cache-dir . \
 && useradd -m -u 10001 zone && mkdir -p /data /models /etc/agentzones && chown zone /data /models
COPY deploy/entrypoint.sh /usr/local/bin/entrypoint.sh

USER zone
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["agentzone", "--help"]
