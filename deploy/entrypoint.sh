#!/bin/sh
# Trust an extra CA bundle (HOST_CA_BUNDLE, mounted at /etc/agentzones/extra-ca.crt) on top of the image's,
# e.g. the root CA of a TLS-inspecting proxy. Python, httpx, pip and curl all read SSL_CERT_FILE.
set -e
EXTRA=/etc/agentzones/extra-ca.crt
if [ -s "$EXTRA" ]; then
  cat /etc/ssl/certs/ca-certificates.crt "$EXTRA" > /tmp/ca-bundle.crt
  export SSL_CERT_FILE=/tmp/ca-bundle.crt REQUESTS_CA_BUNDLE=/tmp/ca-bundle.crt PIP_CERT=/tmp/ca-bundle.crt
fi
exec "$@"
