#!/usr/bin/env bash
# One command to start a zone:  ./scripts/up.sh [zone.yaml]
#  - with Docker: docker compose up (zone + Ollama + model download)
#  - without:     local virtualenv, expects an OpenAI-compatible model at $MODEL_URL (optional)
set -euo pipefail
cd "$(dirname "$0")/.."
CFG=${1:-zone.yaml}
if command -v docker >/dev/null && docker info >/dev/null 2>&1 && [[ "${NO_DOCKER:-}" != 1 ]]; then
  export ZONE_CONFIG_FILE="$(realpath "$CFG")"
  docker compose up -d --build
  echo "zone starting: http://localhost:${ZONE_PORT:-7700}  (logs: docker compose logs -f zone)"
else
  [[ -d .venv ]] || python3 -m venv .venv
  .venv/bin/pip install -q -e .
  exec .venv/bin/agentzone up -c "$CFG"
fi
