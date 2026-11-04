#!/usr/bin/env bash
# Start (or manage) one agent zone:   scripts/zone.sh <zone> [docker compose args...]
#   scripts/zone.sh az                      -> docker compose up -d --build
#   scripts/zone.sh b logs -f mnist-aggregator
#   COMPOSE_PROFILES=sim scripts/zone.sh b  -> also applies zones/b/sim.yml
#   AGENTZONES_GPU=1 COMPOSE_PROFILES=llm scripts/zone.sh az  -> Ollama on the NVIDIA GPU
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
ZONE=${1:?usage: scripts/zone.sh <zone> [compose args...]}
shift
DIR="$ROOT/zones/$ZONE"
[[ -f "$DIR/zone.yml" ]] || { echo "no zone file $DIR/zone.yml" >&2; exit 1; }

docker network inspect agent-mesh >/dev/null 2>&1 || docker network create agent-mesh >/dev/null

FILES=(-f "$DIR/zone.yml")
if [[ ",${COMPOSE_PROFILES:-}," == *",sim,"* && -f "$DIR/sim.yml" ]]; then
  FILES+=(-f "$DIR/sim.yml")
fi
if [[ "${AGENTZONES_GPU:-0}" == "1" ]]; then
  FILES+=(-f "$ROOT/zone/gpu.yml")
fi
[[ $# -eq 0 ]] && set -- up -d --build
exec docker compose --project-directory "$DIR" -p "zone-$ZONE" "${FILES[@]}" "$@"
