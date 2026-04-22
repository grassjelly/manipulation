#!/bin/bash
DOCKER_DIR="$(dirname "$0")/docker"

if ! docker ps --format '{{.Names}}' | grep -q '^dev$'; then
  echo "Starting dev container..."
  docker compose -f "$DOCKER_DIR/docker-compose.yaml" up -d dev
fi

if [ $# -eq 0 ]; then
  docker exec -it dev bash
else
  docker exec dev bash -c "source install/setup.bash && $*"
fi
