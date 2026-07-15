#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -z "${HF_TOKEN:-}" ] && [ ! -f .env ]; then
  echo "ERROR: HF_TOKEN is not set. Export it, or copy .env.example to .env and fill it in." >&2
  exit 1
fi

docker compose pull
docker compose up -d
echo "vLLM starting — watch logs with: docker compose logs -f"
echo "API ready at http://$(hostname -I | awk '{print $1}'):8000/v1"
