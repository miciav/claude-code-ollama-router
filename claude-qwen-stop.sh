#!/usr/bin/env bash
# Ferma lo stack avviato da claude-qwen-auto-v4.sh:
# proxy Anthropic, container ollama-admin e GPU Agent, modello e server Ollama.
#
#   ./claude-qwen-stop.sh                 # ferma tutto
#   ./claude-qwen-stop.sh --keep-ollama   # lascia su Ollama, scarica solo il modello
#
# La logica vive nello script principale (--stop): unica fonte di verita'.
set -Eeuo pipefail
exec "$(dirname "${BASH_SOURCE[0]}")/claude-qwen-auto-v4.sh" --stop "$@"
