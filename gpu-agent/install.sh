#!/usr/bin/env bash
# Installa lo shim GB10 del GPU Agent e, opzionalmente, il service systemd.
#
#   ./install.sh              # solo venv + shim
#   sudo ./install.sh --service   # anche il service di sistema
set -Eeuo pipefail

TARGET_USER="${SUDO_USER:-$USER}"
DIR="${GPU_AGENT_DIR:-/home/$TARGET_USER/gb10-gpu-agent}"
UPSTREAM="https://raw.githubusercontent.com/ollama-admin/ollama-admin/main/gpu-agent/main.py"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

WITH_SERVICE=0
[[ "${1:-}" == "--service" ]] && WITH_SERVICE=1

install_shim() {
    mkdir -p "$DIR"
    # main.py e' upstream e non lo modifichiamo: lo shim lo importa e basta.
    curl -fsSL -o "$DIR/main.py" "$UPSTREAM"
    cp "$HERE/gb10_agent.py" "$DIR/"
    python3 -m venv "$DIR/.venv"
    "$DIR/.venv/bin/pip" -q install --upgrade pip
    "$DIR/.venv/bin/pip" -q install fastapi uvicorn
    echo "Shim installato in $DIR"
    echo "Self-check:"
    "$DIR/.venv/bin/python" "$DIR/gb10_agent.py"
}

install_service() {
    [[ $EUID -eq 0 ]] || { echo "--service richiede root (usa sudo)" >&2; exit 1; }
    sed "s/%i/$TARGET_USER/g" "$HERE/gb10-gpu-agent.service" \
        > /etc/systemd/system/gb10-gpu-agent.service
    systemctl daemon-reload
    systemctl enable --now gb10-gpu-agent
    systemctl --no-pager status gb10-gpu-agent | head -5
}

install_shim
(( WITH_SERVICE )) && install_service
echo "Fatto."
