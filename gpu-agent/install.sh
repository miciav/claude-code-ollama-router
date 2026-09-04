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
    run_as_user mkdir -p "$DIR"
    # main.py e' upstream e non lo modifichiamo: lo shim lo importa e basta.
    run_as_user curl -fsSL -o "$DIR/main.py" "$UPSTREAM"
    run_as_user cp "$HERE/gb10_agent.py" "$DIR/"
    run_as_user python3 -m venv "$DIR/.venv"
    run_as_user "$DIR/.venv/bin/pip" -q install --upgrade pip
    run_as_user "$DIR/.venv/bin/pip" -q install fastapi uvicorn
    echo "Shim installato in $DIR"
    echo "Self-check:"
    run_as_user "$DIR/.venv/bin/python" "$DIR/gb10_agent.py"
}

install_service() {
    [[ $EUID -eq 0 ]] || { echo "--service richiede root (usa sudo)" >&2; exit 1; }
    sed "s/%i/$TARGET_USER/g" "$HERE/gb10-gpu-agent.service" \
        > /etc/systemd/system/gb10-gpu-agent.service
    systemctl daemon-reload
    # Un'istanza avviata a mano terrebbe occupata la 11436.
    if [[ -s "$DIR/agent.pid" ]] && kill -0 "$(<"$DIR/agent.pid")" 2>/dev/null; then
        echo "Fermo l'istanza avviata a mano (PID $(<"$DIR/agent.pid"))"
        kill "$(<"$DIR/agent.pid")" || true
        sleep 2
        rm -f "$DIR/agent.pid"
    fi
    systemctl enable --now gb10-gpu-agent
    systemctl --no-pager status gb10-gpu-agent | head -5
}

# Sotto sudo lo shim va installato come utente finale, non come root:
# altrimenti venv e file finiscono di proprieta' di root e il service, che
# gira come User=$TARGET_USER, non puo' scriverci (__pycache__ compreso).
if [[ $EUID -eq 0 && "$TARGET_USER" != "root" ]]; then
    run_as_user() { runuser -u "$TARGET_USER" -- "$@"; }
else
    run_as_user() { "$@"; }
fi

if [[ -d "$DIR/.venv" ]]; then
    echo "Shim gia' presente in $DIR: aggiorno solo i file."
    run_as_user curl -fsSL -o "$DIR/main.py" "$UPSTREAM"
    run_as_user cp "$HERE/gb10_agent.py" "$DIR/"
else
    install_shim
fi

(( WITH_SERVICE )) && install_service
echo "Fatto."
