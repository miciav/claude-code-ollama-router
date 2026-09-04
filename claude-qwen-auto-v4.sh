#!/usr/bin/env bash
set -Eeuo pipefail

# ============================================================================
# Claude Code + Ollama launcher for DGX Spark
#
# One physical model, multiple logical roles:
#
#   MAIN
#     qwen3.8 27B Q8
#     context: 256K
#     thinking: ENABLED
#     max output: 40K
#
#   AUTO MODE CLASSIFIER
#     same qwen3.8 27B runner
#     thinking: DISABLED
#     preserves Claude Code's small classifier budget, hard-clamped to 4096
#     temperature: 0
#
#   HAIKU / lightweight background requests
#     same qwen3.8 27B runner
#     thinking: DISABLED
#     max output: 40K global ceiling
#
# Ollama:
#   Flash Attention: ON
#   KV cache: q8_0
#   max loaded models: 1
#   parallel requests/model: 1
#   keep alive: infinite
#
# A local Anthropic-compatible proxy:
#   - detects Auto mode classifier requests from their prompt/shape
#   - rewrites classifier/Claude aliases to the local Qwen runtime model
#   - forces thinking OFF only for classifier + Haiku/background requests
#   - forces thinking ON for the main model and Sonnet/Opus auxiliary requests
#   - enforces output ceilings
#   - aborts clearly degenerate/repetitive streamed generations
#
# Usage:
#   ./claude-qwen-auto-v4.sh
#   ./claude-qwen-auto-v4.sh --continue
#   ./claude-qwen-auto-v4.sh --claude-only
#   ./claude-qwen-auto-v4.sh --claude-only --continue
#
# Overrides examples:
#   MAIN_BASE_MODEL=qwen3.8:27b-mlx ./claude-qwen-auto-v4.sh
#   MAIN_CONTEXT=131072 ./claude-qwen-auto-v4.sh
#   CLASSIFIER_MAX_OUTPUT_TOKENS=2048 ./claude-qwen-auto-v4.sh
#   PERMISSION_MODE=default ./claude-qwen-auto-v4.sh
#   MODEL_MENU=0 ./claude-qwen-auto-v4.sh          # salta il menu modelli
#   OLLAMA_ADMIN=1 ./claude-qwen-auto-v4.sh        # dashboard senza chiedere
#   OLLAMA_ADMIN=0 ./claude-qwen-auto-v4.sh        # mai la dashboard
#   OLLAMA_ADMIN_PORT=8080 ./claude-qwen-auto-v4.sh
#   GPU_AGENT=1 ./claude-qwen-auto-v4.sh           # telemetria GPU (watt/VRAM)
#   GPU_AGENT_PORT=11436 ./claude-qwen-auto-v4.sh  # 11435 e' del proxy Anthropic
# ============================================================================

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

# Remember whether the user pinned these from the environment: if so we skip the
# interactive model menu and keep their names verbatim.
MAIN_BASE_MODEL_PINNED="${MAIN_BASE_MODEL+1}"
MAIN_MODEL_PINNED="${MAIN_MODEL+1}"

MAIN_BASE_MODEL="${MAIN_BASE_MODEL:-qwen3.8:27b-mtp-q8_0}"
MAIN_CONTEXT="${MAIN_CONTEXT:-262144}"
# Derivato da MAIN_BASE_MODEL+MAIN_CONTEXT piu' sotto, se non imposto a mano:
# cosi' il nome non dipende dal fatto che il menu sia girato o no.
MAIN_MODEL="${MAIN_MODEL:-}"

# Interactive model menu: auto (only on a TTY), 1 (force), 0 (never).
MODEL_MENU="${MODEL_MENU:-auto}"

# ollama-admin dashboard (https://github.com/ollama-admin/ollama-admin).
# ask = chiedi conferma, 1 = sempre, 0 = mai.
OLLAMA_ADMIN="${OLLAMA_ADMIN:-ask}"
OLLAMA_ADMIN_PORT="${OLLAMA_ADMIN_PORT:-3000}"

# GPU Agent: telemetria hardware (VRAM, utilizzo, temperatura, watt per GPU).
# L'upstream usa 11435, che qui è già del proxy Anthropic: spostato su 11436.
GPU_AGENT="${GPU_AGENT:-ask}"
GPU_AGENT_PORT="${GPU_AGENT_PORT:-11436}"
# Su GB10/DGX Spark il container upstream non funziona: nvidia-smi restituisce
# [N/A] per la memoria e il loro parser solleva ValueError, perdendo anche i
# watt. Se questa directory contiene lo shim, lo preferiamo a Docker.
GPU_AGENT_DIR="${GPU_AGENT_DIR:-$HOME/gb10-gpu-agent}"
# Loopback per default: su macchine con IP pubblico e senza firewall, 0.0.0.0
# esporrebbe la telemetria a Internet. La dashboard gira in rete host e lo
# raggiunge comunque su 127.0.0.1.
GPU_AGENT_BIND="${GPU_AGENT_BIND:-127.0.0.1}"

# Database di ollama-admin: se presente e scrivibile, il proxy vi registra una
# riga per richiesta (token inclusi). Vuoto = funzione disattivata.
OA_DB="${OA_DB:-$HOME/ollama-admin-data/ollama-admin.db}"
OA_SERVER_ID="${OA_SERVER_ID:-}"

# Claude Code global output fuse for ordinary agent requests.
CLAUDE_CODE_MAX_OUTPUT_TOKENS="${CLAUDE_CODE_MAX_OUTPUT_TOKENS:-40000}"

# Auto mode currently uses tiny budgets (fast stage ~64, reasoning stage <=4096).
# We preserve smaller client requests and only clamp oversized classifier calls.
CLASSIFIER_MAX_OUTPUT_TOKENS="${CLASSIFIER_MAX_OUTPUT_TOKENS:-4096}"

# ---------------------------------------------------------------------------
# Ollama server settings
# ---------------------------------------------------------------------------

OLLAMA_API="${OLLAMA_API:-http://127.0.0.1:11434}"
OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1:11434}"
OLLAMA_CONTEXT_LENGTH="${OLLAMA_CONTEXT_LENGTH:-$MAIN_CONTEXT}"
OLLAMA_FLASH_ATTENTION="${OLLAMA_FLASH_ATTENTION:-1}"
OLLAMA_KV_CACHE_TYPE="${OLLAMA_KV_CACHE_TYPE:-q8_0}"
OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:--1}"
OLLAMA_MAX_LOADED_MODELS="${OLLAMA_MAX_LOADED_MODELS:-1}"
OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-1}"
OLLAMA_LOAD_TIMEOUT="${OLLAMA_LOAD_TIMEOUT:-10m}"

# ---------------------------------------------------------------------------
# Claude Code / proxy settings
# ---------------------------------------------------------------------------

# Auto mode is the point of this launcher. Override with PERMISSION_MODE=default
# if you explicitly want Claude Code's normal interactive permission mode.
PERMISSION_MODE="${PERMISSION_MODE:-auto}"

PROXY_HOST="${PROXY_HOST:-127.0.0.1}"
PROXY_PORT="${PROXY_PORT:-11435}"
PROXY_API="http://${PROXY_HOST}:${PROXY_PORT}"

# Conservative anti-degeneration watchdog.
WATCHDOG_ENABLED="${WATCHDOG_ENABLED:-1}"
WATCHDOG_MIN_REPEATED_CHARS="${WATCHDOG_MIN_REPEATED_CHARS:-4096}"
WATCHDOG_PROBE_CHARS="${WATCHDOG_PROBE_CHARS:-256}"
WATCHDOG_MIN_OCCURRENCES="${WATCHDOG_MIN_OCCURRENCES:-3}"
WATCHDOG_TAIL_CHARS="${WATCHDOG_TAIL_CHARS:-32768}"
WATCHDOG_CHAR_RUN="${WATCHDOG_CHAR_RUN:-1024}"

LOG_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/claude-qwen"
OLLAMA_LOG="$LOG_DIR/ollama.log"
OLLAMA_PID_FILE="$LOG_DIR/ollama.pid"
PROXY_LOG="$LOG_DIR/anthropic-proxy.log"
PROXY_PID_FILE="$LOG_DIR/anthropic-proxy.pid"
PROXY_SCRIPT="$LOG_DIR/anthropic_proxy.py"
MODELFILE_MAIN="$LOG_DIR/Modelfile.main"

mkdir -p "$LOG_DIR"

# ---------------------------------------------------------------------------
# Helpers / CLI argument parsing
# ---------------------------------------------------------------------------

die() {
    echo "ERROR: $*" >&2
    exit 1
}

need() {
    command -v "$1" >/dev/null 2>&1 || die "'$1' non trovato nel PATH"
}

need ollama
need curl
need python3

is_uint() {
    [[ "$1" =~ ^[0-9]+$ ]] && (( "$1" > 0 ))
}

is_uint "$MAIN_CONTEXT" || die "MAIN_CONTEXT deve essere un intero positivo"
is_uint "$CLAUDE_CODE_MAX_OUTPUT_TOKENS" || die "CLAUDE_CODE_MAX_OUTPUT_TOKENS deve essere un intero positivo"
is_uint "$CLASSIFIER_MAX_OUTPUT_TOKENS" || die "CLASSIFIER_MAX_OUTPUT_TOKENS deve essere un intero positivo"
[[ "$WATCHDOG_ENABLED" == "0" || "$WATCHDOG_ENABLED" == "1" ]] || die "WATCHDOG_ENABLED deve essere 0 oppure 1"
is_uint "$WATCHDOG_MIN_REPEATED_CHARS" || die "WATCHDOG_MIN_REPEATED_CHARS deve essere un intero positivo"
is_uint "$WATCHDOG_PROBE_CHARS" || die "WATCHDOG_PROBE_CHARS deve essere un intero positivo"
is_uint "$WATCHDOG_MIN_OCCURRENCES" || die "WATCHDOG_MIN_OCCURRENCES deve essere un intero positivo"
is_uint "$WATCHDOG_TAIL_CHARS" || die "WATCHDOG_TAIL_CHARS deve essere un intero positivo"
is_uint "$WATCHDOG_CHAR_RUN" || die "WATCHDOG_CHAR_RUN deve essere un intero positivo"
(( MAIN_CONTEXT > CLAUDE_CODE_MAX_OUTPUT_TOKENS )) || die "MAIN_CONTEXT deve essere > CLAUDE_CODE_MAX_OUTPUT_TOKENS"
(( WATCHDOG_TAIL_CHARS >= WATCHDOG_MIN_REPEATED_CHARS )) || die "WATCHDOG_TAIL_CHARS deve essere >= WATCHDOG_MIN_REPEATED_CHARS"
(( WATCHDOG_PROBE_CHARS < WATCHDOG_MIN_REPEATED_CHARS )) || die "WATCHDOG_PROBE_CHARS deve essere < WATCHDOG_MIN_REPEATED_CHARS"

is_uint "$GPU_AGENT_PORT" || die "GPU_AGENT_PORT deve essere un intero positivo"
is_uint "$OLLAMA_ADMIN_PORT" || die "OLLAMA_ADMIN_PORT deve essere un intero positivo"
# Il default upstream del GPU Agent (11435) coincide con il proxy Anthropic.
(( GPU_AGENT_PORT != PROXY_PORT )) \
    || die "GPU_AGENT_PORT ($GPU_AGENT_PORT) è la stessa porta del proxy Anthropic: usane un'altra (es. 11436)"
(( OLLAMA_ADMIN_PORT != PROXY_PORT )) \
    || die "OLLAMA_ADMIN_PORT ($OLLAMA_ADMIN_PORT) è la stessa porta del proxy Anthropic"
(( OLLAMA_ADMIN_PORT != GPU_AGENT_PORT )) \
    || die "OLLAMA_ADMIN_PORT e GPU_AGENT_PORT non possono coincidere"

CLAUDE_ONLY=0
DO_STOP=0
KEEP_OLLAMA=0
CLAUDE_ARGS=()
for arg in "$@"; do
    case "$arg" in
        --claude-only)
            CLAUDE_ONLY=1
            ;;
        --stop)
            DO_STOP=1
            ;;
        --keep-ollama)
            KEEP_OLLAMA=1
            ;;
        *)
            CLAUDE_ARGS+=("$arg")
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Interactive model menu
# ---------------------------------------------------------------------------

confirm() {
    # $1 = domanda, $2 = default (y|n)
    local answer
    read -r -p "$1 " answer </dev/tty || return 1
    answer="${answer:-$2}"
    [[ "$answer" =~ ^[yYsS] ]]
}

derive_runtime_name() {
    # qwen3.8:27b-mtp-q8_0 + 262144 -> qwen3.8-claude:256k
    local base="$1" ctx="$2" stem
    stem=$(printf '%s' "${base%%:*}" | tr -c 'a-zA-Z0-9._-' '-')
    printf '%s-claude:%sk' "$stem" "$(( ctx / 1024 ))"
}

# Nome di default del profilo runtime, ora che derive_runtime_name esiste.
[[ -n "$MAIN_MODEL" ]] || MAIN_MODEL="$(derive_runtime_name "$MAIN_BASE_MODEL" "$MAIN_CONTEXT")"

select_model() {
    [[ "$MODEL_MENU" == "0" ]] && return 0
    [[ -n "$MAIN_BASE_MODEL_PINNED" ]] && return 0
    if [[ "$MODEL_MENU" != "1" ]] && [[ ! -t 0 || ! -r /dev/tty ]]; then
        return 0
    fi

    local models=()
    # Escludiamo i profili runtime generati da questo script e dalle sue varianti:
    # *-claude:256k, ma anche *-claude-main:27b-256k e *-claude-fast:4b-128k.
    while IFS= read -r name; do
        [[ -n "$name" && "$name" != *-claude*:*k ]] && models+=("$name")
    done < <(ollama list 2>/dev/null | awk 'NR>1 {print $1}')

    if (( ${#models[@]} == 0 )); then
        echo "Nessun modello locale rilevato: uso il default $MAIN_BASE_MODEL"
        return 0
    fi

    echo
    echo "Modelli installati localmente:"
    local i
    for i in "${!models[@]}"; do
        printf '  %2d) %s\n' "$(( i + 1 ))" "${models[$i]}"
    done
    echo

    local choice
    read -r -p "Modello da usare [1-${#models[@]}, invio = $MAIN_BASE_MODEL]: " choice </dev/tty || choice=""
    if [[ -n "$choice" ]]; then
        is_uint "$choice" && (( choice <= ${#models[@]} )) \
            || die "scelta non valida: $choice"
        MAIN_BASE_MODEL="${models[$(( choice - 1 ))]}"
    fi

    [[ -z "$MAIN_MODEL_PINNED" ]] && MAIN_MODEL="$(derive_runtime_name "$MAIN_BASE_MODEL" "$MAIN_CONTEXT")"
    echo
}

# ---------------------------------------------------------------------------
# ollama-admin dashboard
# ---------------------------------------------------------------------------

ollama_admin_container() {
    # L'installer del GPU Agent crea container che iniziano anch'essi per
    # "ollama-admin": vanno esclusi o la dashboard "esiste già" per sbaglio.
    docker ps -a --format '{{.Names}}' 2>/dev/null \
        | grep '^ollama-admin' | grep -v 'gpu-agent' | head -1 || true
}

setup_ollama_admin() {
    [[ "$OLLAMA_ADMIN" == "0" ]] && return 0

    if [[ "$OLLAMA_ADMIN" != "1" ]]; then
        [[ -t 0 && -r /dev/tty ]] || return 0
        confirm "Avviare la dashboard ollama-admin su :$OLLAMA_ADMIN_PORT? [y/N]" n || return 0
    fi

    command -v docker >/dev/null 2>&1 || {
        echo "      docker non trovato: salto ollama-admin." >&2
        return 0
    }

    local container
    container=$(ollama_admin_container)

    if [[ -n "$container" ]]; then
        echo "      Container esistente: $container -> docker start"
        docker start "$container" >/dev/null || {
            echo "      Impossibile avviare $container." >&2
            return 0
        }
    else
        echo "      Installo ollama-admin (installer ufficiale, richiede rete)..."
        if [[ "$OLLAMA_ADMIN" != "1" ]]; then
            confirm "Eseguo 'curl ... install.sh | bash' da github.com/ollama-admin? [y/N]" n \
                || { echo "      Annullato."; return 0; }
        fi
        OLLAMA_ADMIN_PORT="$OLLAMA_ADMIN_PORT" \
            bash -c 'curl -fsSL https://raw.githubusercontent.com/ollama-admin/ollama-admin/main/scripts/install.sh | bash' \
            || { echo "      Installazione ollama-admin fallita: proseguo senza dashboard." >&2; return 0; }
    fi

    echo "      Dashboard: http://localhost:$OLLAMA_ADMIN_PORT"
    if [[ "$OLLAMA_HOST" == 127.0.0.1:* || "$OLLAMA_HOST" == localhost:* ]]; then
        echo "      NB: Ollama ascolta su $OLLAMA_HOST, non raggiungibile dal container."
        echo "          Riavvia con OLLAMA_HOST=0.0.0.0:11434 se la dashboard non vede i modelli."
    fi
    echo
}

setup_gpu_agent() {
    [[ "$GPU_AGENT" == "0" ]] && return 0

    if [[ "$GPU_AGENT" != "1" ]]; then
        [[ -t 0 && -r /dev/tty ]] || return 0
        confirm "Installare/avviare il GPU Agent (watt, VRAM, temperatura) su :$GPU_AGENT_PORT? [y/N]" n \
            || return 0
    fi

    command -v docker >/dev/null 2>&1 || {
        echo "      docker non trovato: salto il GPU Agent." >&2
        return 0
    }

    if curl -fsS --max-time 5 "http://127.0.0.1:$GPU_AGENT_PORT/health" >/dev/null 2>&1; then
        echo "      GPU Agent già attivo su :$GPU_AGENT_PORT"
    elif [[ -x "$GPU_AGENT_DIR/.venv/bin/uvicorn" ]]; then
        echo "      Avvio lo shim GB10 (memoria unificata da /proc/meminfo)..."
        ( cd "$GPU_AGENT_DIR" \
          && nohup ./.venv/bin/uvicorn gb10_agent:app \
                --host "$GPU_AGENT_BIND" --port "$GPU_AGENT_PORT" >agent.log 2>&1 &
          echo $! > "$GPU_AGENT_DIR/agent.pid" )
        local ok=0 _
        for _ in $(seq 1 20); do
            curl -fsS --max-time 5 "http://127.0.0.1:$GPU_AGENT_PORT/health" >/dev/null 2>&1 && { ok=1; break; }
            sleep 0.5
        done
        (( ok )) || {
            echo "      Shim GB10 non risponde; vedi $GPU_AGENT_DIR/agent.log" >&2
            return 0
        }
    else
        local container
        container=$(docker ps -a --format '{{.Names}}' 2>/dev/null | grep -m1 'gpu-agent' || true)

        if [[ -n "$container" ]]; then
            echo "      Container esistente: $container -> docker start"
            docker start "$container" >/dev/null \
                || { echo "      Impossibile avviare $container." >&2; return 0; }
        else
            if [[ "$GPU_AGENT" != "1" ]]; then
                confirm "Eseguo 'curl ... install-gpu-agent.sh | bash' da github.com/ollama-admin? [y/N]" n \
                    || { echo "      Annullato."; return 0; }
            fi
            echo "      Installo il GPU Agent (porta host $GPU_AGENT_PORT -> 11435 nel container)..."
            GPU_AGENT_PORT="$GPU_AGENT_PORT" \
                bash -c 'curl -fsSL https://raw.githubusercontent.com/ollama-admin/ollama-admin/main/scripts/install-gpu-agent.sh | bash' \
                || { echo "      Installazione GPU Agent fallita: proseguo senza telemetria." >&2; return 0; }
        fi

        local ok=0 _
        for _ in $(seq 1 20); do
            if curl -fsS --max-time 5 "http://127.0.0.1:$GPU_AGENT_PORT/health" >/dev/null 2>&1; then
                ok=1
                break
            fi
            sleep 1
        done
        (( ok )) || {
            echo "      GPU Agent non risponde su :$GPU_AGENT_PORT (health check fallito)." >&2
            return 0
        }
    fi

    echo "      GPU Agent: http://localhost:$GPU_AGENT_PORT"
    echo "      Passo manuale: nella dashboard, Admin -> Servers -> GPU Agent URL"
    echo "                     = http://127.0.0.1:$GPU_AGENT_PORT"
    echo "                     (la dashboard gira in rete host: vede il loopback)"
    echo
}

print_config() {
    echo "================================================================"
    echo " Claude Code + Ollama — one Qwen, Auto mode classifier routing"
    echo "================================================================"
    echo " Base model          : $MAIN_BASE_MODEL"
    echo " Runtime model       : $MAIN_MODEL"
    echo " Context             : $MAIN_CONTEXT"
    echo " Main thinking       : ENABLED"
    echo " Main output cap     : $CLAUDE_CODE_MAX_OUTPUT_TOKENS"
    echo " Classifier thinking : DISABLED"
    echo " Classifier cap      : $CLASSIFIER_MAX_OUTPUT_TOKENS"
    echo " Permission mode     : $PERMISSION_MODE"
    echo
    echo " Flash Attention     : $OLLAMA_FLASH_ATTENTION"
    echo " KV cache            : $OLLAMA_KV_CACHE_TYPE"
    echo " Keep alive          : $OLLAMA_KEEP_ALIVE"
    echo " Max loaded models   : $OLLAMA_MAX_LOADED_MODELS"
    echo " Parallel/model      : $OLLAMA_NUM_PARALLEL"
    echo " Anthropic endpoint  : $PROXY_API"
    echo " Watchdog            : $WATCHDOG_ENABLED"
    echo "================================================================"
    echo
}

# ---------------------------------------------------------------------------
# Ollama lifecycle
# ---------------------------------------------------------------------------

ollama_is_ready() {
    curl -fsS "$OLLAMA_API/api/version" >/dev/null 2>&1
}

start_ollama() {
    echo "[1/6] Avvio Ollama..."

    if command -v systemctl >/dev/null 2>&1 && systemctl cat ollama >/dev/null 2>&1; then
        for value in \
            "$OLLAMA_HOST" "$OLLAMA_KEEP_ALIVE" "$OLLAMA_MAX_LOADED_MODELS" \
            "$OLLAMA_CONTEXT_LENGTH" "$OLLAMA_FLASH_ATTENTION" "$OLLAMA_KV_CACHE_TYPE" \
            "$OLLAMA_NUM_PARALLEL" "$OLLAMA_LOAD_TIMEOUT"; do
            [[ $value != *$'\n'* && $value != *$'\r'* && $value != *'"'* && $value != *'\\'* ]] \
                || die "valore Ollama non valido per systemd"
        done

        sudo mkdir -p /etc/systemd/system/ollama.service.d
        {
            echo '[Service]'
            printf 'Environment="OLLAMA_HOST=%s"\n' "$OLLAMA_HOST"
            printf 'Environment="OLLAMA_KEEP_ALIVE=%s"\n' "$OLLAMA_KEEP_ALIVE"
            printf 'Environment="OLLAMA_MAX_LOADED_MODELS=%s"\n' "$OLLAMA_MAX_LOADED_MODELS"
            printf 'Environment="OLLAMA_CONTEXT_LENGTH=%s"\n' "$OLLAMA_CONTEXT_LENGTH"
            printf 'Environment="OLLAMA_FLASH_ATTENTION=%s"\n' "$OLLAMA_FLASH_ATTENTION"
            printf 'Environment="OLLAMA_KV_CACHE_TYPE=%s"\n' "$OLLAMA_KV_CACHE_TYPE"
            printf 'Environment="OLLAMA_NUM_PARALLEL=%s"\n' "$OLLAMA_NUM_PARALLEL"
            printf 'Environment="OLLAMA_LOAD_TIMEOUT=%s"\n' "$OLLAMA_LOAD_TIMEOUT"
        } | sudo tee /etc/systemd/system/ollama.service.d/claude-qwen.conf >/dev/null

        sudo systemctl daemon-reload
        sudo systemctl start ollama

        for _ in $(seq 1 60); do
            if ollama_is_ready; then
                echo "      Servizio Ollama avviato."
                return 0
            fi
            if ! systemctl is-active --quiet ollama; then
                sudo journalctl -u ollama -n 50 --no-pager >&2 || true
                die "il servizio Ollama non è riuscito ad avviarsi"
            fi
            sleep 1
        done
        die "timeout durante l'avvio del servizio Ollama"
    fi

    OLLAMA_HOST="$OLLAMA_HOST" \
    OLLAMA_KEEP_ALIVE="$OLLAMA_KEEP_ALIVE" \
    OLLAMA_MAX_LOADED_MODELS="$OLLAMA_MAX_LOADED_MODELS" \
    OLLAMA_CONTEXT_LENGTH="$OLLAMA_CONTEXT_LENGTH" \
    OLLAMA_FLASH_ATTENTION="$OLLAMA_FLASH_ATTENTION" \
    OLLAMA_KV_CACHE_TYPE="$OLLAMA_KV_CACHE_TYPE" \
    OLLAMA_NUM_PARALLEL="$OLLAMA_NUM_PARALLEL" \
    OLLAMA_LOAD_TIMEOUT="$OLLAMA_LOAD_TIMEOUT" \
        nohup ollama serve >"$OLLAMA_LOG" 2>&1 &

    local pid=$!
    echo "$pid" > "$OLLAMA_PID_FILE"

    for _ in $(seq 1 60); do
        if ollama_is_ready; then
            echo "      Ollama avviato (PID $pid)."
            return 0
        fi
        if ! kill -0 "$pid" 2>/dev/null; then
            tail -50 "$OLLAMA_LOG" >&2 || true
            die "Ollama non è riuscito ad avviarsi"
        fi
        sleep 1
    done

    die "timeout durante l'avvio di Ollama. Vedi $OLLAMA_LOG"
}

stop_ollama() {
    ollama_is_ready || return 0

    echo "[1/6] Arresto il server Ollama esistente..."
    if command -v systemctl >/dev/null 2>&1 && systemctl is-active --quiet ollama; then
        sudo systemctl stop ollama
    elif command -v systemctl >/dev/null 2>&1 && systemctl --user is-active --quiet ollama; then
        systemctl --user stop ollama
    elif [[ -s "$OLLAMA_PID_FILE" ]]; then
        local pid
        pid=$(<"$OLLAMA_PID_FILE")
        if kill -0 "$pid" 2>/dev/null && [[ $(ps -p "$pid" -o args=) == *"ollama serve"* ]]; then
            kill "$pid"
        else
            die "PID file Ollama non valido: $OLLAMA_PID_FILE"
        fi
    else
        die "Ollama è attivo ma non è gestito da systemd o da questo script"
    fi

    for _ in $(seq 1 30); do
        ollama_is_ready || {
            rm -f "$OLLAMA_PID_FILE"
            return 0
        }
        sleep 1
    done
    die "timeout durante l'arresto di Ollama"
}

# ---------------------------------------------------------------------------
# Runtime model
# ---------------------------------------------------------------------------

ensure_model() {
    local model="$1"
    if ollama show "$model" >/dev/null 2>&1; then
        echo "      $model: presente"
    else
        echo "      $model: non presente -> ollama pull"
        ollama pull "$model"
    fi
}

create_runtime_model() {
    echo "[3/6] Creo/aggiorno il profilo runtime..."

    # Sampling is NOT baked here: classifier wants temperature=0 while main
    # Qwen uses its thinking-mode sampling profile. The proxy sets sampling
    # per request; the runner-relevant parameters remain identical.
    cat >"$MODELFILE_MAIN" <<EOF_MODEL
FROM $MAIN_BASE_MODEL
PARAMETER num_ctx $MAIN_CONTEXT
PARAMETER num_predict $CLAUDE_CODE_MAX_OUTPUT_TOKENS
PARAMETER repeat_penalty 1.0
EOF_MODEL

    echo "      $MAIN_MODEL (num_ctx=$MAIN_CONTEXT)"
    ollama create "$MAIN_MODEL" -f "$MODELFILE_MAIN"
}

preload_model() {
    echo "[4/6] Carico e mantengo residente il modello..."
    # Use /api/chat rather than /api/generate: current Ollama handles thinking
    # control more consistently on chat-compatible paths.
    curl -fsS \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"$MAIN_MODEL\",\"messages\":[],\"keep_alive\":-1,\"stream\":false}" \
        "$OLLAMA_API/api/chat" \
        >/dev/null

    echo "      Modello residente:"
    ollama ps
    echo
}

# ---------------------------------------------------------------------------
# Anthropic guard/routing proxy
# ---------------------------------------------------------------------------

write_proxy() {
    cat >"$PROXY_SCRIPT" <<'PY'
#!/usr/bin/env python3
import http.client
import json
import os
import sqlite3
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

UPSTREAM = urlsplit(os.environ["OLLAMA_UPSTREAM"])
LISTEN_HOST = os.environ["PROXY_HOST"]
LISTEN_PORT = int(os.environ["PROXY_PORT"])
MAIN_MODEL = os.environ["MAIN_MODEL"]
MAIN_MAX_OUTPUT = int(os.environ["MAIN_MAX_OUTPUT_TOKENS"])
CLASSIFIER_MAX_OUTPUT = int(os.environ["CLASSIFIER_MAX_OUTPUT_TOKENS"])

# Scrittura opzionale nel database di ollama-admin. Il loro proxy non conta i
# token (registra prima di leggere il corpo della risposta), quindi la riga la
# scrive chi quel dato ce l'ha davvero: noi, a stream concluso.
OA_DB = os.environ.get("OA_DB", "")
OA_SERVER_ID = os.environ.get("OA_SERVER_ID", "")

WATCHDOG_ENABLED = os.environ.get("WATCHDOG_ENABLED", "1") == "1"
WATCHDOG_MIN_REPEATED_CHARS = int(os.environ.get("WATCHDOG_MIN_REPEATED_CHARS", "4096"))
WATCHDOG_PROBE_CHARS = int(os.environ.get("WATCHDOG_PROBE_CHARS", "256"))
WATCHDOG_MIN_OCCURRENCES = int(os.environ.get("WATCHDOG_MIN_OCCURRENCES", "3"))
WATCHDOG_TAIL_CHARS = int(os.environ.get("WATCHDOG_TAIL_CHARS", "32768"))
WATCHDOG_CHAR_RUN = int(os.environ.get("WATCHDOG_CHAR_RUN", "1024"))

HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade"
}

MAIN_SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20}
BACKGROUND_SAMPLING = {"temperature": 0.7, "top_p": 0.80, "top_k": 20}


_DB_LOCK = threading.Lock()


def record_log(model, endpoint, prompt_tokens, completion_tokens, latency_ms, status_code):
    """Inserisce una riga nella tabella Log di ollama-admin.

    Best effort e fuori dal percorso della risposta: la dashboard e' un extra,
    un suo problema non deve mai propagarsi a Claude Code. Ci accoppia allo
    schema di un progetto di terze parti, quindi ogni errore viene loggato e
    ingoiato: se rinominano una colonna, lo si vede nel log del proxy.
    """
    if not OA_DB or not OA_SERVER_ID:
        return

    def write():
        try:
            with _DB_LOCK:
                conn = sqlite3.connect(OA_DB, timeout=5)
                try:
                    conn.execute("PRAGMA busy_timeout=5000")
                    conn.execute(
                        'INSERT INTO "Log" ("id","serverId","model","endpoint",'
                        '"promptTokens","completionTokens","latencyMs","statusCode","createdAt")'
                        " VALUES (?,?,?,?,?,?,?,?,?)",
                        (
                            uuid.uuid4().hex,
                            OA_SERVER_ID,
                            model,
                            endpoint,
                            prompt_tokens or None,
                            completion_tokens or None,
                            int(latency_ms),
                            int(status_code),
                            int(time.time() * 1000),  # Prisma/SQLite: ms epoch
                        ),
                    )
                    conn.commit()
                finally:
                    conn.close()
        except Exception as exc:
            sys.stderr.write("LOG_WRITE_FAILED %s\n" % exc)
            sys.stderr.flush()

    threading.Thread(target=write, daemon=True).start()


def iter_strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from iter_strings(v)
    elif isinstance(value, list):
        for v in value:
            yield from iter_strings(v)


def is_auto_classifier_request(data):
    """Detect Claude Code Auto mode classifier requests without relying on one model ID."""
    stop_sequences = data.get("stop_sequences") or []
    if any(isinstance(s, str) and "</block>" in s.lower() for s in stop_sequences):
        return True

    text = "\n".join(iter_strings({
        "system": data.get("system"),
        "messages": data.get("messages"),
    })).lower()

    strong_signatures = (
        "security monitor for autonomous ai coding agents",
        "auto mode classifier",
        "classifier stage 1",
        "classifier stage 2",
        "err on the side of blocking",
    )
    if any(sig in text for sig in strong_signatures):
        return True

    # Both XML verdicts are distinctive in the Auto mode output contract.
    if "<block>yes</block>" in text and "<block>no</block>" in text:
        return True

    # Fallback for rollout/version changes: current external-session classifier
    # model families + classifier-like request shape. This is deliberately
    # conservative so an ordinary long Claude request is not misclassified.
    model = str(data.get("model") or "").lower()
    known_classifier_family = any(x in model for x in (
        "claude-sonnet-5",
        "claude-fable-5",
        "claude-haiku-4-5",
    ))
    max_tokens = data.get("max_tokens")
    temperature = data.get("temperature")
    if (
        known_classifier_family
        and isinstance(max_tokens, int)
        and max_tokens <= 8192
        and temperature in (0, 0.0)
    ):
        return True

    return False


def clamp_max_tokens(data, ceiling):
    requested = data.get("max_tokens")
    if isinstance(requested, int) and requested > 0:
        data["max_tokens"] = min(requested, ceiling)
    else:
        data["max_tokens"] = ceiling


def classify_and_rewrite(data):
    request_model = str(data.get("model") or "")
    model_lower = request_model.lower()

    if is_auto_classifier_request(data):
        route = "AUTO_CLASSIFIER"
        data["model"] = MAIN_MODEL
        clamp_max_tokens(data, CLASSIFIER_MAX_OUTPUT)
        data["thinking"] = {"type": "disabled"}
        data.pop("output_config", None)
        # Claude Code's classifier is intentionally deterministic.
        data["temperature"] = 0.0
        return route, request_model

    if request_model == MAIN_MODEL:
        route = "MAIN"
        clamp_max_tokens(data, MAIN_MAX_OUTPUT)
        data["thinking"] = {"type": "enabled"}
        data.pop("output_config", None)
        data.update(MAIN_SAMPLING)
        return route, request_model

    # Internal Haiku/background calls use the same physical Qwen runner but
    # without reasoning. This is not the Auto mode classifier path.
    if "haiku" in model_lower:
        route = "BACKGROUND_HAIKU"
        data["model"] = MAIN_MODEL
        clamp_max_tokens(data, MAIN_MAX_OUTPUT)
        data["thinking"] = {"type": "disabled"}
        data.pop("output_config", None)
        data.update(BACKGROUND_SAMPLING)
        return route, request_model

    # If Claude Code asks for another Claude family internally (Sonnet/Opus/
    # Fable), serve it from the same local model with normal thinking enabled.
    if model_lower.startswith("claude-") or model_lower in {"sonnet", "opus", "fable"}:
        route = "AUX_CLAUDE"
        data["model"] = MAIN_MODEL
        clamp_max_tokens(data, MAIN_MAX_OUTPUT)
        data["thinking"] = {"type": "enabled"}
        data.pop("output_config", None)
        data.update(MAIN_SAMPLING)
        return route, request_model

    # Unknown/custom models are forwarded unchanged, but keep the global fuse.
    route = "PASSTHROUGH"
    clamp_max_tokens(data, MAIN_MAX_OUTPUT)
    return route, request_model


class RepetitionWatchdog:
    """Conservative detector for sustained exact periodic/repetitive output."""

    def __init__(self):
        self.tail = ""

    def feed(self, text):
        if not WATCHDOG_ENABLED or not text:
            return None

        self.tail = (self.tail + text)[-WATCHDOG_TAIL_CHARS:]
        tail = self.tail

        if len(tail) >= WATCHDOG_CHAR_RUN:
            c = tail[-1]
            run = 1
            i = len(tail) - 2
            while i >= 0 and tail[i] == c and run < WATCHDOG_CHAR_RUN:
                run += 1
                i -= 1
            if run >= WATCHDOG_CHAR_RUN:
                return f"single character {c!r} repeated >= {WATCHDOG_CHAR_RUN} times"

        if len(tail) < WATCHDOG_MIN_REPEATED_CHARS or len(tail) < WATCHDOG_PROBE_CHARS:
            return None

        probe = tail[-WATCHDOG_PROBE_CHARS:]
        positions = []
        start = 0
        while True:
            pos = tail.find(probe, start)
            if pos < 0:
                break
            positions.append(pos)
            if len(positions) > 512:
                positions.pop(0)
            start = pos + 1

        if len(positions) < WATCHDOG_MIN_OCCURRENCES:
            return None

        latest = positions[-1]
        period = latest - positions[-2]
        if period <= 0:
            return None

        run_positions = [latest, positions[-2]]
        expected = positions[-2] - period
        for pos in reversed(positions[:-2]):
            if pos == expected:
                run_positions.append(pos)
                expected -= period
            elif pos < expected:
                break

        occurrences = len(run_positions)
        repeated_span = latest - run_positions[-1] + WATCHDOG_PROBE_CHARS
        if occurrences >= WATCHDOG_MIN_OCCURRENCES and repeated_span >= WATCHDOG_MIN_REPEATED_CHARS:
            return (
                f"stable repeated pattern detected: period={period} chars, "
                f"occurrences={occurrences}, span={repeated_span} chars"
            )

        return None


def iter_sse_events(resp):
    lines = []
    while True:
        line = resp.readline()
        if not line:
            if lines:
                yield b"".join(lines)
            return
        lines.append(line)
        if line in (b"\n", b"\r\n"):
            yield b"".join(lines)
            lines = []


def event_obj(event_bytes):
    data_lines = []
    for line in event_bytes.splitlines():
        if line.startswith(b"data:"):
            data_lines.append(line[5:].lstrip())
    if not data_lines:
        return None

    try:
        return json.loads(b"\n".join(data_lines))
    except Exception:
        return None


def obj_delta_text(obj):
    if obj.get("type") != "content_block_delta":
        return ""
    delta = obj.get("delta") or {}
    for key in ("text", "thinking", "partial_json"):
        value = delta.get(key)
        if isinstance(value, str):
            return value
    return ""


class Proxy(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "ClaudeQwenAutoGuard/4.0"

    def log_message(self, fmt, *args):
        sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), fmt % args))
        sys.stderr.flush()

    def _emit_stream_error(self, message):
        payload = json.dumps({
            "type": "error",
            "error": {"type": "api_error", "message": message},
        }, separators=(",", ":"))
        frame = f"event: error\ndata: {payload}\n\n".encode("utf-8")
        try:
            self.wfile.write(frame)
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _proxy(self):
        content_length = int(self.headers.get("Content-Length", "0") or "0")
        body = self.rfile.read(content_length) if content_length else b""

        route = "RAW"
        request_model = None
        stream_requested = False
        guarded_messages = False

        if self.path.startswith("/v1/messages") and body:
            try:
                data = json.loads(body)
                stream_requested = bool(data.get("stream", False))
                route, request_model = classify_and_rewrite(data)
                guarded_messages = True

                body = json.dumps(data, separators=(",", ":")).encode("utf-8")
                self.log_message(
                    "ROUTE=%s request_model=%s upstream_model=%s thinking=%s max_tokens=%s temp=%s top_p=%s top_k=%s",
                    route,
                    request_model,
                    data.get("model"),
                    (data.get("thinking") or {}).get("type") if isinstance(data.get("thinking"), dict) else data.get("thinking"),
                    data.get("max_tokens"),
                    data.get("temperature"),
                    data.get("top_p"),
                    data.get("top_k"),
                )
            except Exception as exc:
                self.send_error(400, "invalid JSON request: %s" % exc)
                return

        conn = http.client.HTTPConnection(UPSTREAM.hostname, UPSTREAM.port, timeout=3600)
        headers = {}
        for key, value in self.headers.items():
            lk = key.lower()
            if lk in HOP_BY_HOP or lk in {"host", "content-length"}:
                continue
            headers[key] = value
        headers["Host"] = f"{UPSTREAM.hostname}:{UPSTREAM.port}"
        if body:
            headers["Content-Length"] = str(len(body))

        try:
            conn.request(self.command, self.path, body=body if body else None, headers=headers)
            resp = conn.getresponse()

            self.send_response(resp.status, resp.reason)
            for key, value in resp.getheaders():
                lk = key.lower()
                if lk in HOP_BY_HOP or lk in {"content-length", "connection"}:
                    continue
                self.send_header(key, value)
            self.send_header("Connection", "close")
            self.send_header("X-Claude-Qwen-Route", route)
            self.end_headers()

            content_type = (resp.getheader("Content-Type") or "").lower()
            # We now walk the SSE stream for every guarded /v1/messages request,
            # not only when the watchdog is on: the same pass yields throughput.
            # RepetitionWatchdog.feed() is already a no-op when disabled.
            observe_sse = (
                guarded_messages
                and stream_requested
                and "text/event-stream" in content_type
                and 200 <= resp.status < 300
            )

            if observe_sse:
                watchdog = RepetitionWatchdog()
                started = time.monotonic()
                first_token_at = None
                in_tokens = 0
                out_tokens = 0
                out_chars = 0
                aborted = False

                for event in iter_sse_events(resp):
                    obj = event_obj(event)
                    generated = ""
                    if obj is not None:
                        kind = obj.get("type")
                        if kind == "message_start":
                            usage = ((obj.get("message") or {}).get("usage")) or {}
                            in_tokens = usage.get("input_tokens") or in_tokens
                        elif kind == "message_delta":
                            # Anthropic reports cumulative output_tokens here.
                            usage = obj.get("usage") or {}
                            out_tokens = usage.get("output_tokens") or out_tokens
                        generated = obj_delta_text(obj)

                    if generated:
                        out_chars += len(generated)
                        if first_token_at is None:
                            first_token_at = time.monotonic()

                    reason = watchdog.feed(generated)
                    if reason:
                        aborted = True
                        self.log_message(
                            "WATCHDOG_ABORT route=%s model=%s reason=%s",
                            route,
                            request_model,
                            reason,
                        )
                        conn.close()
                        self._emit_stream_error(
                            "Local generation watchdog stopped a repetitive/degenerate "
                            f"response on route {route}: {reason}"
                        )
                        break
                    self.wfile.write(event)
                    self.wfile.flush()

                now = time.monotonic()
                # Generation rate excludes prefill: it is measured from the first
                # emitted token, so a long prompt does not depress tokens/s.
                gen_seconds = (now - first_token_at) if first_token_at else 0.0
                rate = (out_tokens / gen_seconds) if (out_tokens and gen_seconds > 0) else 0.0
                self.log_message(
                    "STATS route=%s model=%s in_tokens=%s out_tokens=%s out_chars=%s "
                    "ttft_s=%s gen_tok_s=%s total_s=%.2f%s",
                    route,
                    request_model,
                    in_tokens,
                    out_tokens,
                    out_chars,
                    ("%.2f" % (first_token_at - started)) if first_token_at else "n/a",
                    ("%.1f" % rate) if rate else "n/a",
                    now - started,
                    " ABORTED" if aborted else "",
                )
                record_log(
                    model=str(request_model or route),
                    endpoint=self.path,
                    prompt_tokens=in_tokens,
                    completion_tokens=out_tokens,
                    latency_ms=(now - started) * 1000,
                    status_code=resp.status,
                )
            else:
                while True:
                    chunk = resp.read1(65536)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    self.wfile.flush()

        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:
            try:
                self.send_error(502, "upstream error: %s" % exc)
            except Exception:
                pass
        finally:
            conn.close()
            self.close_connection = True

    do_GET = _proxy
    do_POST = _proxy
    do_PUT = _proxy
    do_DELETE = _proxy
    do_PATCH = _proxy


if __name__ == "__main__":
    server = ThreadingHTTPServer((LISTEN_HOST, LISTEN_PORT), Proxy)
    print(
        f"proxy http://{LISTEN_HOST}:{LISTEN_PORT} -> "
        f"{UPSTREAM.scheme}://{UPSTREAM.hostname}:{UPSTREAM.port}; "
        f"MAIN={MAIN_MODEL} think=ON max={MAIN_MAX_OUTPUT}; "
        f"AUTO_CLASSIFIER think=OFF max<={CLASSIFIER_MAX_OUTPUT}; "
        f"watchdog={'ON' if WATCHDOG_ENABLED else 'OFF'}",
        flush=True,
    )
    server.serve_forever()
PY

    chmod 700 "$PROXY_SCRIPT"
    python3 -m py_compile "$PROXY_SCRIPT" || die "proxy Python non valido"
}

proxy_is_ready() {
    curl -fsS "$PROXY_API/api/version" >/dev/null 2>&1
}

stop_proxy() {
    if [[ -s "$PROXY_PID_FILE" ]]; then
        local pid
        pid=$(<"$PROXY_PID_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            if [[ $(ps -p "$pid" -o args=) == *"anthropic_proxy.py"* ]]; then
                kill "$pid" || true
                for _ in $(seq 1 20); do
                    kill -0 "$pid" 2>/dev/null || break
                    sleep 0.1
                done
            fi
        fi
        rm -f "$PROXY_PID_FILE"
    fi

    if proxy_is_ready; then
        die "la porta proxy $PROXY_HOST:$PROXY_PORT è occupata da un processo non gestito da questo script"
    fi
}

detect_oa_server() {
    # Nessun DB o nessun permesso di scrittura: la funzione resta spenta.
    [[ -n "$OA_DB" && -w "$OA_DB" ]] || { OA_DB=""; return 0; }
    [[ -z "$OA_SERVER_ID" ]] || return 0

    OA_SERVER_ID=$(python3 - "$OA_DB" <<'PY' 2>/dev/null || true
import sqlite3, sys
try:
    conn = sqlite3.connect(sys.argv[1], timeout=5)
    row = conn.execute('SELECT id FROM "Server" ORDER BY "createdAt" LIMIT 1').fetchone()
    print(row[0] if row else "")
except Exception:
    pass
PY
)
    if [[ -n "$OA_SERVER_ID" ]]; then
        echo "      Metriche -> ollama-admin (server $OA_SERVER_ID)"
    else
        OA_DB=""
        echo "      ollama-admin: nessun server configurato, metriche disattivate"
    fi
}

start_proxy() {
    echo "[5/6] Avvio il proxy Anthropic/Auto-mode..."
    detect_oa_server
    stop_proxy
    write_proxy

    OLLAMA_UPSTREAM="$OLLAMA_API" \
    PROXY_HOST="$PROXY_HOST" \
    PROXY_PORT="$PROXY_PORT" \
    MAIN_MODEL="$MAIN_MODEL" \
    MAIN_MAX_OUTPUT_TOKENS="$CLAUDE_CODE_MAX_OUTPUT_TOKENS" \
    CLASSIFIER_MAX_OUTPUT_TOKENS="$CLASSIFIER_MAX_OUTPUT_TOKENS" \
    WATCHDOG_ENABLED="$WATCHDOG_ENABLED" \
    WATCHDOG_MIN_REPEATED_CHARS="$WATCHDOG_MIN_REPEATED_CHARS" \
    WATCHDOG_PROBE_CHARS="$WATCHDOG_PROBE_CHARS" \
    WATCHDOG_MIN_OCCURRENCES="$WATCHDOG_MIN_OCCURRENCES" \
    WATCHDOG_TAIL_CHARS="$WATCHDOG_TAIL_CHARS" \
    WATCHDOG_CHAR_RUN="$WATCHDOG_CHAR_RUN" \
    OA_DB="$OA_DB" \
    OA_SERVER_ID="$OA_SERVER_ID" \
        nohup python3 "$PROXY_SCRIPT" >"$PROXY_LOG" 2>&1 &

    local pid=$!
    echo "$pid" > "$PROXY_PID_FILE"

    for _ in $(seq 1 50); do
        if proxy_is_ready; then
            echo "      Proxy avviato (PID $pid): $PROXY_API"
            echo "      Log: $PROXY_LOG"
            return 0
        fi
        if ! kill -0 "$pid" 2>/dev/null; then
            tail -50 "$PROXY_LOG" >&2 || true
            die "il proxy Anthropic non è riuscito ad avviarsi"
        fi
        sleep 0.1
    done

    tail -50 "$PROXY_LOG" >&2 || true
    die "timeout durante l'avvio del proxy Anthropic"
}

# A cheap synthetic routing test. The body resembles Auto mode's XML contract;
# the proxy should report AUTO_CLASSIFIER and force no-thinking. This also
# verifies Ollama accepts thinking=disabled for the runtime model.
verify_classifier_route() {
    echo "[6/6] Verifico il routing Auto-mode/classifier..."

    local headers_file="$LOG_DIR/classifier-check.headers"
    local body_file="$LOG_DIR/classifier-check.json"

    curl -fsS -D "$headers_file" -o "$body_file" \
        "$PROXY_API/v1/messages" \
        -H 'Content-Type: application/json' \
        -H 'x-api-key: ollama' \
        -d "{\
          \"model\":\"claude-sonnet-5\",\
          \"max_tokens\":64,\
          \"temperature\":0,\
          \"stream\":false,\
          \"system\":\"You are a security monitor for autonomous AI coding agents.\",\
          \"messages\":[{\"role\":\"user\",\"content\":\"Reply only <block>no</block>.\"}]\
        }" \
        || die "self-test classifier fallito; vedi $PROXY_LOG"

    grep -qi '^X-Claude-Qwen-Route: AUTO_CLASSIFIER' "$headers_file" \
        || die "il proxy non ha riconosciuto la richiesta classifier; vedi $headers_file"

    python3 - "$body_file" <<'PY'
import json, sys
with open(sys.argv[1], "r", encoding="utf-8") as f:
    data = json.load(f)
for block in data.get("content", []):
    if isinstance(block, dict) and block.get("type") == "thinking" and block.get("thinking"):
        print("ERROR: il classifier ha prodotto thinking nonostante thinking=disabled", file=sys.stderr)
        sys.exit(1)
PY

    echo "      OK: classifier -> stesso Qwen, thinking OFF, budget <= $CLASSIFIER_MAX_OUTPUT_TOKENS."
}

# ---------------------------------------------------------------------------
# Claude Code environment / launch
# ---------------------------------------------------------------------------

configure_claude_env() {
    export ANTHROPIC_AUTH_TOKEN="ollama"
    export ANTHROPIC_API_KEY=""
    export ANTHROPIC_BASE_URL="$PROXY_API"

    # Main session model. We deliberately do NOT override Anthropic's
    # OPUS/SONNET/HAIKU default aliases here: keeping their original logical
    # names visible lets the proxy distinguish auxiliary/background traffic.
    export ANTHROPIC_MODEL="$MAIN_MODEL"
    export ANTHROPIC_DEFAULT_MODEL="$MAIN_MODEL"
    unset ANTHROPIC_DEFAULT_OPUS_MODEL 2>/dev/null || true
    unset ANTHROPIC_DEFAULT_SONNET_MODEL 2>/dev/null || true
    unset ANTHROPIC_DEFAULT_HAIKU_MODEL 2>/dev/null || true

    export CLAUDE_CODE_SUBAGENT_MODEL="$MAIN_MODEL"
    export CLAUDE_CODE_MAX_OUTPUT_TOKENS

    # Do not use MAX_THINKING_TOKENS=0: it is global. Thinking is controlled
    # per request by the proxy.
    unset MAX_THINKING_TOKENS 2>/dev/null || true

    # Qwen's real context size lives in the Ollama runtime model.
    export CLAUDE_CODE_DISABLE_UNKNOWN_MODEL_WINDOW_ENFORCEMENT="${CLAUDE_CODE_DISABLE_UNKNOWN_MODEL_WINDOW_ENFORCEMENT:-1}"
}

launch_claude() {
    need claude
    configure_claude_env

    echo
    echo "Routing attivo:"
    echo "  MAIN             -> $MAIN_MODEL | thinking ON  | max $CLAUDE_CODE_MAX_OUTPUT_TOKENS"
    echo "  AUTO CLASSIFIER  -> $MAIN_MODEL | thinking OFF | max <= $CLASSIFIER_MAX_OUTPUT_TOKENS"
    echo "  Haiku/background -> $MAIN_MODEL | thinking OFF | same physical runner"
    echo "  Sonnet/Opus aux  -> $MAIN_MODEL | thinking ON  | same physical runner"
    echo
    echo "Proxy log: $PROXY_LOG"
    echo "Monitor:   tail -f \"$PROXY_LOG\""
    echo
    echo "Avvio Claude Code..."
    echo

    exec claude \
        --model "$MAIN_MODEL" \
        --permission-mode "$PERMISSION_MODE" \
        "${CLAUDE_ARGS[@]}"
}

# ---------------------------------------------------------------------------
# Teardown (--stop)
# ---------------------------------------------------------------------------

gpu_agent_container() {
    docker ps -a --format '{{.Names}}' 2>/dev/null | grep -m1 'gpu-agent' || true
}

stop_container() {
    # $1 = nome container, $2 = etichetta
    local name="$1" label="$2"
    [[ -n "$name" ]] || { echo "      $label: nessun container"; return 0; }
    if [[ -n "$(docker ps --format '{{.Names}}' 2>/dev/null | grep -Fx "$name" || true)" ]]; then
        docker stop "$name" >/dev/null 2>&1 \
            && echo "      $label: fermato ($name)" \
            || echo "      $label: docker stop fallito ($name)" >&2
    else
        echo "      $label: gia' fermo ($name)"
    fi
}

unload_model() {
    # keep_alive=0 sfratta i pesi dalla memoria lasciando su il server.
    # Scarichiamo cio' che e' davvero residente secondo /api/ps, non $MAIN_MODEL:
    # dopo un cambio di modello dal menu i due nomi non coincidono, e uno stop
    # deve liberare la memoria occupata, non quella che credeva di occupare.
    ollama_is_ready || { echo "      Ollama non attivo: niente da scaricare"; return 0; }

    local resident
    resident=$(curl -fsS --max-time 10 "$OLLAMA_API/api/ps" 2>/dev/null \
        | python3 -c 'import json,sys
try:
    for m in json.load(sys.stdin).get("models", []):
        if m.get("name"):
            print(m["name"])
except Exception:
    pass' 2>/dev/null)

    [[ -n "$resident" ]] || { echo "      Nessun modello residente"; return 0; }

    local name
    while IFS= read -r name; do
        [[ -n "$name" ]] || continue
        if curl -fsS --max-time 30 -H 'Content-Type: application/json' \
            -d "{\"model\":\"$name\",\"messages\":[],\"keep_alive\":0,\"stream\":false}" \
            "$OLLAMA_API/api/chat" >/dev/null 2>&1; then
            echo "      Scaricato dalla memoria: $name"
        else
            echo "      Impossibile scaricare: $name" >&2
        fi
    done <<< "$resident"
}

run_stop() {
    echo "================================================================"
    echo " Arresto stack Claude Code + Ollama"
    echo "================================================================"

    # I sottoshell contengono le die() delle funzioni di stop: un fallimento
    # su un componente non deve impedire di fermare gli altri.
    echo "[1/4] Proxy Anthropic..."
    ( stop_proxy ) || echo "      Proxy: arresto fallito (porta occupata da altri?)" >&2

    echo "[2/4] Container dashboard/telemetria..."
    # Con il service systemd un kill sul PID verrebbe annullato da
    # Restart=on-failure: va fermata l'unita', non il processo.
    if systemctl is-active --quiet gb10-gpu-agent 2>/dev/null; then
        if sudo -n systemctl stop gb10-gpu-agent 2>/dev/null; then
            echo "      Shim GB10: unita' systemd fermata"
        else
            echo "      Shim GB10: gestito da systemd, richiede:" >&2
            echo "                 sudo systemctl stop gb10-gpu-agent" >&2
        fi
    elif [[ -s "$GPU_AGENT_DIR/agent.pid" ]]; then
        local apid
        apid=$(<"$GPU_AGENT_DIR/agent.pid")
        if kill -0 "$apid" 2>/dev/null && [[ $(ps -p "$apid" -o args=) == *uvicorn* ]]; then
            kill "$apid" && echo "      Shim GB10: fermato (PID $apid)"
        else
            echo "      Shim GB10: gia' fermo"
        fi
        rm -f "$GPU_AGENT_DIR/agent.pid"
    fi
    if command -v docker >/dev/null 2>&1; then
        stop_container "$(gpu_agent_container)" "GPU Agent"
        stop_container "$(ollama_admin_container)" "ollama-admin"
    else
        echo "      docker non trovato: salto."
    fi

    if (( KEEP_OLLAMA )); then
        echo "[3/4] Ollama: lasciato attivo (--keep-ollama)"
        echo "[4/4] Scarico il modello dalla VRAM..."
        unload_model
    else
        echo "[3/4] Scarico il modello dalla VRAM..."
        unload_model
        echo "[4/4] Server Ollama..."
        ( stop_ollama ) || echo "      Ollama: arresto fallito" >&2
        ollama_is_ready \
            && echo "      ATTENZIONE: Ollama risponde ancora su $OLLAMA_API" >&2 \
            || echo "      Ollama fermo."
    fi

    echo
    echo "Fatto."
}

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if (( DO_STOP )); then
    run_stop
    exit 0
fi

if (( CLAUDE_ONLY )); then
    ollama_is_ready || die "Ollama non è attivo: esegui prima lo script senza --claude-only"
    select_model
    print_config
    ollama show "$MAIN_MODEL" >/dev/null 2>&1 || die "$MAIN_MODEL non esiste: esegui prima lo script senza --claude-only"
    proxy_is_ready || die "il proxy non è attivo su $PROXY_API: esegui prima lo script senza --claude-only"
    launch_claude
fi

stop_proxy || true
stop_ollama
start_ollama

# Il menu richiede il server attivo: `ollama list` interroga l'API.
select_model
print_config
setup_ollama_admin
setup_gpu_agent

echo "[2/6] Verifico il modello base..."
ensure_model "$MAIN_BASE_MODEL"

create_runtime_model
preload_model
start_proxy
verify_classifier_route

launch_claude
