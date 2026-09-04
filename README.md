# claude-qwen-spark

Claude Code su modelli locali Ollama, con dashboard e telemetria, su **NVIDIA DGX Spark (GB10)**.

Un solo modello fisico serve piu' ruoli logici. Un proxy locale compatibile con
l'API Anthropic si mette tra Claude Code e Ollama e decide, richiesta per
richiesta, se abilitare il thinking e quale tetto di output applicare.

```
Claude Code ──▶ proxy Anthropic (11435) ──▶ Ollama (11434)
                      │
                      ├── MAIN              thinking ON,  max 40K
                      ├── AUTO_CLASSIFIER   thinking OFF, max 4K, temp 0
                      ├── BACKGROUND_HAIKU  thinking OFF
                      └── AUX_CLAUDE        thinking ON
```

## Contenuto

| Percorso | Cosa fa |
|---|---|
| `claude-qwen-auto-v4.sh` | Launcher: avvia Ollama, crea il profilo runtime, avvia il proxy, lancia Claude Code |
| `claude-qwen-stop.sh` | Ferma tutto (wrapper su `--stop`) |
| `gpu-agent/` | Shim GB10 per il GPU Agent di ollama-admin, piu' installer e unit systemd |
| `dashboard/` | `docker-compose.yml` per ollama-admin |

## Uso

```bash
./claude-qwen-auto-v4.sh              # menu modelli, poi avvia tutto
./claude-qwen-auto-v4.sh --claude-only  # riusa uno stack gia' in piedi
./claude-qwen-stop.sh                 # ferma tutto
./claude-qwen-stop.sh --keep-ollama   # libera la memoria, lascia su il server
```

Variabili principali (tutte con default sensati):

```bash
MAIN_BASE_MODEL=...     # salta il menu e usa questo modello
MAIN_CONTEXT=131072     # contesto del profilo runtime
MODEL_MENU=0            # niente menu interattivo
OLLAMA_ADMIN=1          # dashboard senza chiedere conferma
GPU_AGENT=1             # telemetria senza chiedere conferma
GPU_AGENT_PORT=11436    # 11435 e' occupata dal proxy: lo script lo verifica
GPU_AGENT_BIND=0.0.0.0  # default 127.0.0.1
```

## Osservabilita'

Il proxy registra una riga per richiesta:

```
STATS route=MAIN model=qwen3.8-claude:256k in_tokens=18432 out_tokens=1207 \
      out_chars=4933 ttft_s=2.41 gen_tok_s=23.8 total_s=53.15
```

`gen_tok_s` parte dal primo token emesso, non dall'inizio della richiesta: il
costo di prefill resta isolato in `ttft_s`, cosi' un prompt lungo non fa
sembrare lento il modello. `ttft_s` e' anche l'indicatore pratico del riuso
della KV cache: prompt grande con TTFT breve significa prefisso riusato.

```bash
tail -f ~/.cache/claude-qwen/anthropic-proxy.log | grep STATS
```

### Metriche dentro la dashboard

ollama-admin non puo' contare i token: il suo proxy chiama `logAsync` subito
dopo `await fetch()`, quando sono arrivati solo gli header, e poi inoltra
`ollamaRes.body` senza leggerlo. I campi `promptTokens`/`completionTokens`
della tabella `Log` restano vuoti su quel percorso — li popola solo la loro
Chat integrata.

Il proxy di questo progetto attraversa gia' lo stream (per il watchdog) e a
fine risposta conosce i token: scrive lui la riga in `Log`. La scrittura e'
best effort, in un thread separato: un errore viene loggato come
`LOG_WRITE_FAILED` e non tocca la richiesta in corso.

Serve che il database sia scrivibile dall'utente che lancia lo script. Con
l'installazione standard vive in un volume Docker di proprieta' di root: il
`docker-compose.yml` qui usa un bind mount e `user: "1000:1000"` per evitarlo.

```bash
OA_DB=~/ollama-admin-data/ollama-admin.db   # default
OA_SERVER_ID=                               # rilevato dal DB se vuoto
OA_DB= ./claude-qwen-auto-v4.sh             # disattiva la scrittura
```

Il test del writer gira contro uno schema replica, estraendo il proxy dallo
script cosi' da verificare il codice che gira davvero:

```bash
python3 tests-log-writer.py
```

**Accoppiamento:** dipende dallo schema Prisma di un progetto di terze parti.
Se rinominano una colonna la scrittura smette, in modo visibile nel log e
senza conseguenze sulle richieste.

## Note specifiche per DGX Spark / GB10

Tre cose che su questa piattaforma non funzionano come altrove.

**La memoria GPU non e' interrogabile.** `nvidia-smi --query-gpu=memory.used`
restituisce `[N/A]`: la memoria e' unificata, non esiste VRAM dedicata. E'
il comportamento previsto, non un guasto.

**Il GPU Agent upstream si rompe, non degrada.** Il suo parser fa
`int(float(...))` su ogni campo di `nvidia-smi`; su `[N/A]` solleva
`ValueError` e perde l'intera riga — quindi niente memoria *e* niente watt.
`gpu-agent/gb10_agent.py` importa il loro `main.py` e sostituisce la sola
`query_nvidia()`: i campi leggibili restano da `nvidia-smi`, la memoria arriva
da `/proc/meminfo`. Nessun fork, nessuna libreria di sistema rimpiazzata.

**I watt sono parziali.** `power.draw` copre il SoC GB10 (TDP 140 W), non i
240 W del sistema completo. La telemetria della CPU Grace non e' esposta. Per
il consumo reale serve un misuratore a monte della presa.

Il dato piu' utile sulla memoria arriva da Ollama stesso:

```bash
curl -s localhost:11434/api/ps | python3 -m json.tool
```

`size_vram` include pesi **piu'** KV cache preallocata per l'intero `num_ctx`.
Ollama la prealloca al caricamento: e' una riserva fissa, non cresce durante
la sessione.

## Installazione telemetria + dashboard

```bash
# shim GPU + service systemd (parte al boot, riparte se crolla)
sudo ./gpu-agent/install.sh --service

# dashboard
cd dashboard
export NEXTAUTH_SECRET=$(openssl rand -hex 32)
docker compose up -d
```

La dashboard gira in **rete host** con `HOSTNAME` esplicito: cosi' raggiunge
Ollama su `127.0.0.1` senza obbligare a spostarlo su `0.0.0.0`, che
esporrebbe un'API senza autenticazione. `HOSTNAME=127.0.0.1` la vincola al
loopback (accesso via tunnel SSH), `0.0.0.0` la apre all'interfaccia.

Poi, nell'interfaccia: crea l'account amministratore, quindi
*Admin → Servers → GPU Agent URL* = `http://127.0.0.1:11436`.

## Porte

| Porta | Servizio |
|---|---|
| 11434 | Ollama |
| 11435 | Proxy Anthropic |
| 11436 | GPU Agent (upstream usa 11435: collide, spostata) |
| 3000 | Dashboard ollama-admin |
