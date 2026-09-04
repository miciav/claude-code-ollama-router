"""
Shim GB10 per il GPU Agent di ollama-admin.

Sul DGX Spark nvidia-smi restituisce [N/A] per memory.total/used/free: la
memoria e' unificata, non esiste VRAM dedicata da interrogare (risposta
ufficiale NVIDIA, non un bug). Il parser originale fa int(float("[N/A]")) e
solleva ValueError sull'intera riga, quindi l'agent non perde solo la memoria:
non riporta piu' nulla, watt e temperatura compresi.

Questo modulo riusa il loro main.py e sostituisce solo query_nvidia():
  - i campi leggibili (nome, temperatura, utilizzo, watt) restano da nvidia-smi
  - la memoria arriva da /proc/meminfo, che sul GB10 e' la memoria vera
Nessun fork del progetto, nessuna libreria di sistema rimpiazzata.

Uso:  uvicorn gb10_agent:app --host 0.0.0.0 --port 11436
"""

import subprocess

import main as agent  # gpu-agent/main.py upstream, accanto a questo file

FIELDS = (
    "name,memory.total,memory.used,memory.free,"
    "temperature.gpu,utilization.gpu,power.draw"
)


def _num(value, cast=float, default=None):
    """nvidia-smi usa '[N/A]' per i campi non supportati: mai sollevare."""
    try:
        return cast(float(value))
    except (TypeError, ValueError):
        return default


def _unified_memory():
    """Totale/usata/disponibile dalla memoria di sistema, in byte."""
    values = {}
    with open("/proc/meminfo", encoding="ascii") as handle:
        for line in handle:
            key, _, rest = line.partition(":")
            parts = rest.split()
            if parts:
                values[key] = int(parts[0]) * 1024  # kB -> byte

    total = values.get("MemTotal", 0)
    # MemAvailable tiene conto della cache riclamabile: e' la stima giusta di
    # quanto spazio resta per caricare un altro modello.
    available = values.get("MemAvailable", values.get("MemFree", 0))
    return total, total - available, available


def query_nvidia() -> list[dict]:
    result = subprocess.run(
        ["nvidia-smi", f"--query-gpu={FIELDS}", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed: {result.stderr.strip()}")

    mem_total, mem_used, mem_free = _unified_memory()

    gpus = []
    for line in result.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue

        total_mib = _num(parts[1], int)
        # Memoria non interrogabile = architettura unificata (GB10, Grace-*).
        unified = total_mib is None

        if unified:
            total, used, free = mem_total, mem_used, mem_free
        else:
            total = total_mib * agent.MIB_TO_BYTES
            used = (_num(parts[2], int) or 0) * agent.MIB_TO_BYTES
            free = (_num(parts[3], int) or 0) * agent.MIB_TO_BYTES

        gpus.append(
            {
                "name": parts[0] + (" (memoria unificata)" if unified else ""),
                "memoryTotal": total,
                "memoryUsed": used,
                "memoryFree": free,
                "temperature": _num(parts[4], int, 0),
                "utilization": _num(parts[5], int, 0),
                "powerDraw": round(_num(parts[6], float, 0.0), 1) if len(parts) > 6 else None,
            }
        )
    return gpus


agent.query_nvidia = query_nvidia
app = agent.app


if __name__ == "__main__":
    # Self-check: nessun campo [N/A] deve far saltare la lettura.
    import json

    print(json.dumps(query_nvidia(), indent=2, ensure_ascii=False))
