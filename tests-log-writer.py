import os, sqlite3, time, tempfile
db = os.path.join(tempfile.mkdtemp(), "t.db")
c = sqlite3.connect(db)
c.executescript('''
CREATE TABLE "Server" ("id" TEXT NOT NULL PRIMARY KEY, "name" TEXT, "url" TEXT,
  "gpuAgentUrl" TEXT, "active" BOOLEAN, "createdAt" DATETIME);
CREATE TABLE "Log" ("id" TEXT NOT NULL PRIMARY KEY, "serverId" TEXT NOT NULL,
  "model" TEXT NOT NULL, "endpoint" TEXT NOT NULL, "promptTokens" INTEGER,
  "completionTokens" INTEGER, "latencyMs" INTEGER NOT NULL, "statusCode" INTEGER NOT NULL,
  "userId" TEXT, "apiKeyId" TEXT, "ip" TEXT,
  "createdAt" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  CONSTRAINT "Log_serverId_fkey" FOREIGN KEY ("serverId") REFERENCES "Server" ("id"));
INSERT INTO "Server" VALUES ('srv1','My Ollama Server','http://127.0.0.1:11434',NULL,1,1788516806812);
''')
c.commit(); c.close()

os.environ.update({"OLLAMA_UPSTREAM":"http://127.0.0.1:11434","PROXY_HOST":"127.0.0.1",
  "PROXY_PORT":"11435","MAIN_MODEL":"m","MAIN_MAX_OUTPUT_TOKENS":"40000",
  "CLASSIFIER_MAX_OUTPUT_TOKENS":"4096","OA_DB":db,"OA_SERVER_ID":"srv1"})
import proxy2 as proxy

proxy.record_log(model="claude-sonnet-5", endpoint="/v1/messages",
                 prompt_tokens=18432, completion_tokens=1207,
                 latency_ms=53150.4, status_code=200)
# il thread e' daemon: attendo che scriva
for _ in range(50):
    time.sleep(0.05)
    rows = sqlite3.connect(db).execute('SELECT * FROM "Log"').fetchall()
    if rows: break
assert rows, "nessuna riga scritta"
r = rows[0]
cols = [d[1] for d in sqlite3.connect(db).execute('PRAGMA table_info(Log)')]
row = dict(zip(cols, r))
print("riga scritta:", {k: row[k] for k in ("model","endpoint","promptTokens","completionTokens","latencyMs","statusCode")})
assert row["promptTokens"] == 18432 and row["completionTokens"] == 1207
assert row["latencyMs"] == 53150, row["latencyMs"]
assert 1_700_000_000_000 < row["createdAt"] < 2_000_000_000_000, "createdAt non e' ms epoch"

# token a zero -> NULL, non 0 (le metriche sommano con || 0)
proxy.record_log(model="m", endpoint="/v1/messages", prompt_tokens=0,
                 completion_tokens=0, latency_ms=10, status_code=200)
time.sleep(0.5)
n = sqlite3.connect(db).execute('SELECT count(*) FROM "Log" WHERE "promptTokens" IS NULL').fetchone()[0]
assert n == 1, n

# un DB inesistente non deve sollevare né bloccare
proxy.OA_DB = "/percorso/che/non/esiste.db"
proxy.record_log(model="m", endpoint="/x", prompt_tokens=1, completion_tokens=1,
                 latency_ms=1, status_code=200)
time.sleep(0.5)
print("errore DB ingoiato senza propagarsi: OK")
print("TUTTI I TEST PASSATI")
