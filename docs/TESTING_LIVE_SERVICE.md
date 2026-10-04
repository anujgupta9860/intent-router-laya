# Testing the live Laya encoder router

Full testing guide for the deployed `laya-encoder-router` Cloud Run
service (Innovation Lab GCP project). Public — no API key needed.

**Service:** `https://laya-encoder-router-1031371624665.us-central1.run.app`
**What's serving:** your fine-tuned ModernBERT intent encoder
(`train/finetune.py` — val accuracy **1.000**, fitted temperature
**0.573**), `SYSTEM1_BACKEND=encoder`, System 2 mock.
**Checkpoint:** `gs://laya-checkpoints-anuj/intent-encoder/`
**Intents:** `account_update`, `billing_inquiry`, `fallback`,
`order_status`, `sales_question`, `technical_support`

## 1. Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness + which backend/checkpoint is resident |
| POST | `/classify` | System 1 decision only (intent, confidence, guardrail) |
| POST | `/route` | Full routing decision + worker dispatch |

Request body for POSTs: `{"text": "<user query>"}`.

## 2. Smoke test

```bash
BASE=https://laya-encoder-router-1031371624665.us-central1.run.app

# should return status ok, system1 "encoder", 6 intents
curl -s $BASE/health | python3 -m json.tool
```

Expected:

```json
{
  "status": "ok",
  "system1": "encoder",
  "system1_checkpoint": "/srv/app/models/intent-encoder",
  "system2": "mock",
  "intents": ["billing_inquiry", "technical_support", "sales_question",
              "account_update", "order_status", "fallback"]
}
```

## 3. Routing tests (curl)

```bash
BASE=https://laya-encoder-router-1031371624665.us-central1.run.app

route() {
  curl -s -X POST $BASE/route -H 'content-type: application/json' \
    -d "{\"text\": \"$1\"}" | python3 -c "
import sys, json
d = json.load(sys.stdin)
print('intent:', d['intent'], '| conf:', round(d['confidence'], 3),
      '| path:', d['path'], '| worker:', d['worker_name'],
      '| guardrail:', d['guardrail_band'], '| latency_ms:', round(d['latency_ms'], 1))
"
}

route "Where is my order?"
route "My bill seems too high this month"
route "The app keeps crashing on login"
route "I want to close my account"
```

Verified results (2026-10-03, warm instance):

| Query | Intent | Confidence | Path | Worker | Notes |
|---|---|---|---|---|---|
| Where is my order? | order_status | 0.914 | fast | worker-b | System 2 skipped |
| My bill seems too high this month | billing_inquiry | 1.000 | fast | worker-a | |
| The app keeps crashing on login | technical_support | 1.000 | fast | worker-a | |
| I want to close my account | account_update | 0.845 | — | — | via `/classify`; guardrail heuristic flags "close account" as critical (3.6) — heuristic, not learned |

## 4. Python client

```python
import requests

BASE = "https://laya-encoder-router-1031371624665.us-central1.run.app"

def route(text: str) -> dict:
    r = requests.post(f"{BASE}/route", json={"text": text}, timeout=120)
    r.raise_for_status()
    return r.json()

d = route("Where is my order?")
print(d["intent"], round(d["confidence"], 3), d["worker_name"])
# order_status 0.914 worker-b
```

## 5. Reading a `/route` response

```json
{
  "path": "fast",              // fast | review | block | fallback
  "intent": "order_status",    // System 1 (or System 2 override) intent
  "confidence": 0.914,         // temperature-calibrated probability
  "probabilities": {...},      // full distribution over the 6 intents
  "needs_human": 0.236,        // 1 - confidence (heuristic)
  "utterance_type": "question",
  "guardrail_score": 0.2,      // 0-4 heuristic score
  "guardrail_band": "safe",    // safe|low|medium|high|critical
  "gates": {                   // which policy gates fired
    "low_confidence": false,
    "human_review": false,
    "guardrail_review": false,
    "guardrail_block": false
  },
  "system2_used": false,       // true when Gemma/mock review ran
  "worker": "http://worker-b:8002",
  "worker_name": "worker-b",
  "task_id": "task-...",
  "latency_ms": 196.7
}
```

`path: fast` means System 1 was confident and the guardrail was safe, so
no System 2 review ran — zero marginal cost per query.

## 6. Known limitations (observed live)

- **Template-data generalization gap.** The model was trained on
  template-generated data (232 records), so phrasings far from the
  templates can misroute *confidently*. Example (2026-10-03):
  `"I want to delete my account"` → `fallback` at confidence 1.0,
  while the template-adjacent `"I want to close my account"` →
  `account_update` at 0.845. This is a data limitation, not a pipeline
  bug — replace the templates with real labeled queries for production
  (see `train/README.md`).
- **Heuristic guardrail.** `guardrail_score` / `guardrail_band` are
  keyword heuristics in this track, not a learned head. Words like
  "close/delete account" trip the critical band. Fine-tuning Laya
  itself (RLCD track) learns the Score head instead.
- **Cold starts.** First request after idle loads the ~600MB checkpoint;
  allow up to ~60s. Warm p99 is a few hundred ms on 2 vCPU.

## 7. Troubleshooting

- **Connection timeout on first call** — cold start; retry once.
- **`{"detail": ...}` 503** — router still initializing; wait and retry.
- **Unexpected intent** — check `probabilities` for the runner-up; if the
  phrasing is far from `data/intents.yaml` examples, see §6.

## 8. Related docs

- `docs/TRAIN_ON_GCP.md` — how this model was trained and deployed
- `train/README.md` — both training tracks, honest scope notes
- `docs/RUN_LOCALLY.md` — run the same stack on your machine
- `docs/SDR.md` — system design record
