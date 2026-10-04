# Intent Router — Laya Edition (self-hosted, trainable System 1)

Route user queries to the right worker agent with **zero commercial API
dependency**: System 1 is a self-hosted **Laya** open-weights decision
model (Apache-2.0), and you can **train your own** classifier and serve
it the same way.

- **System 1 — Laya** (self-hosted, ~25–45 ms on GPU; ~4–11 s on CPU — see SDR §10): one `predict` call
  returns the intent (Choice), human-review probability (Noul),
  utterance type (Choice), and a **Score-based guardrail risk**
  (0=safe … 4=critical). Your hardware, your data, $0 marginal cost.
- **System 1 (alt) — your fine-tuned encoder**: `train/finetune.py`
  trains a ModernBERT classifier on your intents with temperature
  scaling; serve it as `SYSTEM1_BACKEND=encoder`.
- **System 2 — Gemma** (optional, also open weights): reviews only the
  queries System 1 is uncertain about (`mock` / `ollama` / `vertex`).

## Routing policy

```
query ──▶ Laya (System 1, local)
              │
              ├─ guardrail_score >= 3.0 ──▶ BLOCK ──▶ fallback worker
              │
              ├─ conf >= 0.6, noul <= 0.5,
              │  score < 1.5 ──▶ FAST PATH ──▶ worker (no API cost at all)
              │
              └─ otherwise ──▶ Gemma (System 2) reviews Laya's read
                                  ├─ escalate ──▶ fallback worker
                                  └─ confirm/override ──▶ worker
```

## Quick start (no weights needed)

```bash
pip install -r requirements.txt
uvicorn src.app:app --port 8080   # both systems mocked
```

```bash
curl -s -X POST localhost:8080/route -H 'content-type: application/json' \
  -d '{"text": "please delete my account now"}' | python3 -m json.tool
# -> path: guardrail_block, worker: fallback
```

## Quick start (real Laya, self-hosted)

```bash
pip install laya torch   # CPU torch is fine for the 421M checkpoint
SYSTEM1_BACKEND=laya uvicorn src.app:app --port 8080
# checkpoint defaults to convaiinnovations/laya-typed-decisions
# (downloaded from Hugging Face on first run, ~1.6GB)
```

## Train your own model

```bash
pip install torch transformers scikit-learn
python train/build_dataset.py --out train/dataset.jsonl --repeat 8
python train/finetune.py --data train/dataset.jsonl --out models/intent-encoder
SYSTEM1_BACKEND=encoder LAYA_CHECKPOINT=models/intent-encoder \
  uvicorn src.app:app --port 8080
```

See [`train/README.md`](train/README.md) for the full story, including
fine-tuning Laya itself with its RLCD flow.

Full local guide: [`docs/RUN_LOCALLY.md`](docs/RUN_LOCALLY.md).
Design review: [`docs/SDR.md`](docs/SDR.md).

## Run the tests

```bash
pytest eval/ -q     # 25 tests + 1 opt-in live test, no network by default
# LAYA_LIVE_TEST=1 pytest eval/ -q   # also runs against the real checkpoint
```

## Docker Compose demo

```bash
docker compose up --build
# router :8080, worker-a :8001, worker-b :8002
# live Laya: SYSTEM1_BACKEND=laya docker compose up --build
```

## Deploy (GCP)

**Cloud Run** (2 vCPU / 4GB is enough for the 421M checkpoint on CPU):

```bash
gcloud run deploy intent-router-laya \
  --source . \
  --cpu 2 --memory 4Gi \
  --set-env-vars SYSTEM1_BACKEND=laya,SYSTEM2_BACKEND=mock
```

No secrets needed for System 1 — the model is yours. Bake the
checkpoint into the image for prod (see Dockerfile) so cold starts
don't download weights.

**GKE:** `helm/intent-router-laya/` — ConfigMap (backends, checkpoint,
guardrail thresholds), `/health` probes. No Secret required unless
System 2 uses Vertex.

### Live deployment (Innovation Lab, GCP)

The fine-tuned encoder (`train/finetune.py` — ModernBERT, val accuracy
1.000, fitted temperature 0.573) is deployed as `laya-encoder-router`
on Cloud Run — public, no API key needed:

```bash
# health: which backend + checkpoint is resident
curl -s https://laya-encoder-router-1031371624665.us-central1.run.app/health

# route a query
curl -s -X POST https://laya-encoder-router-1031371624665.us-central1.run.app/route \
  -H 'content-type: application/json' \
  -d '{"text": "Where is my order?"}' | python3 -m json.tool
# -> intent: order_status, confidence: 0.914, path: fast, worker: worker-b

# system 1 decision only (no routing)
curl -s -X POST https://laya-encoder-router-1031371624665.us-central1.run.app/classify \
  -H 'content-type: application/json' \
  -d '{"text": "My bill seems too high this month"}' | python3 -m json.tool
# -> system1_intent: billing_inquiry
```

Model checkpoint: `gs://laya-checkpoints-anuj/intent-encoder/`.

## Layout

```
├── Dockerfile  docker-compose.yml  requirements.txt
├── data/intents.yaml            # taxonomy = the Choice options
├── train/
│   ├── build_dataset.py         # labeled JSONL from the taxonomy
│   ├── finetune.py              # train your own encoder + calibration
│   └── README.md                # both training tracks
├── docs/
│   ├── SDR.md                   # design review
│   └── RUN_LOCALLY.md           # step-by-step local run guide
├── eval/test_laya_hybrid.py     # 25 unit tests + 1 opt-in live test
├── helm/intent-router-laya/
├── scripts/
└── src/
    ├── app.py  config.py  hybrid.py  router_util.py
    ├── laya_client.py     # System 1 (mock | laya | via LayaClient)
    ├── encoder_client.py  # System 1 (self-trained checkpoint)
    ├── gemma_client.py    # System 2 (mock | ollama | vertex)
    ├── intents.py  a2a_client.py
    └── workers/stub_worker.py
```

## Laya vs Jev vs own encoder

| | Laya (this repo) | Jev (commercial) | Own encoder (train/) |
|---|---|---|---|
| Weights | open, Apache-2.0 | closed, API-only | yours, trained here |
| Cost | $0 marginal | $0.042/M in | $0 marginal |
| Latency | ~25–45 ms local | 70–500 ms + network | ~10–30 ms local |
| Data | never leaves your net | leaves your net | never leaves your net |
| Guardrail | learned Score head | learned Score head | heuristic (honest limit) |
| Trainable | yes (RLCD flow) | no | yes (plain HF Trainer) |

## Notes

- A Laya engine failure degrades to **maximum guardrail score** — the
  block path — so a dead System 1 can never silently auto-route.
- Tune all four thresholds from labeled shadow traffic, not defaults.
- Laya's vendor benchmark: fine-tuned `laya-typed-decisions` 0.766 vs
  base 0.362 accuracy on their 2,000-decision set — fine-tuning is
  where the accuracy lives.
