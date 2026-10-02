# Run Locally — Step by Step

Three System 1 options: **mock** (no weights), **laya** (self-hosted
open weights), **encoder** (your fine-tuned checkpoint). System 2
(Gemma) stays mocked unless you point it at Ollama/Vertex.

## Option A — Plain Python, mocked (fastest, no downloads)

### 1. Prerequisites

- Python 3.11+

### 2. Install and run

```bash
cd intent-router-laya
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn src.app:app --port 8080
```

### 3. Try all four routing paths

```bash
# fast path (System 1 confident, risk low)
curl -s -X POST http://localhost:8080/route \
  -H 'content-type: application/json' \
  -d '{"text": "My app keeps crashing on login"}' | python3 -m json.tool

# System 2 review + confirm
curl -s -X POST http://localhost:8080/route \
  -H 'content-type: application/json' \
  -d '{"text": "invoice login problem"}' | python3 -m json.tool

# System 2 escalation
curl -s -X POST http://localhost:8080/route \
  -H 'content-type: application/json' \
  -d '{"text": "I want a refund for my invoice"}' | python3 -m json.tool

# guardrail block (System 2 not consulted)
curl -s -X POST http://localhost:8080/route \
  -H 'content-type: application/json' \
  -d '{"text": "please delete my account now"}' | python3 -m json.tool
```

### 4. Run the test suite

```bash
pytest eval/ -q        # 25 tests, no network, no weights
```

## Option B — Real Laya, self-hosted

### 1. Install Laya (once)

```bash
# CPU torch first: the default CUDA wheel is ~2.5GB and unnecessary
# for a 421M decision model. Install in this order!
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install laya
```

### 2. Run with the Laya backend

```bash
SYSTEM1_BACKEND=laya uvicorn src.app:app --port 8080
# checkpoint defaults to convaiinnovations/laya-typed-decisions
```

What to expect on first run (measured):
- **Checkpoint download**: ~1.6GB from Hugging Face, ~6 minutes on a
  decent connection. Cached in `~/.cache/huggingface` afterwards
  (override with `HF_HOME`).
- **Model load**: part of the ~6 minutes above; `LAYA_PRELOAD=true`
  (default) loads at startup so the first request isn't the slow one.
- **Per-decision latency**: ~25–45 ms on GPU, **~4–11 s on CPU**.
  For anything beyond experimentation, use a GPU or the ONNX build.

The four curl commands from Option A now run against the real model.
Watch the logs for `Laya checkpoint loaded in Xs`.

> **Calibration note (observed)**: the checkpoint logs
> "treat confidence from the affected entries as uncalibrated", and
> measured Choice confidences were 0.016–0.235 even when the choice
> was right. Tune the gate thresholds on your labeled data before
> trusting them — see `train/README.md`.

Other checkpoints:

```bash
# multilingual (100+ languages, 322M params)
LAYA_CHECKPOINT=convaiinnovations/laya-multilingual SYSTEM1_BACKEND=laya \
  uvicorn src.app:app --port 8080
```

### 3. Opt-in live test

```bash
LAYA_LIVE_TEST=1 pytest eval/ -q -k live
```

## Option C — Train and serve your own model

```bash
pip install torch transformers scikit-learn

# 1. build labeled data from the taxonomy
python train/build_dataset.py --out train/dataset.jsonl --repeat 8

# 2. fine-tune (CPU ok for this size; GPU faster)
python train/finetune.py --data train/dataset.jsonl --out models/intent-encoder

# 3. serve it as System 1
SYSTEM1_BACKEND=encoder LAYA_CHECKPOINT=models/intent-encoder \
  uvicorn src.app:app --port 8080
```

See `train/README.md` for the Laya RLCD fine-tuning track.

## Option D — Docker Compose (router + stub workers)

```bash
docker compose up --build
# router :8080, worker-a :8001, worker-b :8002
```

For a Laya-backed container:

```bash
SYSTEM1_BACKEND=laya docker compose up --build
# (checkpoint downloads on first container start unless baked in —
#  for prod, bake it: see Dockerfile notes)
```

## Configuration cheat sheet

| Env var | Default | What it does |
|---|---|---|
| `SYSTEM1_BACKEND` | `mock` | `mock` \| `laya` \| `encoder` |
| `LAYA_CHECKPOINT` | `convaiinnovations/laya-typed-decisions` | HF id or local path (also used for encoder) |
| `LAYA_DEVICE` | `auto` | torch device override |
| `LAYA_PRELOAD` | `true` | load checkpoint at startup (not first request) |
| `SYSTEM2_BACKEND` | `mock` | `mock` \| `ollama` \| `vertex` |
| `GEMMA_MODEL` | `gemma-3-4b-it` | Ollama model name |
| `CONFIDENCE_THRESHOLD` | `0.6` | Below → System 2 |
| `HUMAN_REVIEW_THRESHOLD` | `0.5` | Noul above → System 2 |
| `GUARDRAIL_REVIEW_SCORE` | `1.5` | Risk ≥ → System 2 (0=safe..4=critical) |
| `GUARDRAIL_BLOCK_SCORE` | `3.0` | Risk ≥ → hard block to fallback |
| `WORKER_AGENTS_JSON` | stub defaults | Maps intent → worker A2A URL |

## Troubleshooting

- **`ModuleNotFoundError: src`** — run from the repo root.
- **`pip install laya` fails with "No space left"** — install CPU
  torch first (see Option B); the default CUDA torch is ~2.5GB.
  Also make sure your venv isn't on a tiny tmpfs mount.
- **Hugging Face download fails with `httpx.InvalidURL: Invalid port`**
  — some sandboxes export `NO_PROXY` with bracketed IPv6 entries
  (`[::1]`) that older httpx can't parse. Workaround for the download:
  ```bash
  NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1 \
    SYSTEM1_BACKEND=laya uvicorn src.app:app --port 8080
  ```
- **First Laya decision is slow** — checkpoint download + cold model;
  `LAYA_PRELOAD=true` (default) loads at startup; `warmup()` runs once.
  On CPU expect seconds per decision; that's normal, not a bug.
- **OOM on small machines** — use `laya-multilingual` (322M) or the
  encoder track (smaller still).
- **Port 8080 in use** — change `--port`.
