# Local Laya Stack — run the whole chain on your machine

One portable script (`scripts/laya-stack.sh`) that sets up, runs, and trains the
full intent stack **locally**: intent-router-laya (this repo) → intent-analyzer-unified
→ order-agent. Works on macOS (Apple Silicon + Intel) and Linux (x86_64 + ARM).
No root needed.

## Office-laptop quickstart

```bash
git clone https://github.com/anujgupta9860/intent-router-laya.git
cd intent-router-laya
bash scripts/laya-stack.sh setup   # clones the other two repos, builds venvs, installs deps
bash scripts/laya-stack.sh run     # starts all three services + smoke-tests the chain
```

That's it. Everything lives under `~/laya-local` (override with `LAYA_BASE`).

## Commands

| Command | What it does |
|---|---|
| `setup` | Clones/updates the 3 repos, creates a `.venv` per repo, installs requirements (+ torch), optionally downloads the pre-trained analyzer checkpoint from GCS |
| `run` | Starts analyzer `:8081`, order-agent `:8082`, router `:8080` (wired together), waits for health, then fires two real queries through the chain |
| `test` | Just the two sample queries against a running router |
| `train analyzer` | Builds the analyzer datasets if missing, downloads the ModernBERT base model if missing, then the full 5-epoch fine-tune (MPS/CUDA auto-detected) |
| `train router` | Builds the router dataset if missing, then fine-tunes the ModernBERT encoder head (5 epochs) |
| `stop` | Stops everything |

Knobs: `LAYA_BASE`, `ROUTER_PORT`, `ANALYZER_PORT`, `ORDER_PORT`,
`ANALYZER_CHECKPOINT_GCS`, `PYTHON_BIN`.

## What `run` gives you

- Router: `http://localhost:8080` — `POST /route` is the front door
- Analyzer: `http://localhost:8081` — Tier-2 typed decisions (only starts when a trained checkpoint exists at `intent-analyzer-unified/models/unified/`; otherwise the router runs without it)
- Order agent: `http://localhost:8082` — A2A worker
- Logs: `~/laya-local/logs/`

Sample queries:

```bash
curl -s -X POST localhost:8080/route -H 'content-type: application/json' \
  -d '{"text":"track my order with id 48291"}' | python3 -m json.tool
```

## Prerequisites

- **Python 3.11+** — macOS ships 3.9 (too old). `brew install python@3.12` works; the script accepts `PYTHON_BIN=/opt/homebrew/bin/python3.12`.
- `git`, `curl`.
- `uv` is auto-installed by the script (via brew on macOS, installer elsewhere) — used only to fetch a fallback Python if needed.

## Known macOS quirk (Tahoe)

On macOS 26, Python 3.12.13's `platform.mac_ver()` returns `''`, which crashes
pip's truststore and uv's interpreter probe. The script works around it with a
venv-local `sitecustomize.py` that patches `mac_ver()` at startup, reading the
real version from `/System/Library/CoreServices/SystemVersion.plist`. Contained
to the venvs it creates; documented in `make_venv`.

## Training notes

- The analyzer's training data (`train/data/*.jsonl`) and base-model dir
  (`model-hub/`) are gitignored — `train analyzer` rebuilds/downloads them automatically.
- After `train analyzer` finishes: `bash scripts/laya-stack.sh stop && bash scripts/laya-stack.sh run`
  to serve the fresh checkpoint with Tier-2 live.
- `train router` fine-tunes the router's ModernBERT encoder head (the
  `SYSTEM1_BACKEND=encoder` alternative to Laya). Laya RLCD fine-tuning is a
  separate track (`train/train_single_gpu.py`, DDP-based).
