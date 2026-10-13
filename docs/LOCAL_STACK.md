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
| `run` | Starts analyzer `:8081`, all registered workers, router `:8080` (wired together), waits for health, then fires two real queries through the chain |
| `test` | Just the two sample queries against a running router |
| `train analyzer` | Builds the analyzer datasets if missing, downloads the ModernBERT base model if missing, then the full 5-epoch fine-tune (MPS/CUDA auto-detected) |
| `train router` | Builds the router dataset if missing, then fine-tunes the ModernBERT encoder head (5 epochs) |
| `new-agent <name> [intent] [port]` | Scaffolds a new A2A worker agent (template + venv + deps), registers it in `worker.json`; `run` auto-discovers it |
| `stop` | Stops everything |

## System 1: mock vs Laya

The router's System 1 defaults to `mock` (deterministic keyword rules).
For the real model:

```bash
SYSTEM1_BACKEND=laya bash scripts/laya-stack.sh run
```

`setup` installs the `laya` package and offers the fine-tuned checkpoint
from GCS (`gs://laya-checkpoints-anuj/laya-rlcd/laya_finetuned/` — the
production canary weights, trained on our RLCD data). If skipped, `run`
falls back to the base HF checkpoint (`convaiinnovations/laya-typed-decisions`,
downloaded on first use), which answers our questions zero-shot — expect
lower accuracy than the fine-tuned weights until Laya is retrained on new
taxonomies.

## Onboarding a new agent (end to end)

1. **Taxonomy** — analyzer Studio → Training Data tab → new agent `<name>`
   (labels + Noul flags + seeds) → Save → Build dataset.
2. **Worker** — `bash scripts/laya-stack.sh new-agent <name>` — scaffolds
   `~/laya-local/<name>-agent/` (FastAPI `/health` + `/message` + agent card),
   assigns the next free port from 8083, registers `worker.json`.
   Implement your logic in `src/app.py` → `handle()`; declare skills in
   `skills/skills.yaml`.
3. **Retrain the analyzer** — `bash scripts/laya-stack.sh train analyzer`
   (picks up the new agent's heads automatically).
4. **Run** — `bash scripts/laya-stack.sh run` discovers the new worker and
   wires the router's intent→URL map from `worker.json` files. The new
   worker's `intent` (default: the agent name) is what System 1 must output —
   for a genuinely new intent, retrain the router (`train router`) with
   labeled examples so it learns to route there.

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

- **Python 3.11+** — macOS ships 3.9 (too old). The script auto-installs a
  uv-managed Python 3.12 when the system one is older — no manual upgrade
  needed. (`PYTHON_BIN=/opt/homebrew/bin/python3.12` still works to force one.)
- `git`, `curl`.
- `uv` is auto-installed by the script (via brew on macOS, installer elsewhere).

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
