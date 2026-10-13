#!/usr/bin/env bash
#
# laya-stack.sh — one portable script for the Laya intent stack.
#
# Works on: macOS (Apple Silicon + Intel) and Linux (x86_64 + ARM).
# No root needed. Everything lives under $LAYA_BASE (default: ~/laya-local).
#
#   bash laya-stack.sh setup              # clone repos, venvs, deps, checkpoint
#   bash laya-stack.sh run                # start analyzer :8081, workers, router :8080 + smoke test
#   bash laya-stack.sh test               # run sample queries against the router
#   bash laya-stack.sh train analyzer     # unified analyzer, 5 epochs (GPU: MPS/CUDA auto, else CPU)
#   bash laya-stack.sh train router       # router ModernBERT encoder head, 5 epochs
#   bash laya-stack.sh new-agent <name> [intent] [port]
#                                         # scaffold a new A2A worker agent (auto-discovered by run)
#   bash laya-stack.sh stop               # stop everything
#
# Knobs (env): LAYA_BASE, ROUTER_PORT, ANALYZER_PORT, ORDER_PORT,
#              ANALYZER_CHECKPOINT_GCS, PYTHON_BIN, SYSTEM1_BACKEND
#              (mock| laya — laya uses the fine-tuned checkpoint if
#              downloaded, else the base HF checkpoint)
#
set -euo pipefail

# Base dir: the parent of the repo this script lives in, so the script
# operates on the checkout you ran it from (e.g. ~/IdeaProjects) instead of
# always ~/laya-local. An explicit LAYA_BASE still wins. The standalone copy
# in ~/laya-setup (not inside a repo) falls back to ~/laya-local.
_script_dir="$(cd "$(dirname "$0")" 2>/dev/null && pwd)"
_repo_dir="$(dirname "$_script_dir")"
case "$(basename "$_repo_dir")" in
  intent-router-laya|intent-analyzer-unified|order-agent)
    _default_base="$(dirname "$_repo_dir")" ;;
  *) _default_base="$HOME/laya-local" ;;
esac
LAYA_BASE="${LAYA_BASE:-$_default_base}"
ROUTER_PORT="${ROUTER_PORT:-8080}"
ANALYZER_PORT="${ANALYZER_PORT:-8081}"
ORDER_PORT="${ORDER_PORT:-8082}"
ANALYZER_CHECKPOINT_GCS="${ANALYZER_CHECKPOINT_GCS:-gs://laya-checkpoints-anuj/intent-analyzer-unified/unified/}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
LOGDIR="$LAYA_BASE/logs"

OS="$(uname -s)"
ARCH="$(uname -m)"

REPOS="intent-router-laya intent-analyzer-unified order-agent"
GH="https://github.com/anujgupta9860"

die() { echo "ERROR: $*" >&2; exit 1; }
info() { echo "==> $*"; }

need() { command -v "$1" >/dev/null 2>&1 || die "'$1' not found on PATH"; }

check_python() {
  need "$PYTHON_BIN"
  "$PYTHON_BIN" -c 'import sys; assert sys.version_info >= (3, 11), "Python >= 3.11 required"' \
    || die "Python >= 3.11 required (have: $("$PYTHON_BIN" -V 2>&1))"
}

ensure_python() {
  # The system python3 on many Macs is 3.9 — too old. Switch to a uv-managed
  # 3.12 instead of dying. Idempotent: a good interpreter is left alone.
  # Called by every command that uses PYTHON_BIN (setup/train/run/new-agent),
  # since each `bash $0 ...` invocation is a fresh process.
  if "$PYTHON_BIN" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' 2>/dev/null; then
    return 0
  fi
  command -v uv >/dev/null 2>&1 || export PATH="$HOME/.local/bin:$PATH"
  command -v uv >/dev/null 2>&1 || die "Python >= 3.11 required (have: $("$PYTHON_BIN" -V 2>&1)) and 'uv' is not installed — install it first: curl -LsSf https://astral.sh/uv/install.sh | sh"
  info "System Python too old ($("$PYTHON_BIN" -V 2>&1)) — installing uv-managed Python 3.12"
  uv python install 3.12 || die "uv python install failed"
  PYTHON_BIN="$(uv python find 3.12)" || die "uv python find failed"
  info "Using managed interpreter: $PYTHON_BIN"
}

wait_for() { # wait_for <url> <name> <timeout_s>
  local url="$1" name="$2" timeout="${3:-60}" i=0
  until curl -sf -o /dev/null "$url" 2>/dev/null; do
    i=$((i + 1))
    if [ "$i" -ge "$timeout" ]; then
      echo "TIMEOUT waiting for $name ($url)" >&2
      echo "--- tail of $LOGDIR/$name.log ---" >&2
      tail -20 "$LOGDIR/$name.log" >&2 || true
      exit 1
    fi
    sleep 1
  done
  info "$name up"
}

make_venv() { # make_venv <repo_dir> — venv + pip, with mac_ver() shim
  local dir="$1"
  # Complete = activate + python + importable pip + working mac_ver() (shim).
  # A partial venv (e.g. from a killed `uv venv`) can have the binary but no
  # pip — treat that as rubble, not as done.
  if [ -f "$dir/.venv/bin/activate" ] \
    && "$dir/.venv/bin/python" -c "import pip, platform; assert platform.mac_ver()[0]" 2>/dev/null; then
    return 0
  fi
  # Anything less than a complete venv is rubble — clear it and rebuild.
  rm -rf "$dir/.venv"
  "$PYTHON_BIN" -m venv --without-pip "$dir/.venv" \
    || die "could not create venv in $dir"
  # Python 3.12.13's platform.mac_ver() returns '' on macOS 26 (Tahoe),
  # which crashes pip's truststore and uv's interpreter probe. Shim it at
  # interpreter startup via sitecustomize (venv-local only), reading the
  # real version straight from the system plist.
  local sp
  sp="$("$dir/.venv/bin/python" -c 'import sysconfig;print(sysconfig.get_path("purelib"))')"
  cat > "$sp/sitecustomize.py" <<'PYEOF'
import os as _os
import platform as _platform

_orig_mac_ver = _platform.mac_ver

def _mac_ver(*args, **kwargs):
    try:
        v = _orig_mac_ver(*args, **kwargs)
    except Exception:
        v = ("", ("", "", ""), "")
    if not v[0]:
        try:
            import plistlib
            with open("/System/Library/CoreServices/SystemVersion.plist", "rb") as f:
                pv = plistlib.load(f)["ProductVersion"]
            v = (pv, tuple(pv.split(".")), _os.uname().machine)
        except Exception:
            v = ("26.0", ("26", "0"), "arm64")
    return v

_platform.mac_ver = _mac_ver
PYEOF
  # With the shim in place, the stock pip bootstrap works again.
  curl -sS https://bootstrap.pypa.io/get-pip.py -o "$dir/get-pip.py" \
    || die "could not download get-pip.py (network?)"
  "$dir/.venv/bin/python" "$dir/get-pip.py" -q \
    || die "pip bootstrap failed in $dir"
  rm -f "$dir/get-pip.py"
}

start_svc() { # start_svc <name> <dir> <port> [env assignments...]
  local name="$1" dir="$2" port="$3"; shift 3
  info "Starting $name (:$port)"
  (
    cd "$LAYA_BASE/$dir"
    # shellcheck disable=SC1091
    source .venv/bin/activate
    # shellcheck disable=SC2086
    env $* nohup uvicorn src.app:app --port "$port" > "$LOGDIR/$name.log" 2>&1 &
  )
}

list_workers() { # prints "<dirname> <intent> <port>" for each ready worker
  local wdir
  for wdir in "$LAYA_BASE"/*-agent; do
    [ -f "$wdir/worker.json" ] || continue
    [ -d "$wdir/.venv" ] || continue
    "$PYTHON_BIN" - "$wdir" 2>/dev/null <<'PYEOF' || continue
import json, sys
w = json.load(open(sys.argv[1] + "/worker.json"))
print(sys.argv[1].rsplit("/", 1)[-1], w["intent"], w["port"])
PYEOF
  done
}

worker_map_json() { # JSON intent->URL map for WORKER_AGENTS_JSON (no spaces: env $* splits)
  "$PYTHON_BIN" - "$LAYA_BASE" 2>/dev/null <<'PYEOF' || echo '{}'
import json, glob, sys
entries = {}
for wfile in sorted(glob.glob(sys.argv[1] + "/*-agent/worker.json")):
    try:
        w = json.load(open(wfile))
        if w.get("intent") and w.get("port"):
            entries[w["intent"]] = "http://localhost:%s" % w["port"]
    except Exception:
        pass
if entries and "fallback" not in entries:
    entries["fallback"] = next(iter(entries.values()))
print(json.dumps(entries, separators=(",", ":")))
PYEOF
}

write_worker_template() { # write_worker_template <dir> <name> <title>
  # Minimal A2A worker: /health, /message, agent card. Fill in handle().
  # Unquoted heredocs below expand $name/$title (template has no $ of its own).
  local dir="$1" name="$2" title="$3"
  cat > "$dir/requirements.txt" <<REQEOF
fastapi>=0.115
uvicorn[standard]>=0.30
pydantic>=2.0
REQEOF
  touch "$dir/src/__init__.py"
  cat > "$dir/src/app.py" <<PYEOF
"""$title A2A worker agent (scaffolded by laya-stack.sh).

A2A surface:
  GET  /.well-known/agent-card.json
  POST /message - {"text", "intent", "context": {"analyzer": {...}}}
                  -> {"task_id", "status", "action", "message", "data"}
  GET  /health

Implement your domain logic in handle(). Tier-2 analyzer decisions arrive
in context["analyzer"] (define the taxonomy in the analyzer Studio's
Training Data tab first).
"""
from __future__ import annotations

import logging
import time
import uuid

from fastapi import FastAPI
from pydantic import BaseModel, Field

AGENT_NAME = "$name"
AGENT_TITLE = "$title"

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = FastAPI(title="$title (A2A worker)", version="0.1.0")


def build_agent_card() -> dict:
    return {
        "name": "$name-agent",
        "description": "$title worker. Scaffolded by laya-stack.sh.",
        "version": "0.1.0",
        "protocol": "a2a-http",
        "endpoints": {"message": "/message"},
        "skills": [{"name": "greet", "description": "Greet the user."}],
        "input": {"text": "customer message", "intent": "router intent",
                  "context": {"analyzer": "Tier-2 analyzer decisions"}},
    }


@app.get("/.well-known/agent-card.json")
def agent_card():
    return build_agent_card()


@app.get("/health")
def health():
    return {"status": "ok", "agent": "$name-agent"}


class MessageRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    intent: str = ""
    context: dict = Field(default_factory=dict)


def handle(text: str, intent: str, analyzer: dict) -> dict:
    """TODO: implement your domain logic here.

    \`analyzer\` carries your Tier-2 decisions, e.g.
    analyzer.get("<your_choice_question>") plus your Noul flags.
    Whatever you put in \`message\` is what the user sees.
    """
    task_id = "task-" + uuid.uuid4().hex[:8]
    q = text.lower()
    if any(g in q for g in ("hi", "hello", "hey", "namaste")):
        return {"task_id": task_id, "status": "completed", "action": "greet",
                "message": "Hello! I'm the $title agent. (scaffold — implement handle() in src/app.py)",
                "data": {}}
    return {"task_id": task_id, "status": "completed", "action": "echo",
            "message": "[$name] received: " + text + " (scaffold — no domain logic yet)",
            "data": {"intent": intent, "analyzer": analyzer}}


@app.post("/message")
def message(req: MessageRequest):
    started = time.monotonic()
    analyzer = (req.context or {}).get("analyzer") or {}
    result = handle(req.text.strip(), req.intent.strip(), analyzer)
    result["latency_ms"] = round((time.monotonic() - started) * 1000, 1)
    result["ts"] = time.time()
    log.info("task=%s action=%s status=%s", result["task_id"],
             result.get("action"), result.get("status"))
    return result
PYEOF
  cat > "$dir/skills/skills.yaml" <<YAMLEOF
# Skills for the $name worker. List what your agent can do; wire them
# into handle() in src/app.py as you implement them.
version: 1
skills:
  - name: greet
    description: Greet the user.
YAMLEOF
  cat > "$dir/README.md" <<MDEOF
# $title agent (scaffolded by laya-stack.sh)

1. **Taxonomy**: analyzer Studio → Training Data tab → new agent \`$name\`
   (labels + Noul flags + seeds) → Build dataset → \`bash laya-stack.sh train analyzer\`.
2. **Logic**: implement \`handle()\` in \`src/app.py\`; read Tier-2 decisions from
   the \`analyzer\` dict.
3. **Skills**: declare them in \`skills/skills.yaml\`.
4. **Run**: \`bash laya-stack.sh run\` — this worker is auto-discovered via \`worker.json\`.

Endpoints: \`GET /health\`, \`GET /.well-known/agent-card.json\`, \`POST /message\`.
MDEOF
}

cmd_setup() {
  need git
  # uv first: it provides the Python itself when the system one is too old
  # (macOS ships python3 3.9 — check_python below would die before the
  # fallback ever ran).
  # uv: venv + package management without pip's macOS TLS quirks, much faster.
  if ! command -v uv >/dev/null 2>&1; then
    info "Installing uv"
    if [ "$OS" = "Darwin" ] && command -v brew >/dev/null 2>&1; then
      brew install -q uv || die "brew install uv failed"
    else
      curl -LsSf https://astral.sh/uv/install.sh | sh || die "uv install failed"
      export PATH="$HOME/.local/bin:$PATH"
    fi
  fi
  ensure_python
  check_python
  # Some Homebrew Pythons can't report the macOS version (platform.mac_ver()
  # returns ''), which breaks pip's truststore AND uv's interpreter probing.
  # Fall back to a uv-managed standalone build in that case (macOS only —
  # mac_ver() is always empty on Linux, which is normal there).
  if [ "$OS" = "Darwin" ] && ! "$PYTHON_BIN" -c \
      "import platform,sys;sys.exit(0 if platform.mac_ver()[0] else 1)" 2>/dev/null; then
    info "Interpreter can't report macOS version — installing uv-managed Python 3.12"
    uv python install 3.12 || die "uv python install failed"
    PYTHON_BIN="$(uv python find 3.12)" || die "uv python find failed"
    info "Using managed interpreter: $PYTHON_BIN"
  fi
  info "OS=$OS ARCH=$ARCH  base=$LAYA_BASE"
  mkdir -p "$LAYA_BASE" "$LOGDIR"
  cd "$LAYA_BASE"

  info "Cloning / updating repos"
  for repo in $REPOS; do
    if [ -d "$repo/.git" ]; then
      info "  $repo: pulling latest"
      git -C "$repo" pull --ff-only -q
    else
      info "  $repo: cloning"
      git clone -q "$GH/$repo.git"
    fi
  done

  info "Creating venvs + installing deps (takes a few minutes)"
  for repo in $REPOS; do
    info "  $repo"
    make_venv "$LAYA_BASE/$repo"
    local vpy="$LAYA_BASE/$repo/.venv/bin/python"
    # NOTE: use the venv's own pip, NOT `uv pip` — uv probes interpreters in
    # isolated mode (-I), which skips sitecustomize and trips over the broken
    # platform.mac_ver() again. The venv pip runs normally, with the shim active.
    "$vpy" -m pip install -q --upgrade pip
    "$vpy" -m pip install -q -r "$LAYA_BASE/$repo/requirements.txt"
    case "$repo" in
      intent-analyzer-unified|intent-router-laya)
        # torch: MPS on Apple Silicon, CUDA/CPU on Linux — pip picks the right wheel
        "$vpy" -m pip install -q "torch>=2.0" ;;
    esac
  done

  # Worker registry: order-agent is the built-in worker. Ports are captured
  # at setup time; re-run setup if you change *_PORT afterwards.
  printf '{"intent": "order_status", "port": %s}\n' "$ORDER_PORT" \
    > "$LAYA_BASE/order-agent/worker.json"

  # Scaffolded workers (from `new-agent`): venvs + deps.
  local wdir
  for wdir in "$LAYA_BASE"/*-agent; do
    case "$wdir" in */order-agent) continue ;; esac
    [ -f "$wdir/requirements.txt" ] || continue
    info "  $(basename "$wdir") (scaffolded worker)"
    make_venv "$wdir"
    "$wdir/.venv/bin/python" -m pip install -q --upgrade pip
    "$wdir/.venv/bin/python" -m pip install -q -r "$wdir/requirements.txt"
  done

  info "Analyzer checkpoint (~596MB, optional)"
  if [ ! -d "$LAYA_BASE/intent-analyzer-unified/models/unified" ]; then
    if command -v gcloud >/dev/null 2>&1; then
      printf "  Download pre-trained checkpoint from GCS? [y/N] "
      read -r ans
      if [[ "$ans" =~ ^[Yy]$ ]]; then
        mkdir -p "$LAYA_BASE/intent-analyzer-unified/models"
        gcloud storage cp -r "$ANALYZER_CHECKPOINT_GCS" \
          "$LAYA_BASE/intent-analyzer-unified/models/unified"
      fi
    else
      echo "  gcloud not found — skipped. Train locally instead: bash $0 train analyzer"
    fi
  else
    info "  checkpoint already present"
  fi

  # Laya fine-tuned checkpoint (for SYSTEM1_BACKEND=laya). The base HF
  # checkpoint downloads automatically on first use if this is skipped.
  info "Laya fine-tuned checkpoint (optional)"
  if [ ! -d "$LAYA_BASE/intent-router-laya/checkpoints/laya_finetuned" ]; then
    if command -v gcloud >/dev/null 2>&1; then
      printf "  Download fine-tuned Laya checkpoint from GCS? [y/N] "
      read -r ans
      if [[ "$ans" =~ ^[Yy]$ ]]; then
        mkdir -p "$LAYA_BASE/intent-router-laya/checkpoints"
        gcloud storage cp -r "gs://laya-checkpoints-anuj/laya-rlcd/laya_finetuned/" \
          "$LAYA_BASE/intent-router-laya/checkpoints/laya_finetuned"
      fi
    else
      echo "  gcloud not found — skipped. Base HF checkpoint downloads on first use."
    fi
  else
    info "  fine-tuned checkpoint already present"
  fi

  info "Setup done. Next:  bash $0 run"
}

cmd_run() {
  ensure_python
  mkdir -p "$LOGDIR"
  for repo in $REPOS; do
    [ -d "$LAYA_BASE/$repo/.venv" ] || die "not set up — run 'bash $0 setup' first"
  done
  # Clean up any orphaned services from an interrupted run.
  pkill -f "uvicorn src.app:app" 2>/dev/null || true
  sleep 1

  # The analyzer cannot start without a trained checkpoint — run Tier-2 only
  # when one exists; otherwise start the router without it (degraded mode:
  # System 1 + worker agents still work).
  local analyzer_url=""
  if [ -d "$LAYA_BASE/intent-analyzer-unified/models/unified" ]; then
    start_svc analyzer intent-analyzer-unified "$ANALYZER_PORT" \
      "ANALYZER_CHECKPOINT_DIR=models"
    wait_for "http://localhost:$ANALYZER_PORT/health" analyzer 90
    analyzer_url="http://localhost:$ANALYZER_PORT"
  else
    info "No analyzer checkpoint at models/unified — starting WITHOUT Tier-2 analyzer."
    info "Build one via 'bash $0 train analyzer' (or download it from GCS)."
  fi

  # Every *-agent/worker.json is a worker: start them all and build the
  # router's intent->URL map from the same files (single source of truth).
  local wdir
  for wdir in "$LAYA_BASE"/*-agent; do
    [ -f "$wdir/worker.json" ] || continue
    [ -d "$wdir/.venv" ] || die "$(basename "$wdir") has no venv — run 'bash $0 setup' first"
  done
  local wmap
  wmap="$(worker_map_json)"
  [ "$wmap" != "{}" ] || die "no workers registered — run 'bash $0 setup' first"
  info "Worker map: $wmap"

  local wname wintent wport
  while read -r wname wintent wport; do
    [ -n "$wname" ] || continue
    start_svc "$wname" "$wname" "$wport"
    wait_for "http://localhost:$wport/health" "$wname" 30
  done < <(list_workers)

  # System 1 backend: mock (default) or laya. For laya, prefer the local
  # fine-tuned checkpoint; fall back to the base HF checkpoint (the laya
  # package downloads it on first use, zero-shot on our questions).
  local s1_backend="${SYSTEM1_BACKEND:-mock}"
  local s1_checkpoint=""
  if [ "$s1_backend" = "laya" ]; then
    if [ -d "$LAYA_BASE/intent-router-laya/checkpoints/laya_finetuned" ]; then
      s1_checkpoint="$LAYA_BASE/intent-router-laya/checkpoints/laya_finetuned"
      info "System 1: Laya (fine-tuned checkpoint)"
    else
      s1_checkpoint="convaiinnovations/laya-typed-decisions"
      info "System 1: Laya (base HF checkpoint — no fine-tuned checkpoint found)"
    fi
  else
    info "System 1: $s1_backend (set SYSTEM1_BACKEND=laya for the real model)"
  fi

  start_svc router intent-router-laya "$ROUTER_PORT" \
    "ANALYZER_URL=$analyzer_url" \
    "ANALYZER_TIMEOUT_S=10" \
    "WORKER_AGENTS_JSON=$wmap" \
    "SYSTEM1_BACKEND=$s1_backend" \
    "LAYA_CHECKPOINT=$s1_checkpoint"

  wait_for "http://localhost:$ROUTER_PORT/health" router 60
  echo ""
  cmd_test
  echo ""
  info "All up. Logs in $LOGDIR. Stop with: bash $0 stop"
  info "Chat UI: http://localhost:$ROUTER_PORT/chat"
}

cmd_test() {
  info "Query 1: track an order (full chain: router -> analyzer -> order agent)"
  curl -s -X POST "localhost:$ROUTER_PORT/route" -H 'content-type: application/json' \
    -d '{"text":"track my order with id 48291"}' | "$PYTHON_BIN" -m json.tool
  echo ""
  info "Query 2: list orders"
  curl -s -X POST "localhost:$ROUTER_PORT/route" -H 'content-type: application/json' \
    -d '{"text":"list all my orders"}' | "$PYTHON_BIN" -m json.tool
}

cmd_train() {
  ensure_python
  local target="${1:-}"
  # MPS fallback flag only matters on Apple Silicon; harmless elsewhere.
  if [ "$OS" = "Darwin" ] && [ "$ARCH" = "arm64" ]; then
    export PYTORCH_ENABLE_MPS_FALLBACK=1
    info "Apple Silicon detected — training on MPS (Metal GPU)"
  fi
  case "$target" in
    analyzer)
      local vpy="$LAYA_BASE/intent-analyzer-unified/.venv/bin/python"
      [ -x "$vpy" ] || die "analyzer venv missing — run 'bash $0 setup' first"
      # Training data is gitignored — build it if absent.
      local missing=0 a
      for a in billing orders support account sales; do
        [ -f "$LAYA_BASE/intent-analyzer-unified/train/data/$a.jsonl" ] || missing=1
      done
      if [ "$missing" = 1 ]; then
        info "Building training datasets (synthetic, takes a minute or so)"
        ( cd "$LAYA_BASE/intent-analyzer-unified" \
          && "$vpy" train/build_dataset.py --out train/data ) \
          || die "dataset build failed"
      fi
      info "Training unified intent analyzer (5 epochs)"
      "$vpy" -c "import torch; print('mps:', torch.backends.mps.is_available(), '| cuda:', torch.cuda.is_available())"
      # Base model: prefer a local model-hub copy, else download from the Hub.
      local model_arg="model-hub/ModernBERT-base"
      if [ ! -d "$LAYA_BASE/intent-analyzer-unified/model-hub/ModernBERT-base" ]; then
        info "No local base model — downloading answerdotai/ModernBERT-base from Hugging Face"
        model_arg="answerdotai/ModernBERT-base"
      fi
      ( cd "$LAYA_BASE/intent-analyzer-unified" \
        && "$vpy" train/finetune_unified.py --data train/data --out models/unified --model "$model_arg" )
      info "Done. Restart the stack (bash $0 stop && bash $0 run) to serve the new checkpoint."
      ;;
    router)
      local vpy="$LAYA_BASE/intent-router-laya/.venv/bin/python"
      [ -x "$vpy" ] || die "router venv missing — run 'bash $0 setup' first"
      if [ ! -f "$LAYA_BASE/intent-router-laya/train/dataset.jsonl" ]; then
        info "Building router training dataset (synthetic, takes a minute or so)"
        ( cd "$LAYA_BASE/intent-router-laya" \
          && "$vpy" train/build_dataset.py --out train/dataset.jsonl ) \
          || die "dataset build failed"
      fi
      info "Training router encoder head (5 epochs)"
      "$vpy" -c "import torch; print('mps:', torch.backends.mps.is_available(), '| cuda:', torch.cuda.is_available())"
      ( cd "$LAYA_BASE/intent-router-laya" \
        && "$vpy" train/finetune.py --data train/dataset.jsonl --out models/intent-encoder )
      info "Done. Serve with: SYSTEM1_BACKEND=encoder"
      ;;
    laya)
      # Full RLCD fine-tune of Laya on Apple Silicon (MPS). Output lands at
      # checkpoints/laya_finetuned — exactly where `run` looks for it when
      # SYSTEM1_BACKEND=laya. Takes ~1-2 hours; rolling checkpoints each epoch.
      local vpy="$LAYA_BASE/intent-router-laya/.venv/bin/python"
      [ -x "$vpy" ] || die "router venv missing — run 'bash $0 setup' first"
      [ "$OS" = "Darwin" ] && [ "$ARCH" = "arm64" ] || die "train laya needs Apple Silicon (MPS)"
      ( cd "$LAYA_BASE/intent-router-laya" || die "router repo missing"
        info "Building RLCD dataset from data/intents.yaml"
        "$vpy" train/build_dataset.py --out train/dataset.jsonl --repeat 14 \
          || die "dataset build failed"
        info "Converting to RLCD format"
        "$vpy" train/convert_dataset.py --in train/dataset.jsonl --out train/laya_rlcd.jsonl \
          || die "RLCD convert failed"
        info "Preprocessing (downloads base Laya model from Hugging Face)"
        "$vpy" train/preprocess.py --data train/laya_rlcd.jsonl --out train_items.pt \
          || die "preprocess failed"
        local model_dir
        model_dir="$(cat model_dir.txt)" || die "model_dir.txt missing after preprocess"
        info "Fine-tuning Laya on MPS (4 epochs)"
        TRAIN_ITEMS=train_items.pt "$vpy" train/train_mps.py "$model_dir" checkpoints/laya_finetuned \
          || die "Laya fine-tune failed"
      ) || die "train laya failed"
      info "Done. Serve with: SYSTEM1_BACKEND=laya bash $0 run"
      ;;
    laya-feedback)
      # Retrain Laya on human-reviewed System 2 feedback (reviewed in the
      # /review UI first — pending records are skipped at export). Mixes the
      # approved feedback with the base synthetic dataset so the model
      # learns the hard cases without forgetting the base distribution.
      # Output lands at checkpoints/laya_finetuned (where `run` picks it up).
      local vpy="$LAYA_BASE/intent-router-laya/.venv/bin/python"
      [ -x "$vpy" ] || die "router venv missing — run 'bash $0 setup' first"
      [ "$OS" = "Darwin" ] && [ "$ARCH" = "arm64" ] || die "train laya-feedback needs Apple Silicon (MPS)"
      ( cd "$LAYA_BASE/intent-router-laya" || die "router repo missing"
        info "Exporting human-approved feedback"
        "$vpy" -c "from src.feedback import feedback_to_dataset; print(feedback_to_dataset('feedback/fallbacks.jsonl', 'train/rlcd_feedback.jsonl'))" \
          || die "feedback export failed"
        if [ ! -f train/dataset.jsonl ]; then
          info "Building base dataset (synthetic)"
          "$vpy" train/build_dataset.py --out train/dataset.jsonl --repeat 14 \
            || die "dataset build failed"
        fi
        info "Mixing base + approved feedback, converting to RLCD"
        cat train/dataset.jsonl train/rlcd_feedback.jsonl > train/dataset_mixed.jsonl
        "$vpy" train/convert_dataset.py --in train/dataset_mixed.jsonl --out train/laya_rlcd.jsonl \
          || die "RLCD convert failed"
        info "Preprocessing (downloads base Laya model from Hugging Face if needed)"
        "$vpy" train/preprocess.py --data train/laya_rlcd.jsonl --out train_items.pt \
          || die "preprocess failed"
        local model_dir
        model_dir="$(cat model_dir.txt)" || die "model_dir.txt missing after preprocess"
        info "Fine-tuning Laya on MPS (4 epochs)"
        TRAIN_ITEMS=train_items.pt "$vpy" train/train_mps.py "$model_dir" checkpoints/laya_finetuned \
          || die "Laya fine-tune failed"
      ) || die "train laya-feedback failed"
      info "Done. Serve with: SYSTEM1_BACKEND=laya bash $0 run"
      ;;
    *)
      die "usage: bash $0 train [analyzer|router|laya|laya-feedback]"
      ;;
  esac
}

cmd_new_agent() { # new-agent <name> [intent] [port]
  ensure_python
  local name="${1:-}"
  [ -n "$name" ] || die "usage: bash $0 new-agent <name> [intent] [port]"
  case "$name" in
    ''|*[!a-z0-9_]*|_*|[0-9]*) die "name must be lowercase_underscores, e.g. refunds" ;;
  esac
  local intent="${2:-$name}"
  case "$intent" in
    ''|*[!a-z0-9_]*|_*|[0-9]*) die "intent must be lowercase_underscores" ;;
  esac
  local dir="$LAYA_BASE/${name}-agent"
  [ -e "$dir" ] && die "$dir already exists"
  local port="${3:-}"
  if [ -z "$port" ]; then
    port=8083
    local used
    used="$(list_workers | awk '{print $3}')"
    while printf '%s\n' "$used" | grep -qx "$port"; do
      port=$((port + 1))
    done
  fi
  info "Scaffolding worker '$name' — intent '$intent' on :$port"
  mkdir -p "$dir/src" "$dir/skills"
  write_worker_template "$dir" "$name" "$(printf '%s' "$name" | tr '_' ' ')"
  printf '{"intent": "%s", "port": %d}\n' "$intent" "$port" > "$dir/worker.json"
  make_venv "$dir"
  "$dir/.venv/bin/python" -m pip install -q --upgrade pip \
    || die "pip upgrade failed in $dir"
  "$dir/.venv/bin/python" -m pip install -q -r "$dir/requirements.txt" \
    || die "pip install failed in $dir"
  echo ""
  info "Worker ready at $dir"
  info "  Taxonomy:  analyzer Studio -> Training Data tab -> new agent '$name'"
  info "             (then: bash $0 train analyzer)"
  info "  Logic:     $dir/src/app.py  (implement handle())"
  info "  Skills:    $dir/skills/skills.yaml"
  info "  Run:       bash $0 run   (this worker is auto-discovered)"
  if [ "$intent" != "order_status" ]; then
    info "  NOTE: '$intent' is a new intent — System 1 must learn it before"
    info "        the router will dispatch to this worker:"
    info "        bash $0 train router   (needs labeled examples for '$intent')"
  fi
}

cmd_stop() {
  pkill -f "uvicorn src.app:app" && info "Stopped." || info "Nothing running."
}

cmd_help() {
  sed -n '2,20p' "$0"
}

case "${1:-help}" in
  setup) cmd_setup ;;
  run)   cmd_run ;;
  test)  cmd_test ;;
  train) cmd_train "${2:-}" ;;
  new-agent) cmd_new_agent "${2:-}" "${3:-}" "${4:-}" ;;
  stop)  cmd_stop ;;
  *)     cmd_help ;;
esac
