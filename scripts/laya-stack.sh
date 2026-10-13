#!/usr/bin/env bash
#
# laya-stack.sh — one portable script for the Laya intent stack.
#
# Works on: macOS (Apple Silicon + Intel) and Linux (x86_64 + ARM).
# No root needed. Everything lives under $LAYA_BASE (default: ~/laya-local).
#
#   bash laya-stack.sh setup              # clone repos, venvs, deps, checkpoint
#   bash laya-stack.sh run                # start analyzer :8081, order-agent :8082, router :8080 + smoke test
#   bash laya-stack.sh test               # run sample queries against the router
#   bash laya-stack.sh train analyzer     # unified analyzer, 5 epochs (GPU: MPS/CUDA auto, else CPU)
#   bash laya-stack.sh train router       # router ModernBERT encoder head, 5 epochs
#   bash laya-stack.sh stop               # stop everything
#
# Knobs (env): LAYA_BASE, ROUTER_PORT, ANALYZER_PORT, ORDER_PORT,
#              ANALYZER_CHECKPOINT_GCS, PYTHON_BIN
#
set -euo pipefail

LAYA_BASE="${LAYA_BASE:-$HOME/laya-local}"
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

cmd_setup() {
  need git
  check_python
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

  info "Setup done. Next:  bash $0 run"
}

cmd_run() {
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

  start_svc order-agent order-agent "$ORDER_PORT"
  wait_for "http://localhost:$ORDER_PORT/health" order-agent 30

  start_svc router intent-router-laya "$ROUTER_PORT" \
    "ANALYZER_URL=$analyzer_url" \
    "ANALYZER_TIMEOUT_S=10" \
    "WORKER_AGENTS_JSON={\"order_status\":\"http://localhost:$ORDER_PORT\",\"fallback\":\"http://localhost:$ORDER_PORT\"}"

  wait_for "http://localhost:$ROUTER_PORT/health" router 60
  echo ""
  cmd_test
  echo ""
  info "All up. Logs in $LOGDIR. Stop with: bash $0 stop"
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
    *)
      die "usage: bash $0 train [analyzer|router]"
      ;;
  esac
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
  stop)  cmd_stop ;;
  *)     cmd_help ;;
esac
