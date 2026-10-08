#!/usr/bin/env bash
# RLCD scheduled retraining driver.
#
# Intended to run on a schedule (cron / Cloud Scheduler) against the live
# service. It:
#   1. Checks /rlcd/stats — skips training unless enough newly approved
#      records accumulated since the last run.
#   2. Exports the approved dataset via /rlcd/export.
#   3. Triggers training via /rlcd/train (mode=spot-vm returns the gcloud
#      commands; a human or a follow-up automation runs them on the GPU VM).
#
# Env:
#   ROUTER_URL   base URL of the intent-router service (default localhost:8080)
#   MIN_APPROVED minimum newly-approved records to bother training (default 50)
#   STATE_FILE   where the last-run watermark lives
set -euo pipefail

ROUTER_URL="${ROUTER_URL:-http://localhost:8080}"
MIN_APPROVED="${MIN_APPROVED:-50}"
STATE_FILE="${STATE_FILE:-$HOME/.rlcd_sched_state}"
MODE="${RLCD_MODE:-spot-vm}"

approved_now=$(curl -s "$ROUTER_URL/rlcd/stats" | python3 -c \
  "import sys,json; print(json.load(sys.stdin).get('approved_for_training',0))")
approved_last=0
[ -f "$STATE_FILE" ] && approved_last=$(cat "$STATE_FILE")

new=$((approved_now - approved_last))
echo "approved_for_training=$approved_now (new since last run: $new)"

if [ "$new" -lt "$MIN_APPROVED" ]; then
  echo "fewer than $MIN_APPROVED new approvals; skipping training."
  exit 0
fi

ts=$(date +%Y%m%d-%H%M%S)
dataset="train/rlcd_feedback_$ts.jsonl"
echo "exporting dataset -> $dataset"
curl -s -X POST "$ROUTER_URL/rlcd/export" \
  -H 'content-type: application/json' \
  -d "{\"out\": \"$dataset\"}"

echo "triggering training (mode=$MODE)"
curl -s -X POST "$ROUTER_URL/rlcd/train" \
  -H 'content-type: application/json' \
  -d "{\"dataset\": \"$dataset\", \"out_dir\": \"models/laya_rlcd_$ts\", \"mode\": \"$MODE\"}" \
  | python3 -m json.tool

echo "$approved_now" > "$STATE_FILE"
echo "done."
