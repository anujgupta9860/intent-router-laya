# Software Design Review (SDR): Intent Router — Laya Edition (self-hosted, trainable System 1)

| | |
|---|---|
| **Author** | GoGo (for Anuj Gupta) |
| **Date** | 2026-10-02 |
| **Status** | Draft — POC built, mock/mock verified; live Laya + training tracks documented |
| **Repo** | `intent-router-laya` (local; GitHub push pending) |
| **Predecessors** | `intent-router-poc` (Gemma-only), `intent-router-jev` (Jev-only), `intent-router-hybrid` (Jev S1 + Gemma S2) |
| **Driver** | Manager review of the hybrid: build/train our own model, deploy it as a service, stop depending on commercial models |

## 1. Problem statement

The hybrid (Jev System 1 + Gemma System 2) is architecturally right but
operationally wrong for us: System 1 — the component that touches every
query — is a commercial API. That means per-call billing, data leaving
our network on every request, no ability to train on our own decisions,
and a vendor outage degrading the whole fast path.

The manager's directive: **own System 1**. Use Laya (Convai Innovations'
open-weights System One decision model, Apache-2.0) or another
classification model we train ourselves, deployed as our own service.

## 2. Goals

- G1. **Zero commercial dependency on the hot path**: System 1 is a
  self-hosted Laya checkpoint (or our own fine-tuned encoder). No API
  keys, no per-decision billing, no query text leaving our network.
- G2. Eleven typed decisions per query (expanded 2026-10-08 from the
  original four): intent (Choice), worker agent (Choice), skill required
  (Choice), human-review probability (Noul), utterance type (Choice),
  guardrail risk (Score over `[safe, low, medium, high, critical]`),
  plus five routing flags (Noul): needs RAG, needs more input, needs
  user details, multi-turn, needs async.
- G3. Same routing policy: guardrail ≥ 3.0 → hard block; confident +
  safe → fast path; else Gemma System 2 reviews (also open weights).
- G4. **Trainable**: ship a worked fine-tuning track (encoder +
  temperature scaling, in-repo) and a documented Laya RLCD fine-tuning
  track, both served behind the identical SystemOneDecision contract.
- G5. Deploy as a service: FastAPI + Docker + Helm, checkpoint baked
  into the image or mounted from a volume.
- G6. Fail safe: engine failure → max guardrail (block path).

## 3. Non-goals

- NG1. We do not train a foundation model — we fine-tune/adapt existing
  open checkpoints (Laya's RLCD flow, or a ModernBERT classifier head).
- NG2. The router still doesn't replace worker agents or answer
  queries.
- NG3. Beating Jev's zero-shot quality out of the box is not the goal;
  matching it *after fine-tuning on our data* is.

## 4. Design

### 4.1 System 1 options (one interface, three engines)

`src/hybrid.py::SystemOne` fronts three backends behind one contract:

| Backend | Engine | Latency | Guardrail | Trainable |
|---|---|---|---|---|
| `mock` | deterministic keywords | ~1 ms | heuristic | n/a (dev/CI) |
| `laya` | `convaiinnovations/laya-typed-decisions` (default), `laya`, or `laya-multilingual` via `pip install laya` | ~25–45 ms local | **learned** Score head | yes — vendor RLCD flow |
| `encoder` | our checkpoint from `train/finetune.py` | ~10–30 ms local | heuristic (documented limit) | yes — plain HF Trainer in-repo |

The Laya client (`src/laya_client.py`) uses single-model mode
(`laya.load(checkpoint)` → `agent.predict(text, questions)`), preloads
at startup (`LAYA_PRELOAD`), and warms up so the first real decision
isn't the slow one.

### 4.2 The eleven questions (expanded 2026-10-08)

The original four (intent, human-review, utterance-type, guardrail-risk)
are unchanged, plus seven routing decisions added 2026-10-08: worker agent
(Choice), skill required (Choice), and five Noul flags — needs RAG, needs
more input, needs user details, multi-turn, needs async. All eleven are
answered in one Laya `predict` call. The routing policy, gates, thresholds
(`CONFIDENCE_THRESHOLD` 0.6, `HUMAN_REVIEW_THRESHOLD` 0.5,
`GUARDRAIL_REVIEW_SCORE` 1.5, `GUARDRAIL_BLOCK_SCORE` 3.0), the Gemma
System 2 reviewer, and the A2A dispatch are unchanged — only the System
1 engine is swapped. This is deliberate: the bake-off between Jev,
Laya, and our encoder is a config change, not a rewrite.

### 4.3 Training tracks

**Track A — encoder fine-tune (worked, in-repo).**
`train/build_dataset.py` generates labeled JSONL from
`data/intents.yaml` (template paraphrase augmentation);
`train/finetune.py` trains ModernBERT + classification head with HF
Trainer and fits a **temperature** on a held-out split so confidence
numbers are calibrated — the property the gates depend on. Exports
`models/intent-encoder/` (weights + tokenizer + `calibration.json`).
Served as `SYSTEM1_BACKEND=encoder`.

**Track B — Laya RLCD fine-tune (documented).**
Laya's own flow (notebook for Kaggle 2xT4 / Apple Silicon script in
their repo) trains the actual decision heads with reinforcement learning
from calibrated decisions. Our `build_dataset.py` output feeds it.
Serve the result as `SYSTEM1_BACKEND=laya
LAYA_CHECKPOINT=/path/to/finetune`.

Vendor evidence for why B matters: fine-tuned `laya-typed-decisions`
0.766 vs base 0.362 accuracy on their 2,000-decision benchmark.
Independent JevBench: Laya composite 54.4 vs Jev 74.4 — the gap is
zero-shot quality and calibration, which fine-tuning + temperature
scaling directly address.

### 4.4 Deployment as a service

- **Local / dev**: `SYSTEM1_BACKEND=laya uvicorn src.app:app` (checkpoint
  downloads from HF on first run, ~1.6GB) or mock.
- **Docker**: image installs `laya` + CPU torch; checkpoint baked at
  build time via `LAYA_PRELOAD=true` (recommended for prod so cold
  starts don't download weights) or mounted from a volume.
- **Cloud Run**: 2 vCPU / 4GB suffices for the 421M checkpoint on CPU.
  No secrets needed for System 1.
- **GKE**: `helm/intent-router-laya/` — ConfigMap (backend, checkpoint,
  guardrail thresholds), `/health` probes. No Secret unless System 2
  uses Vertex.

### 4.5 Failure handling

- Laya engine failure → synthetic max-guardrail decision (block path),
  never silent auto-routing.
- Gemma System 2 failure → escalate to human.
- Same A2A dispatch as predecessors.

## 5. Alternatives considered

| Option | Verdict |
|---|---|
| Jev System 1 (previous hybrid) | Architecturally identical, operationally rejected per manager: commercial API on the hot path. Kept as the bake-off baseline. |
| Laya zero-shot (this design, default) | **Chosen for System 1.** Open weights, Apache-2.0, $0 marginal, ~25–45 ms, learned guardrail head, fine-tunable. |
| Own encoder fine-tune (Track A) | **Chosen as the train-in-repo path.** Fully reproducible; honest limit: guardrail stays heuristic. |
| Laya RLCD fine-tune (Track B) | Recommended when the guardrail must be learned. More effort (GPU + vendor flow). |
| djev / OpenJev / SemIf | Noted in the ecosystem; Laya chosen for maturity (30k+ stars, SDK, Docker, MCP) and Apache-2.0. Revisit if multilingual or image input becomes a requirement. |
| Gemma classifier as System 1 | Open weights but generative — pays the parse tax Jev/Laya avoid. Gemma stays as System 2 reviewer only. |

## 6. Security & data

- Query text never leaves our network on the fast path — the core
  privacy win over Jev/OpenAI.
- No API keys for System 1: nothing to leak, rotate, or bill.
- Guardrail rubric/thresholds are versioned code/config, auditable.

## 7. Testing & evaluation

- **Unit** (`eval/test_laya_hybrid.py`, 25 tests + 1 opt-in live):
  config, four-question construction, Laya result parsing (all four
  answers, clamping, band mapping, unknown-intent fallback),
  fail-safe degradation, mock System 1/2, all four routing paths.
  No network, no weights by default.
- **Live** (`LAYA_LIVE_TEST=1`): runs one real decision against the
  downloaded checkpoint; asserts contract shape, not quality.
- **Training eval**: `finetune.py` reports val accuracy + mean
  confidence; add a reliability diagram before trusting gates in prod.
- **Bake-off (recommended before the manager commits)**: same labeled
  set through `mock` / `laya` / `encoder` / Jev-hybrid; compare
  accuracy, calibration, p99 latency, and cost. The router makes this a
  config change.

## 8. Rollout plan

1. mock/mock: validate policy and wiring (done in POC).
2. Zero-shot Laya in shadow vs current classifier; measure the
   accuracy/calibration gap on *our* queries.
3. Track A encoder on real labeled traffic; compare vs Laya zero-shot.
4. Track B (Laya RLCD) if the guardrail needs to be learned.
5. Cut over per-intent; fallback worker stays the safety net.
6. Monitor: path distribution, guardrail band histogram, System 2 rate
   (= the only remaining variable cost), p99 per path.

## 10. Live verification (2026-10-02, this POC)

Ran the real `convaiinnovations/laya-typed-decisions` checkpoint via
`pip install laya` + `laya.load(...)` on CPU. Findings:

- **Wire shape confirmed**: `agent.predict(text, questions)` returns
  `result["answers"][<id>]` with `choice`/`confidence` (Choice),
  `noul` (Noul), and `score` + per-level `probabilities` (Score) —
  exactly what `src/laya_client.py::parse_decision` expects.
- **Latency on CPU**: 3.8–11 s per 4-question decision (cold/warm),
  not the vendor's 25–45 ms (GPU figure). Plan serving hardware
  accordingly; GPU or ONNX is recommended for production.
- **Calibration warning is real**: the checkpoint logs
  "ships invalid temperatures... Treat confidence from the affected
  entries as uncalibrated", and observed Choice confidences were low
  (0.016–0.235) even when the choice was right. **Do not trust the
  default thresholds** — fit temperatures / tune gates on your labeled
  data before production (Track A does this explicitly).
- **Zero-shot quality gap is real**: "please delete my account now"
  → `technical_support` (wrong), guardrail 1.83 (medium, not
  critical). Fine-tuning on your decisions is not optional for
  production quality — see train/README.md.

## 11. Open questions

- OQ1. Laya Score-head calibration on our adversarial set — measure
  before trusting the 3.0 block threshold.
- OQ2. Checkpoint size vs latency on our serving hardware (421M on
  CPU is fine; GPU changes the math).
- OQ3. Multilingual needs → `laya-multilingual` (322M) bake-off.
- OQ4. Who owns the labeling pipeline for Track A/B training data?
- OQ5. GitHub PAT for the repo push is pending — repo currently local only.

## 10. Risks

- R1. **Zero-shot quality gap** (JevBench 54.4 vs 74.4). Mitigation:
  fine-tuning tracks A/B; the hybrid's System 2 catches what System 1
  misses in the meantime.
- R2. **Calibration**: open decision models need temperature scaling /
  vendor calibration before gates are trustworthy. Mitigation: Track A
  does it explicitly; Laya has built-in calibration tooling —
  measure, don't assume.
- R3. **We operate the model now**: versioning, monitoring, rollback
  are ours. Mitigation: checkpoint pinned by HF revision in config;
  Helm-probed `/health`; mock backend always available as instant
  fallback.
- R4. Disk/memory: 421M checkpoint ≈ 1.6GB — bake into the image, don't
  download on cold start.

## 12. System 1 expansion: 11 typed decisions (2026-10-08)

**Driver:** Anuj's System 1 notes — for a given query, System 1 should
decide not just the intent but the full routing picture: which worker,
what skill, and whether the query needs RAG, clarification, user
details, conversation context, or async execution.

**New decisions** (4 original unchanged, same phrasing):

| # | Question | Type | Purpose |
|---|----------|------|---------|
| 5 | `worker_agent` | Choice (6) | billing / support / sales / account / orders / fallback |
| 6 | `skill_required` | Choice (7) | none / billing_lookup / refund_process / order_tracking / account_modify / knowledge_search / escalation |
| 7 | `needs_rag` | Noul | answer needs knowledge base / help docs |
| 8 | `needs_more_input` | Noul | query missing details needed to act |
| 9 | `needs_user_details` | Noul | handling needs the user's account data |
| 10 | `is_multi_turn` | Noul | follow-up in an ongoing conversation |
| 11 | `needs_async` | Noul | needs long-running background work |

**Single source of truth:** `src/system1_questions.py` holds the exact
instructions + criteria strings. Training (`train/convert_dataset.py`)
and serving (`src/laya_client.build_questions`) both import it verbatim
— verified byte-identical. This is the direct lesson from the 2026-10-07
confidence-gap incident: prompt mismatch between training and serving
silently degrades confidence.

**Dataset:** `train/build_dataset.py` emits all 11 labels via
deterministic heuristics over the 29 taxonomy examples + 14 synthetic
follow-up/how-to examples (kept in-code, not in `data/intents.yaml`):
602 records (`--repeat 14`). Label balance: needs_rag 98, needs_more_input
210, needs_user_details 406, is_multi_turn 196, needs_async 70 (of 602).
→ 6,622 RLCD sequences (602 × 11).

**Routing policy changes** (`src/hybrid.py`):
- Worker resolution prefers System 1's `worker_agent` (falls back to
  intent→worker mapping); routed-to-fallback still uses the fallback worker.
- `needs_more_input >= 0.8` forces the slow path (System 2 decides whether
  to ask a clarifying question).
- `skill_required` + Noul flags ride along in `/route` and `/classify`
  responses for worker adaptation.

**Training:** RLCD fine-tune on a GCP spot T4 (same flow as the 2026-10-06
run), 4 epochs, temperature calibration on holdout. Checkpoint target:
`gs://laya-checkpoints-anuj/laya-rlcd-v2/laya_finetuned_v2/`.
Status: _blocked 2026-10-08 — T4/L4 capacity exhausted across all tried
zones (us-central1 a/b/c/f, us-east1 b/d, us-west1 a, europe-west4 a);
two spot VMs preempted within minutes. Fallback: Kaggle 2×T4 notebook
(docs/TRAIN_ON_KAGGLE.md, needs interactive session) or retry GCP later.

**Deployment plan (once trained):** new image tag, `CHECKPOINT_GCS`
pointed at the v2 checkpoint, canary 10% → verify (`delete my account`,
`Where is my order?`, `How do I reset my password?`) → 100%. Revision
`00002-rwb` stays as instant rollback. The v2 serving code sends 11
questions — it must NOT ship against the v1 4-question checkpoint.

## 13. RLCD improvement loop: turning fallbacks into training signal (2026-10-08)

**Problem:** System 1 is confident on easy queries but falls back to
System 2 (Gemma) when confidence is low. Retraining on random data wastes
capacity on cases it already handles. RLCD (Reinforcement Learning from
Calibrated Decisions) fixes the *training distribution*.

**The loop:**

1. **Log the fallbacks.** Every query where System 1 confidence drops
   below threshold routes to Gemma. Log query + Gemma's decision. These
   are, by definition, the hardest cases — the exact frontier of
   System 1's competence.
2. **System 2 as labeler.** Gemma's decisions on fallback cases become
   training labels. Targeted distillation: only the cases System 1
   couldn't handle, not everything.
3. **Retrain on the hard distribution.** Fine-tune the encoder on the
   fallback set mixed with a sample of easy cases (anti-forgetting).
   The decision boundary moves — queries that scored 0.55 now score 0.75.
4. **Recalibrate — non-negotiable.** Re-fit temperature scaling on
   held-out data after every retrain. Without this, confidence numbers
   lie and the fallback threshold becomes meaningless: either too much
   routes to the expensive LLM, or bad fast-path decisions get trusted.
5. **Repeat.** Each cycle shrinks the fallback rate. Converges toward
   System 1 handling everything except genuinely novel cases.

**Why "reinforcement":** the reward isn't just label accuracy — it's
*accuracy + honest confidence*. A model that's 90% accurate but claims
99% confidence is worse for a router than 85% accurate with honest 85%,
because the gates depend on the numbers meaning what they say.

**Note for this POC:** the 2026-10-07 confidence gap (0.93 train vs 0.59
live) was a prompt-mismatch bug, not a data problem. But once the
11-decision model ships, this loop is the production improvement path:
log Gemma fallback decisions, retrain on a cadence, watch fallback rate
drop. Track it as a KPI: `% queries resolved on System 1 fast path`.

## 14. RLCD loop API (2026-10-08)

Implements §13 as a live feedback pipeline: every System 2 review is
logged, a human reviews it, approved records train, training runs on
demand or on schedule.

**Logging** (`src/feedback.py` — `FeedbackLogger`): every System 2
review appends one JSONL record to `feedback/fallbacks.jsonl` with the
query, all 11 System 1 decisions, all 11 System 2 judgments, and
`usable_label` (false when Gemma escalated). Review status starts at
`pending`. Gemma's review prompt now asks for all 11 decisions, not just
intent (`src/gemma_client.py`). Disable with `FEEDBACK_ENABLED=false`.

**Human review gate:** nothing trains without approval. `pending` →
`approved` (Gemma's labels stand) | `corrected` (human `corrections`
override Gemma at export) | `rejected` (excluded).

**Endpoints** (`src/app.py`):

| Method & path | Purpose |
|---|---|
| `GET /rlcd/stats` | fallback counts, pending_review, approved_for_training |
| `GET /rlcd/review?limit=N` | pending Gemma decisions, newest first |
| `POST /rlcd/review/{id}` | `{"decision": "approved"\|"corrected"\|"rejected", "corrections": {...}}` |
| `POST /rlcd/export` | `{"out": "train/rlcd_feedback.jsonl"}` — approved records → training JSONL (schema matches `train/build_dataset.py`) |
| `POST /rlcd/train` | `{"dataset": ..., "mode": "local"\|"spot-vm", "epochs": 4}` — launches training |
| `GET /rlcd/train/{job_id}` | job status |
| `GET /rlcd/train` | list jobs |

**Training modes** (`src/rlcd.py`): `local` runs `train/finetune.py` in
a background thread (CPU demo only — the serving container has no GPU).
`spot-vm` exports the dataset and returns the exact `gcloud` commands for
the GPU flow (docs/TRAIN_ON_GCP.md). Production should back this with a
real queue (Cloud Tasks / Pub/Sub) and Vertex AI Custom Jobs.

**Scheduling:** `scripts/rlcd_scheduled.sh` — cron/Cloud Scheduler
driver: skips unless ≥50 new approvals since last run, then exports and
triggers. KPI: fast-path rate from `/rlcd/stats` should climb each cycle.

### §14.1 Curl reference (expected outputs; endpoints land with the v2 deploy)

```bash
BASE=https://laya-encoder-router-1031371624665.us-central1.run.app
```

**Stats — feedback counts + the fast-path KPI:**
```bash
curl -s $BASE/rlcd/stats | python3 -m json.tool
```
```json
{
  "feedback_log": "feedback/fallbacks.jsonl",
  "total_fallbacks": 37,
  "usable_labels": 29,
  "pending_review": 12,
  "approved_for_training": 17
}
```

**Review queue — pending Gemma decisions, newest first:**
```bash
curl -s "$BASE/rlcd/review?limit=1" | python3 -m json.tool
```
```json
{
  "pending": [
    {
      "id": "a3f9c1d2e4b5",
      "ts": 1791489000.0,
      "query": "refund my last charge",
      "s1": {"intent": "order_status", "confidence": 0.593, "worker_agent": "orders",
             "skill_required": "none", "needs_human": 0.1, "guardrail_score": 1.0, ...},
      "s2": {"intent": "billing_inquiry", "confidence": 0.92, "escalate_to_human": false,
             "worker_agent": "billing", "skill_required": "refund_process",
             "needs_rag": 0.0, "needs_more_input": 0.0, "needs_user_details": 1.0,
             "is_multi_turn": 0.0, "needs_async": 0.0,
             "rationale": "Refund request maps to billing with refund_process skill."},
      "usable_label": true,
      "review_status": "pending"
    }
  ]
}
```

**Approve / correct / reject:**
```bash
# approve — Gemma's labels stand
curl -s -X POST $BASE/rlcd/review/a3f9c1d2e4b5 \
  -H 'content-type: application/json' -d '{"decision": "approved"}'
# {"record_id": "a3f9c1d2e4b5", "decision": "approved"}

# correct — human labels override Gemma's at export time
curl -s -X POST $BASE/rlcd/review/a3f9c1d2e4b5 \
  -H 'content-type: application/json' \
  -d '{"decision": "corrected", "corrections": {"intent": "billing_inquiry", "skill_required": "refund_process"}}'
# {"record_id": "a3f9c1d2e4b5", "decision": "corrected"}

# reject — excluded from training
curl -s -X POST $BASE/rlcd/review/a3f9c1d2e4b5 \
  -H 'content-type: application/json' -d '{"decision": "rejected"}'
# {"record_id": "a3f9c1d2e4b5", "decision": "rejected"}
```

**Export approved records → training JSONL:**
```bash
curl -s -X POST $BASE/rlcd/export \
  -H 'content-type: application/json' \
  -d '{"out": "train/rlcd_feedback.jsonl"}' | python3 -m json.tool
```
```json
{"written": 17, "skipped": 20, "dataset": "train/rlcd_feedback.jsonl"}
```
(skipped = pending/rejected/unusable — only approved/corrected train.)

**Trigger training:**
```bash
# local mode: background thread on the service host (CPU demo only)
curl -s -X POST $BASE/rlcd/train \
  -H 'content-type: application/json' \
  -d '{"dataset": "train/rlcd_feedback.jsonl", "mode": "local"}' | python3 -m json.tool
```
```json
{"job_id": "7b2e9a1c4f03", "mode": "local", "status": "started",
 "detail": "training in background thread; poll GET /rlcd/train/{job_id}"}
```
```bash
# spot-vm mode: returns the gcloud commands for the GPU flow
curl -s -X POST $BASE/rlcd/train \
  -H 'content-type: application/json' \
  -d '{"dataset": "train/rlcd_feedback.jsonl", "mode": "spot-vm", "epochs": 4}' \
  | python3 -m json.tool
```
```json
{"job_id": "c41d88f2a6e0", "mode": "spot-vm", "status": "awaiting_vm",
 "detail": "Dataset exported. Run these on a spot T4 VM (see docs/TRAIN_ON_GCP.md):",
 "commands": ["gcloud compute instances create laya-rlcd-train ...", "..."]}
```

**Job status:**
```bash
curl -s $BASE/rlcd/train/7b2e9a1c4f03 | python3 -m json.tool
```
```json
{"job_id": "7b2e9a1c4f03", "mode": "local", "dataset": "train/rlcd_feedback.jsonl",
 "out_dir": "models/laya_rlcd_v2", "epochs": 4.0, "status": "running",
 "created_ts": 1791489000.0, "started_ts": 1791489001.5}
```

## 15. End-to-end live system: router + analyzer + order agent on Cloud Run (2026-10-10)

Everything below is deployed and live-verified in GCP project
`innovation-lab-2026` (region `us-central1`).

### 15.1 Architecture: who does what

```
                        ┌─────────────────────────────────┐
 end user ──POST /route─▶│  INTENT ROUTER (Cloud Run)       │
   {"text": "..."}       │                                 │
                         │  System 1 (Laya/mock): EVERY    │
                         │  decision — intent, worker_     │
                         │  agent, skill, needs_human,     │
                         │  guardrail (typed outputs only) │
                         │         │                       │
                         │         ▼                       │
                         │  System 2 (Gemma/mock): TOKENS  │
                         │  ONLY — rationale text. Never   │
                         │  overrides a decision.          │
                         │         │                       │
                         │         ▼                       │
                         │  ADK execution plane:           │
                         │  AdkSkillExecutor runs every    │
                         │  skill as an ADK FunctionTool   │
                         │  (run_async + ToolContext).     │
                         │  MCP preferred when offered.    │
                         └────────┬────────────────────────┘
                                  │ Tier-2: POST /analyze
                                  ▼
                         ┌─────────────────────────────────┐
                         │  UNIFIED INTENT ANALYZER        │
                         │  (Cloud Run, 1 checkpoint,      │
                         │   5 agents) → per-agent Choice  │
                         │  + Noul decisions               │
                         └────────┬────────────────────────┘
                                  │ Tier-2.5: POST /decide
                                  ▼ (policy tree → action)
                         ┌─────────────────────────────────┐
                         │  ORDER AGENT (Cloud Run)        │
                         │  A2A: POST /message             │
                         │  MCP: POST /mcp (7 tools)       │
                         │  skills: track/cancel/modify/   │
                         │  return/reorder/place/list/     │
                         │  estimate                       │
                         └─────────────────────────────────┘
```

The rule, enforced in code (`hybrid.py`, `gemma_client.py`,
`adk_execution.py`):

- **System 1 decides.** Laya typed decisions (Choice/Noul/Score) are the
  only source of routing truth. System 2 cannot override intent, worker,
  skill, or escalation — verified by unit test.
- **System 2 generates.** Gemma produces rationale text only. No
  decision fields are consulted for routing.
- **ADK executes.** `AdkSkillExecutor` wraps every skill call as an ADK
  `FunctionTool`; dispatch modes are `adk-a2a` and `adk-mcp`.

### 15.2 Live services

| Service | URL | Notes |
|---|---|---|
| intent-router | https://intent-router-1031371624665.us-central1.run.app | `ANALYZER_URL` → analyzer; `WORKER_AGENTS_JSON` → order agent |
| intent-analyzer | https://intent-analyzer-7yydsv7ybq-uc.a.run.app | 596 MB unified checkpoint from `gs://laya-checkpoints-anuj/intent-analyzer-unified/unified/`; 4 GiB RAM |
| order-agent | https://order-agent-1031371624665.us-central1.run.app | A2A + MCP; in-memory order store |

Images: `us-central1-docker.pkg.dev/innovation-lab-2026/{laya-router,intent-analyzer,order-agent}/*:latest`
(Artifact Registry repos created 2026-10-10).

### 15.3 Unified intent analyzer (Tier 2)

Repo: `intent-analyzer-unified` (public GitHub). One shared-encoder
checkpoint, per-agent Choice+Noul heads; `POST /analyze` takes
`worker_agent` + `query`. 5-epoch full training 2026-10-09: 1.000/1.000
Choice/Noul on all five agents.

**Decision framework** (layman-configurable, same repo):
`policy/decision-tree.yaml` — human-readable first-match-wins rules;
`POST /decide` = analyzer + policy → `{action, worker, risk}` for A2A
dispatch. Decision Tree Studio at `/studio` (rule editor with
valid-value dropdowns + live test panel). Hot-reload: the file is
re-checked on every decision; broken edits keep the last good tree
(error in `GET /policy/status`); `ANALYZER_POLICY_PATH` env override
for mounted volumes.

### 15.4 Order agent (A2A worker + mock MCP server)

Repo: `~/workspace/order-agent` (local only, not yet on GitHub).
`GET /.well-known/agent-card.json`, `POST /message`
(`{text, intent, context:{analyzer}}`), `GET /health`.
Mock order DB (`src/orders_db.py`) — swap for a real orders API.
Confidence-gated analyzer override: analyzer confidence < 0.8 falls
back to the agent's text inference (handles requests outside the
analyzer's trained labels, e.g. list/estimate). 20 tests passing.

**Mock MCP server** (`POST /mcp`, JSON-RPC 2.0, streamable HTTP):
`initialize`, `tools/list`, `tools/call`; 7 tools mirroring the skills.
Consumable by any MCP client; the router prefers MCP via ADK when the
agent card advertises it.

### 15.5 Central skill registry (router)

`GET /skills` — unified catalog of every skill from every reachable
worker (discovered from agent cards; 5-min TTL cache; unreachable
agents keep last-known skills flagged `stale`).
`POST /skills/refresh`, `GET /skills/{name}`,
`POST /skills/{name}/call` (ADK execution, MCP preferred).

### 15.6 Live verification (2026-10-10)

```bash
BASE=https://intent-router-1031371624665.us-central1.run.app
curl -s -X POST $BASE/route -H 'content-type: application/json' \
  -d '{"text":"where is my package, order #48291"}'
# intent=order_status → analyzer track@0.983 → adk-a2a dispatch →
# "Order #48291 is shipped. UPS tracking 1Z999AA10123456784, arriving 2026-10-14."

curl -s -X POST $BASE/route -H 'content-type: application/json' \
  -d '{"text":"cancel my order #77340"}'
# analyzer cancel@1.0 → "Order #77340 cancelled. $129.00 will be refunded."

curl -s -X POST $BASE/skills/track_order/call \
  -H 'content-type: application/json' \
  -d '{"args":{"order_id":"48291"},"protocol":"mcp"}'
# mode=adk-mcp, ok=true
```

### 15.7 Known gaps / next

- "delete my order" → analyzer says `modify` @ 0.939 (should be
  `cancel`): "delete" phrasing missing from cancel training data.
  Fix: retrain orders head or add a policy-tree rule.
- Order store is in-memory: state resets on Cloud Run scale-to-zero.
  Swap `orders_db` for a real orders API before any real use.
- `place_order` skill (new orders from catalog) in progress, not yet
  deployed.
- Dynamic skill hot-reload (`skills/skills.yaml` + mtime check, same
  pattern as the policy tree) in progress: one edit updates the agent
  card and MCP tools without redeploy; the registry re-discovers.
- First `/route` after a deploy takes ~60 s (ADK import + analyzer cold
  start); consider a startup warmup probe.
