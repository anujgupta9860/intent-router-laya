# Manager Q&A — Intent Router POC

100 questions a manager can ask about this POC, with answers. Covers
architecture, training, safety, local development, deployment, cost,
limitations, and roadmap. All answers describe the system as it exists
today — where something is unproven, the answer says so.

## 1. Big picture

### 1. What is this POC in one sentence?
A routing layer that reads a user's message, decides which worker agent
should handle it and what it needs, then dispatches it — with every
decision typed, logged, and auditable instead of buried inside an LLM
prompt.

### 2. Why build a router instead of letting an LLM handle the message directly?
An LLM asked to "handle this message" mixes routing, policy, and
generation in one opaque step. The router separates them: a small
decision model routes, deterministic code enforces policy and task
state, and a generator only writes text. Each layer is testable on its
own, and a routing mistake is visible in the trace instead of hidden in
a paragraph.

### 3. Who is this for?
Any product surface where user messages must reach the right backend
capability with predictable behavior — support, orders, billing,
account — and where "the model decided" is not an acceptable audit
trail.

### 4. What does a user actually see?
A chat UI (`/chat`). They type a message; the assistant answers. A
collapsible routing trace shows path, intent, worker, analyzer decision,
task state, latency, and session for every turn.

### 5. What is "Laya"?
An open-weights "System One" decision model (ModernBERT-large based,
Apache-2.0). It answers typed questions — Choice, Noul (yes/no with
probability), Score — in a single forward pass with no text generation.
We use it as System 1: the router's decision-maker.

### 6. What is "System 2"?
Gemma, a generative model used only as a reviewer. When System 1 is
not confident enough to take the fast path, System 2 reviews the query
and System 1's decision and returns a typed judgment plus a rationale.
System 2 generates text; it does not make the final routing call alone.

### 7. Why two systems instead of one?
Speed, cost, and control. System 1 answers in one forward pass
(~tens of ms) and handles the clear cases. System 2 is slower and
more expensive, so it only sees the hard cases. The split also keeps
the common path deterministic and cheap.

### 8. What is the "analyzer" (Tier-2)?
A unified ModernBERT model with one shared encoder and a separate
classification head per agent. After System 1 picks a worker, the
analyzer answers that agent's fine-grained questions: which action, and
which flags (has order id, urgent, needs address, …). One checkpoint
serves all agents.

### 9. What is a "worker agent"?
A small service that owns one capability (orders, billing, support,
account, sales). It exposes an A2A interface (`/message`, agent card)
and a skill registry. Workers do the real work: look up orders, cancel,
place, list, estimate.

### 10. What is A2A?
Agent-to-Agent protocol: the standard envelope workers use to receive
tasks and return results. The router dispatches over A2A; workers are
interchangeable as long as they speak the protocol.

## 2. Architecture

### 11. Walk me through one message end to end.
`POST /route` → decision gate: is there an active task? If yes, the
locked intent carries the turn (no System 1 call). If no, System 1
(Laya) decides intent, worker, skill, and four Noul flags → if the
fast-path gates pass, dispatch; else System 2 reviews → Tier-2
analyzer classifies the action within the worker → deterministic
workflow / task framework fills slots → ADK executes the skill → worker
responds → trace recorded.

### 12. What are System 1's 11 decisions?
intent (Choice), human_review (Noul), utterance_type (Choice),
guardrail_risk (Score), worker_agent (Choice), skill_required (Choice),
needs_rag, needs_more_input, needs_user_details, is_multi_turn,
needs_async (all Noul).

### 13. What is a "Choice / Noul / Score" question?
The three typed question kinds Laya answers. Choice: pick one option
from a list. Noul: probability that a statement is true (yes/no with
calibration). Score: expected value over an ordered rubric
(e.g. safe → critical). No free text anywhere in a decision.

### 14. Why typed decisions instead of free-text LLM output?
Free text has to be parsed, and parsing fails. Typed outputs are
machine-checkable: the router can validate ranges, compare against
thresholds, and log them exactly. A decision is a data record, not a
paragraph to interpret.

### 15. What is the decision gate?
A deterministic rule: if a task is active for this session, the locked
intent carries the turn (slot fill, recovery, abort) with NO System 1
call. System 1 is invoked only when a genuine new decision is needed.
This is what stops the models from re-deciding mid-task.

### 16. What is a task, and how is it different from a workflow?
A task is the per-session state machine for one user goal (e.g. cancel
order): slots, status (started → inprogress → completed | aborted).
A workflow is a reusable multi-step definition (e.g.
return_and_replace) that the task framework executes. Tasks are
instances; workflows are templates.

### 17. What is the policy decision tree?
A layman-configurable YAML rule set (`policy/decision-tree.yaml`,
first-match-wins) that maps analyzer output to an action. It is
hot-reloaded on every decision, validated before saving, and edited in
Studio. It is the human-readable policy layer between model output and
dispatch.

### 18. What is Studio?
The single unified web UI for the whole system (one URL, all agents):
Decision Tree editor, Workflows editor, Skills browser, Training Data
tab, and feedback review. Served by the analyzer at `/studio`.

### 19. What is ADK, and what does it do here?
Google's Agent Development Kit. The router's execution plane uses ADK
to invoke skills (FunctionTool) and to dispatch over A2A. ADK is the
"hands"; the models are the "brain"; the gate and task framework are
the "rules".

### 20. What is the MCP server in the order agent?
A mock Model Context Protocol server exposing 9 tools (order lookup,
cancel, modify, …). It stands in for a real backend. The worker calls
it through the same tool interface a production backend would use.

### 21. How do the router, analyzer, and workers communicate?
HTTP + JSON. Router → analyzer: `POST /analyze`, `POST /decide`,
`GET /workflows`. Router → worker: A2A `POST /message`. Everything is
plain request/response; no message bus, no shared database in the POC.

### 22. How is session state stored?
In memory, keyed by `session_id`. The chat UI sends the session id back
with each turn. Sessions are lost on restart / scale-to-zero; Firestore
is the planned durable store.

### 23. What is the "fallback" worker?
The worker that receives messages no intent matched, or messages Laya
answered with an unknown intent. It is the safe default, not an error:
every message gets a handling path.

### 24. What happens when System 1 returns an intent that isn't in the taxonomy?
The client logs a warning (`Laya returned unknown intent …; using
'fallback'`) and routes to the fallback worker. The event is visible in
`logs/router.log`.

### 25. What is "skill selection", and is it model inference?
No — it is deterministic policy/registry logic. The router looks up
the skill in the registry by name and validates it. The model proposes
`skill_required`; the registry decides what actually runs.

## 3. Routing and decision paths

### 26. What are the possible `path` values in a trace?
`fast` (System 1 decided alone), `system2` (System 2 reviewed),
`task_continuation` (active task carried the turn, no model call),
`workflow`, `workflow_abort`, `guardrail_block`, `empty`.

### 27. What exactly gates the fast path?
All four must hold: System 1 confidence ≥ 0.6, needs_human ≤ 0.5,
guardrail_score below the review threshold, needs_more_input < 0.8.
Any one failing sends the turn to System 2.

### 28. Why did "Where is my order?" go to System 2?
No order number in the text. The training data labels order queries
without a number as needing more input, so Laya predicts high
needs_more_input and the fast-path gate fails. The worker then asks
for the ID (no-ID recovery).

### 29. Why did "What is the delivery status of order #12345?" go to System 2 in one test?
Same query shape as a training template labeled "needs more input",
with a number inserted. If Laya pattern-matches the sentence shape
more strongly than the number-presence signal, needs_more_input stays
high. Fix: training examples with the number in different positions.

### 30. Why did "list all my orders" go to System 2 under the mock?
The mock had a blanket rule: any order_status query without a number
gets needs_more_input = 0.9. "List all" needs no number, but the mock
couldn't tell. Fixed by exempting list/show queries in the mock. The
trained Laya should learn the distinction from data.

### 31. What is `task_continuation`, and why is it fast?
When a task is active, the turn skips System 1 and the analyzer
entirely — the locked intent carries it. Example: the worker asks for
an order ID, the user replies "1231211", and that bare number just
fills the `order_id` slot. Latency drops (observed ~13ms vs ~58ms)
because no model is called.

### 32. Can System 2 override System 1's intent?
No — by architectural decision. System 2 reviews and returns a typed
judgment plus rationale, but it cannot change the intent or force
escalation. The final routing call stays with System 1 + the gate.

### 33. Can System 2 change routing at all?
It influences the outcome when the fast path fails: its judgment feeds
the dispatch decision and is logged as a training label. But it cannot
rewrite the intent, and the deterministic gate still applies.

### 34. What is a guardrail block?
When the guardrail score crosses the block threshold, the turn is
refused before dispatch (`guardrail_block`). The guardrail is a
learned Score head in Laya (trained), not a keyword list.

### 35. What is `workflow_abort`?
The user abandoned a multi-step workflow mid-way (e.g. said "never
mind"). The workflow ends as aborted; the next message is routed fresh.

### 36. What is `empty`?
The message carried no routable content (empty / whitespace). No
decision is made.

### 37. How does the router handle "cancel it" with no prior context?
"it" has no referent. The follow-up / pronoun-fragment pattern gets
high needs_more_input and high is_multi_turn, so it goes to System 2,
which asks for clarification. Without an active task, there is nothing
to attach "it" to.

### 38. How does the router handle a bare number like "48291"?
If a task is active and expects an order ID, the number fills the slot
(`task_continuation`). If no task is active, it is treated as a fresh
query — likely fallback, since a bare number matches no intent.

### 39. What is no-ID recovery?
When the worker needs an order ID and the user didn't give one, it
asks for it — or lists the user's orders so they can pick. It is a
worker behavior, not a model decision.

### 40. How are multi-turn conversations kept coherent?
Session id + task state. The task framework holds slots across turns;
the decision gate prevents re-routing mid-task. The user never
re-states the goal.

## 4. Training — Laya (System 1)

### 41. How is Laya trained?
RLCD — reinforcement learning from calibrated decisions. The model
samples noisy logit distributions (GRPO-style groups), scores them
with a proper scoring rule (spherical + ranked-probability), and
updates by policy gradient plus a cross-entropy guidance term.
Post-training, per-question-type temperature calibration is fitted on
a held-out slice.

### 42. What is the training data?
Synthetic labeled decisions generated from `data/intents.yaml`:
seed example queries per intent, expanded by paraphrase templates,
with labels derived deterministically (intent, worker, skill, Noul
flags, guardrail band). ~600 rows → ~6,600 sequences (one per
question per row).

### 43. Where does the "hard" training data come from?
The feedback loop. Every System 2 turn is logged to
`feedback/fallbacks.jsonl` — by definition the cases System 1 found
hard. After human review, approved records become training rows.

### 44. What does `train laya` do?
One command, end to end: build dataset → convert to RLCD format →
preprocess (tokenize) → 4-epoch RLCD fine-tune on MPS → temperature
calibration → export to `checkpoints/laya_finetuned`.

### 45. What does `train laya-feedback` do?
Same as `train laya`, but mixes in the human-approved System 2
feedback before training. Pending (unreviewed) records are skipped.
This is the RLCD loop closing: production hard cases become training
data.

### 46. How long does Laya training take?
Roughly 1–2 hours on Apple Silicon (MPS) for 4 epochs. On CUDA, the
DDP script (`train_single_gpu.py`) runs the same loop. A rolling
checkpoint is written after every epoch, so an interruption loses at
most one epoch.

### 47. What is the base model Laya fine-tunes from?
`convaiinnovations/laya` (ModernBERT-large based). The fine-tune
starts from the base, not from the vendor's `laya-typed-decisions`
checkpoint — our RLCD data teaches our taxonomy from scratch.

### 48. How do you know the fine-tune worked?
Smoke tests: verbatim seed examples should route with high
confidence; the review queue should shrink over retrains; the
`/rlcd/stats` fast-path rate should rise. Full accuracy evaluation
against a held-out labeled set is still manual — there is no automated
eval gate yet.

### 49. What is temperature calibration, and why does it matter?
After training, a temperature per question type (choice / score /
noul) is fitted on held-out data so the model's probabilities mean
what they say. Without it, a "0.9 confidence" might be right only 60%
of the time. Calibrated confidence is what the fast-path gates rely on.

### 50. What happens when a new intent is added?
Laya must be retrained — the new intent is a new Choice option it has
never seen. Add the intent + examples to `data/intents.yaml`, then
`train laya`. Until retrained, System 1 cannot route to the new intent.

### 51. What happens when a new agent is added?
Both models retrain: the analyzer (new per-agent head) via
`train analyzer`, and Laya (new intent/worker option) via `train laya`.
Plus the worker scaffold (`new-agent`) and its skill implementation.

### 52. What happens when a new label is added inside an existing agent?
Only the analyzer retrains (`train analyzer`). Routing is unchanged —
System 1 still sends those messages to the same worker; only the
analyzer's choice within that worker grew a new option. No Laya
retrain needed.

### 53. Can Laya be trained on real user queries instead of synthetic data?
Yes — that is the point of the feedback loop. Approved feedback rows
are real queries with trusted labels. Over time the mix shifts from
synthetic to real. The review gate (human approve/correct) is what
keeps bad labels out.

### 54. What is the human review gate, exactly?
Every feedback record sits at `review_status: pending` until a human
approves, corrects (with label overrides), or rejects it. Export
(`feedback_to_dataset`) includes only approved/corrected records.
Nothing trains on unreviewed data.

### 55. Where is the review UI?
`http://localhost:8080/review` (router, local). It shows the pending
queue with System 1 vs System 2 side by side, and Approve / Correct /
Reject buttons. Stats come from `GET /rlcd/stats`.

## 5. Training — analyzer (Tier-2)

### 56. How is the analyzer trained?
Standard supervised fine-tuning: one shared ModernBERT encoder, one
classification head per agent, 5 epochs on synthetic per-agent
datasets (seed queries per label + paraphrases). Checkpoint at
`models/unified/` (~600MB).

### 57. What is in an analyzer training dataset?
One JSONL per agent (`train/data/<agent>.jsonl`): query, choice label,
Noul flag values. Built from the agent spec (`agents/<agent>.json`)
via the Training Data tab: Save spec → Build dataset.

### 58. How do you add a new agent to the analyzer?
Studio → Training Data tab → define the agent (choice question,
labels + criteria, Noul flags, seed queries) → Save → Build dataset →
`train analyzer`. The new agent's head is added automatically.

### 59. How do you add a new label to an existing agent?
Training Data tab → select the agent → add label + criteria + seed
queries → Save → Build dataset → `train analyzer`. No router or Laya
changes.

### 60. What is the analyzer's confidence threshold?
Below it, `needs_human=true` and the decision is logged to
`feedback/analyzer_fallbacks.jsonl` for review. Same review-gate
pattern as the router, scoped to Tier-2.

### 61. What is `analyzer_fallbacks.jsonl`?
The analyzer-side feedback log: low-confidence Tier-2 decisions, with
agent, query, full decision, and router context. The router-side
`fallbacks.jsonl` logs System 2 turns; this one logs analyzer
uncertainty.

### 62. Why a shared encoder instead of one model per agent?
One checkpoint, one serving process, shared representation learning
across agents. A new agent adds a head, not a whole model. Serving
cost stays flat as agents are added.

### 63. What is the calibration fix in the analyzer?
An MPS/CPU device bug in the calibration loop (tensors on the wrong
device) that crashed training after the main loop. Fixed by moving
batch tensors to the model device and logits back to CPU. Same fix
applied to the router encoder trainer.

### 64. Does the analyzer need a GPU to serve?
No. Inference is a single forward pass; CPU serving works. Training
is faster on MPS/CUDA but also runs on CPU.

### 65. What is `/decide`?
Analyzer + policy tree in one call: the analyzer classifies, the
decision tree maps the output to an action. The router uses it at the
workflow level.

## 6. Safety, guardrails, human review

### 66. What stops the model from doing something dangerous?
Layers: (a) guardrail Score head — high-risk queries block before
dispatch; (b) deterministic gate — models cannot override task state;
(c) policy tree — human-readable rules between model and action;
(d) System 2 cannot override intent; (e) human review gate on
training data. No single layer is the whole story.

### 67. What is `needs_human`, and who acts on it?
A Noul probability that a human should review. Above 0.5, the fast
path fails and System 2 reviews. It is a routing signal, not a
ticket — in the POC there is no human-in-the-loop queue for live
traffic, only the training-data review gate.

### 68. What does "System 2 cannot override routing" mean in practice?
System 2's judgment is an input to the dispatch decision, but it
cannot rewrite the intent, and the gate + task framework still apply.
The architecture was changed to this explicitly: an earlier design
let System 2 override, and it was removed.

### 69. Is there proof System 2 cannot change routing?
Not yet at the standard we'd want. The honest gap list includes:
immutable decision IDs/events, invocation counters, backend/model
identifiers in traces, strict provenance, and conflict tests proving
System 2 cannot change routing. The code enforces it; the proof
harness doesn't exist yet.

### 70. What is logged for every decision?
Router: trace (path, intent, confidence, worker, analyzer, task,
latency, session) + feedback records for System 2 turns (all 11 S1
decisions + all 11 S2 decisions + rationale). Analyzer: decision +
low-confidence feedback records. Service logs in `<base>/logs/`.

### 71. Where do System 2 decisions get logged?
`feedback/fallbacks.jsonl` in the router repo (gitignored, local
only). One JSON object per line: id, timestamp, query, S1 block, S2
block, usable_label, review_status.

### 72. What is `usable_label`?
False when System 2 escalated to a human (no trustworthy label) or
the query was a guardrail block / empty. Only usable records can
become training data — and only after human approval.

### 73. What if the feedback log contains a bad label?
The review gate catches it: a human rejects the record, and export
skips it. A corrected record uses the human's overrides. Unreviewed
records never train.

### 74. Is any user data sent to a third party?
No, in the local setup. Laya runs locally, the analyzer runs
locally, Ollama Gemma runs locally. Nothing leaves the machine. The
cloud deployment (Cloud Run) runs the same self-hosted models — no
commercial LLM API is called in either setup.

### 75. What about PII in the feedback logs?
Feedback logs contain raw user queries. In the POC they are local
files. Before any production use, they need retention + redaction
policy — that does not exist yet.

## 7. Local development

### 76. How do I run the whole stack locally?
`bash scripts/laya-stack.sh setup` (once), then
`bash scripts/laya-stack.sh run`. Chat at
http://localhost:8080/chat, Studio at http://localhost:8081/studio,
Review at http://localhost:8080/review. `stop` shuts it down.

### 77. What does `setup` do?
Clones/updates the 3 repos, creates a `.venv` per repo, installs
requirements (+ torch), optionally downloads the analyzer checkpoint
from GCS and the fine-tuned Laya checkpoint from GCS. Auto-installs
uv and a uv-managed Python 3.12 if the system Python is too old.

### 78. What is `laya-stack.sh`?
One portable bash script that sets up, runs, trains, and scaffolds
the whole stack. It detects the checkout it was run from (e.g.
`~/IdeaProjects`) and operates on the repos next to it — no hardcoded
`~/laya-local`. Explicit `LAYA_BASE` still overrides.

### 79. What are the system requirements?
macOS or Linux, git, curl, bash. Python ≥ 3.11 (auto-installed via uv
if older). Apple Silicon uses MPS (Metal GPU) for training; CUDA
machines use the DDP script. ~2GB free for checkpoints.

### 80. What is `new-agent`?
`bash scripts/laya-stack.sh new-agent <name> [intent] [port]`.
Scaffolds a real A2A worker: FastAPI `/health`, `/message`, agent
card, `skills.yaml`, README, venv + deps, `worker.json`. You fill in
`handle()`. `run` auto-discovers it.

### 81. How do I run with real Laya instead of the mock?
`SYSTEM1_BACKEND=laya bash scripts/laya-stack.sh run`. Uses the
fine-tuned checkpoint if downloaded, else the base HF checkpoint
(zero-shot). The mock is still the default when the variable is unset.

### 82. How do I run with real System 2 instead of the mock?
`SYSTEM2_BACKEND=ollama bash scripts/laya-stack.sh run` (after
`brew install --cask ollama` + `ollama pull gemma3:4b`). The router
POSTs to Ollama with temperature 0.

### 83. What if Ollama isn't installed / isn't running?
Nothing breaks. `SYSTEM2_BACKEND` defaults to `mock`, so the mock
reviewer is used unless you explicitly set `ollama`. If you set
`ollama` but Ollama is unreachable, the router logs an error, prints
a warning at startup, and System 2 fails safe — it returns an
"escalate to human" judgment instead of crashing. The mock is
untouched by the Ollama wiring.

### 84. Can I run both real models at once?
Yes: `SYSTEM1_BACKEND=laya SYSTEM2_BACKEND=ollama bash
scripts/laya-stack.sh run`. No mocks anywhere in the decision path.

### 85. Where are the logs?
`<base>/logs/` — `router.log`, `analyzer.log`, `<worker>.log`.
`<base>` is the parent of your checkout (e.g. `~/IdeaProjects/logs`).
`run` prints "Logs in …" at startup.

### 86. How do I add a worker agent + dataset, end to end?
Studio Training Data → define agent → Save → Build dataset →
`new-agent <name> <intent>` → implement `handle()` + skills →
add intent to router `data/intents.yaml` → `train analyzer` →
`train laya` → commit + push → `run`.

### 87. How do I add a new intent to an existing agent's dataset?
Training Data tab → select agent → add label + criteria + seed
queries → Save → Build dataset → `train analyzer`. No Laya retrain:
routing is unchanged.

### 88. Where does the decision tree land when I edit it in Studio?
`policy/decision-tree.yaml` in the analyzer repo. Hot-reloaded on
every decision; broken edits keep the last good tree. Commit + push
it like any repo file.

### 89. What is the Training Data tab?
The fourth Studio tab. It manages agent specs (`agents/<agent>.json`),
builds per-agent training datasets, and exposes save/list/build
endpoints. It is how non-engineers define a new agent's taxonomy
without touching code.

### 90. What is `train router` (encoder track)?
Fine-tunes a ModernBERT intent classifier (Track A). It learns the
intent only; needs_human / utterance_type / guardrail are derived
heuristically. Served as `SYSTEM1_BACKEND=encoder`. Useful to prove
the serving path; Laya (Track B) is the production System 1.

## 8. Deployment, cost, performance

### 91. Where is this deployed?
Google Cloud Run (project `innovation-lab-2026`, `us-central1`):
router, analyzer/Studio, order agent. Checkpoints in GCS
(`gs://laya-checkpoints-anuj/`). The local stack is the development
environment; Cloud Run is the demo deployment.

### 92. What is the canary?
Laya serves 10% of production traffic; 90% goes to the rollback
revision. Promotion to 100% requires explicit approval. The canary is
the production validation of the fine-tuned Laya checkpoint.

### 93. What does it cost to run?
Local: $0 (own hardware). Cloud Run: scale-to-zero, pay per
request — small for a demo. Training: ~1–2 hours on a local Mac (free)
or a few dollars on an on-demand T4. No per-call LLM billing anywhere
— all models are self-hosted.

### 94. What is the latency?
System 1: one forward pass, ~tens of ms. End-to-end traces observed
locally: ~13ms (task continuation, no model call) to ~300ms (full
path with analyzer + worker). System 2 (Ollama) adds generation time
on the slow path only.

### 95. How does it scale?
Workers are independent services; the router discovers them from
`worker.json`. Adding a worker is a scaffold + a retrain, not a
router rewrite. The analyzer serves all agents from one checkpoint.
Horizontal scaling is standard Cloud Run / container scaling — not
yet load-tested.

### 96. What is the model size?
Laya: ~421M params (ModernBERT-large). Analyzer checkpoint: ~600MB.
Both fit comfortably on a laptop and on small Cloud Run instances.

## 9. Limitations, gaps, roadmap

### 97. What is unproven or incomplete? (Honest list)
- A complete `return_and_replace` workflow end to end.
- Strong System 1 proof: immutable decision IDs, invocation
  counters, backend/model identifiers in traces, strict provenance,
  conflict tests proving System 2 cannot change routing.
- Registry/action mismatch: registry says `return_order` /
  `place_order` / `cancel_order`; the order agent implements
  `return` / `place` / `cancel`. Works today via mapping, but the
  names should be unified.
- No automated accuracy eval gate for retrains.
- Session state is in-memory (lost on restart); Firestore planned.
- Feedback logs have no retention / redaction policy.
- The public Cloud Run Studio has no auth layer on its write
  endpoints — the Training Data tab is not deployed publicly for
  this reason.

### 98. What breaks if the analyzer is down?
The router still routes (System 1 + System 2), but Tier-2
classification is unavailable. Depending on configuration, the router
degrades (no analyzer choice) rather than failing outright. The
analyzer is a hard dependency for action-level accuracy, not for
basic routing.

### 99. What breaks if Laya is down / the checkpoint is missing?
The router falls back to the configured backend. If
`SYSTEM1_BACKEND=laya` but the checkpoint can't load, serving fails
at startup — there is no silent fallback to mock in that path (by
design: silent fallbacks hide outages). Set the backend explicitly
and watch startup logs.

### 100. What is the roadmap?
1. Close the proof gaps in Q97 (decision IDs, conflict tests).
2. Automated eval gate for every retrain.
3. Firestore sessions + feedback retention/redaction.
4. Auth on public Studio write endpoints, then deploy the Training
   Data tab.
5. Shift training mix from synthetic toward reviewed real queries.
6. Canary promotion decision (10% → 100%) with eval evidence.
