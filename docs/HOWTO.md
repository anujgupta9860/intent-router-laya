# HOWTO — Operating the Intent Router System

Live services (GCP `innovation-lab-2026`, `us-central1`):

| Service | URL |
|---------|-----|
| Router (front door) | https://intent-router-1031371624665.us-central1.run.app |
| Analyzer + Studio | https://intent-analyzer-7yydsv7ybq-uc.a.run.app |
| Studio UI | https://intent-analyzer-7yydsv7ybq-uc.a.run.app/studio |
| Order agent | https://order-agent-1031371624665.us-central1.run.app |

Repos: `intent-router-laya`, `intent-analyzer-unified`, `order-agent`
(all public under `anujgupta9860`).

---

## 1. Send a request

```bash
curl -X POST https://intent-router-1031371624665.us-central1.run.app/route \
  -H "content-type: application/json" \
  -d '{"text":"track my order 48291","session_id":"demo1"}'
```

- `session_id` is how multi-turn conversations stick together.
  Same `session_id` = same task/workflow session.
- Key response fields: `path`, `decision_made`, `decision_reason`,
  `system1_workflow`, `workflow_selected_by`, `system1_skill`,
  `skill_selected_by`, `answer`.

## 2. Add a new workflow (no code, no redeploy)

**Via Studio (recommended):** open the Studio → Workflows tab →
Add workflow. Fill in: name, agent, triggers (phrases), steps (each
with `action`, `requires` slots, `on_success`/`on_failure`). Save —
it PUTs to `/workflows` and hot-reloads instantly.

**Via YAML:** edit `workflows/workflows.yaml` in
`intent-analyzer-unified`, then either redeploy the analyzer or PUT
the file content to `/workflows`.

Workflow selection is System 1's decision 2 — triggers are the
current backend until the Laya workflow head is trained.

## 3. Add a new skill to an agent (no redeploy)

1. Edit `skills/skills.yaml` in the agent repo (e.g. `order-agent`):
   add the skill name, description, and handler.
2. The agent hot-reloads on next request (mtime check) — the agent
   card and MCP `tools/list` update automatically.
3. Force the router to re-discover: `POST /skills/refresh`,
   or wait for the registry TTL.
4. Verify in Studio → Skills tab.

Skill selection is System 1's decision 3, made AFTER the workflow's
slots are filled. The workflow step's `action` is only a hint.

## 4. Edit decision-tree policy (no redeploy)

**Via Studio:** Decision Tree tab → edit rules → save (PUT
`/policy`, hot-reloads; broken edits keep the last good tree).

**Via YAML:** edit `policy/decision-tree.yaml`, same hot-reload.

The tree handles single-task actions only — it never selects
workflows (that's System 1's job).

## 5. Test a skill directly

```bash
# list all skills
curl https://intent-router-1031371624665.us-central1.run.app/skills

# call one (ADK executes: MCP preferred, A2A fallback)
curl -X POST https://intent-router-1031371624665.us-central1.run.app/skills/track_order/call \
  -H "content-type: application/json" \
  -d '{"args":{"order_id":"48291"},"protocol":"auto"}'
```

Or use Studio → Skills tab → click a skill → Test call.

## 6. Add a new worker agent

1. Build the agent (A2A agent card at `/.well-known/agent.json`
   + `POST /message`; optional MCP at `POST /mcp`).
2. Deploy it to Cloud Run.
3. Add its URL to the router's `WORKER_AGENTS` env
   (format `agent_name=url`).
4. Redeploy the router. Its skills appear in `GET /skills` and the
   Studio automatically.

## 7. Deploy changes

```bash
# router
cd ~/workspace/intent-router-laya
gcloud builds submit \
  --tag us-central1-docker.pkg.dev/innovation-lab-2026/laya-router/intent-router:latest \
  --project=innovation-lab-2026 --quiet
gcloud run deploy intent-router \
  --image us-central1-docker.pkg.dev/innovation-lab-2026/laya-router/intent-router:latest \
  --region us-central1 --project innovation-lab-2026 --quiet

# analyzer (same pattern, image intent-analyzer/intent-analyzer)
# order-agent (same pattern, image order-agent/order-agent)
```

First request after deploy takes ~60s (cold start).

## 8. Push to GitHub

```bash
# paste a fresh PAT when asked, revoke after
git push 'https://<PAT>@github.com/anujgupta9860/<repo>.git' master
git update-ref refs/remotes/origin/master $(git rev-parse master)
```

## 9. Prove System 1 made the decisions

Every `/route` response carries:
- `decision_made` / `decision_reason` — was this turn a fresh
  decision or a continuation (`slot_fill`, `task_continuation`)?
- `workflow_selected_by: "system1"`, `skill_selected_by: "system1"`
- `system1_workflow`, `system1_skill`

Continuation turns show `decision_made: false` — proof System 1
was skipped and the locked intent carried the turn.

## 10. Known limitations

- Workflow/skill backends are trigger/registry matching until the
  Laya workflow + skill heads are trained (interfaces are ready).
- Sessions are in-memory — lost on scale-to-zero/restarts
  (Firestore planned).
- Mock order DB — resets on redeploy, not persistent.
- Studio edits (policy/workflows) are instance-local — lost on
  redeploy (shared storage planned).
