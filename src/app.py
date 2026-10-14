"""FastAPI service exposing the hybrid (Laya System 1 + Gemma System 2) router.

Endpoints:
  GET  /health    - liveness/readiness probe (also reports resident checkpoint)
  POST /classify  - {"text": "..."} -> System 1 decision + guardrail score
                                        (+ System 2 judgment when triggered)
  POST /route     - {"text": "..."} -> full routing decision + A2A dispatch

No commercial API is required: System 1 is a self-hosted Laya checkpoint,
System 2 defaults to a mock reviewer (or local Ollama Gemma).
"""
from __future__ import annotations

import logging
import re
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .a2a_client import A2AClient
from .analyzer_client import AnalyzerClient
from .config import Settings, WORKERS_LOCAL
from .feedback import FeedbackLogger, feedback_to_dataset
from .gemma_client import GemmaReviewer
from .hybrid import MORE_INPUT_REVIEW_THRESHOLD, HybridRouter, SystemOne
from .intents import intent_names, load_intents
from .rlcd import get_job, list_jobs, start_training_job

log = logging.getLogger(__name__)

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings()
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    intents = load_intents()
    system_one = SystemOne(
        intents,
        backend=settings.system1_backend,
        checkpoint=settings.laya_checkpoint,
        device=settings.laya_device,
        preload=settings.laya_preload,
    )
    system_two = GemmaReviewer(
        intents,
        backend=settings.system2_backend,
        model_name=settings.gemma_model,
        ollama_url=settings.ollama_url,
        vertex_project=settings.vertex_project,
        vertex_region=settings.vertex_region,
        vertex_endpoint_id=settings.vertex_endpoint_id,
    )
    # Central skill registry: discovers every worker agent's card once,
    # serves the unified catalog at GET /skills.
    from .skill_registry import SkillRegistry
    skill_registry = SkillRegistry(settings.worker_agents)
    _state["skills"] = skill_registry
    # Execution plane: ADK runs every skill invocation. System 1 decides
    # WHAT; the executor handles HOW.
    from .adk_execution import AdkSkillExecutor
    executor = AdkSkillExecutor(skill_registry=skill_registry)
    _state["executor"] = executor
    router = HybridRouter(
        system_one,
        system_two,
        settings.worker_agents,
        confidence_threshold=settings.confidence_threshold,
        human_review_threshold=settings.human_review_threshold,
        guardrail_review_score=settings.guardrail_review_score,
        guardrail_block_score=settings.guardrail_block_score,
        a2a_client=A2AClient(),
        executor=executor,
        # RLCD feedback loop: log every System 2 review for human
        # review + retraining. Disable with FEEDBACK_ENABLED=false.
        feedback_logger=FeedbackLogger(
            enabled=getattr(settings, "feedback_enabled", True)),
        # Tier-2 unified intent analyzer: disabled unless ANALYZER_URL set.
        analyzer_client=AnalyzerClient(
            base_url=settings.analyzer_url,
            timeout_s=settings.analyzer_timeout_s),
    )
    _state.update(settings=settings, intents=intents, router=router)
    log.info(
        "intent-router-laya ready: system1=%s(%s) system2=%s intents=%s",
        settings.system1_backend, settings.laya_checkpoint,
        settings.system2_backend, intent_names(intents),
    )
    yield
    _state.clear()


app = FastAPI(title="Intent Router (Hybrid: Laya S1 + Gemma S2)", version="0.1.0",
              lifespan=lifespan)

REPO_ROOT = Path(__file__).resolve().parent.parent
CHAT_INDEX = REPO_ROOT / "chat" / "index.html"
REVIEW_INDEX = REPO_ROOT / "review" / "index.html"
ONBOARD_INDEX = REPO_ROOT / "onboard" / "index.html"


@app.get("/chat")
def chat():
    """Browser chat UI over POST /route (same session mechanism as the API)."""
    if not CHAT_INDEX.exists():
        raise HTTPException(status_code=404, detail="chat UI not installed")
    return FileResponse(CHAT_INDEX, media_type="text/html")


@app.get("/review")
def review():
    """Browser UI for the RLCD human review queue (approve/correct/reject)."""
    if not REVIEW_INDEX.exists():
        raise HTTPException(status_code=404, detail="review UI not installed")
    return FileResponse(REVIEW_INDEX, media_type="text/html")


@app.get("/onboard")
def onboard():
    """Browser UI to onboard a new A2A worker agent from its agent card."""
    if not ONBOARD_INDEX.exists():
        raise HTTPException(status_code=404, detail="onboard UI not installed")
    return FileResponse(ONBOARD_INDEX, media_type="text/html")


class OnboardFetchRequest(BaseModel):
    agent_url: str = Field(min_length=1, max_length=500,
                           description="Agent base URL, e.g. http://localhost:8083")


class OnboardRegisterRequest(BaseModel):
    intent: str = Field(min_length=1, max_length=64,
                        description="Router intent name, e.g. refund_status")
    agent_url: str = Field(min_length=1, max_length=500)


def _fetch_agent_card(agent_url: str) -> dict:
    """Fetch and minimally validate an A2A agent card."""
    import httpx

    base = agent_url.strip().rstrip("/")
    if not (base.startswith("http://") or base.startswith("https://")):
        raise HTTPException(status_code=400,
                            detail="agent_url must start with http:// or https://")
    url = base + "/.well-known/agent-card.json"
    try:
        resp = httpx.get(url, timeout=10.0)
        resp.raise_for_status()
        card = resp.json()
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"could not fetch agent card from {url}: {exc}") from exc
    if not isinstance(card, dict) or not card.get("name"):
        raise HTTPException(status_code=502,
                            detail="agent card is not a valid JSON object "
                                   "with a 'name' field")
    return card


def _card_summary(card: dict) -> dict:
    skills = card.get("skills") or []
    return {
        "name": card.get("name"),
        "description": card.get("description", ""),
        "version": card.get("version", ""),
        "skills": [{"name": s.get("name"), "description": s.get("description", "")}
                   for s in skills if isinstance(s, dict) and s.get("name")],
        "warnings": [
            w for w in [
                None if skills else "card lists no skills",
                None if (card.get("endpoints", {}).get("message")
                         or (card.get("protocols", {}).get("a2a") or {}).get("message"))
                else "card has no /message endpoint — the router dispatches to {url}/message",
            ] if w
        ],
    }


@app.post("/onboard/fetch")
def onboard_fetch(req: OnboardFetchRequest):
    """Fetch an agent's A2A card for preview before registering."""
    card = _fetch_agent_card(req.agent_url)
    summary = _card_summary(card)
    return {"ok": True, "agent_url": req.agent_url.strip().rstrip("/"),
            "card": summary}


_INTENT_RE = re.compile(r"^[a-z][a-z0-9_]{1,63}$")


@app.post("/onboard/register")
def onboard_register(req: OnboardRegisterRequest):
    """Register a new worker agent: intent -> agent URL.

    Validates the agent card, persists the mapping to
    data/workers.local.json (picked up on restart), hot-updates the
    live router, and refreshes the skill registry so the agent's skills
    appear in GET /skills immediately.
    """
    import json

    intent = req.intent.strip()
    if not _INTENT_RE.match(intent):
        raise HTTPException(
            status_code=400,
            detail="intent must be lowercase letters/digits/underscores, "
                   "e.g. refund_status")
    agent_url = req.agent_url.strip().rstrip("/")
    card = _fetch_agent_card(agent_url)
    summary = _card_summary(card)

    settings = _state.get("settings")
    if settings is None:
        raise HTTPException(status_code=503,
                            detail="router not initialized")
    # Live update: HybridRouter holds this same dict object, so routing
    # picks up the new intent immediately (no restart needed).
    settings.worker_agents[intent] = agent_url
    # Persist for restarts (config merges this file over the env map).
    try:
        WORKERS_LOCAL.parent.mkdir(parents=True, exist_ok=True)
        current = {}
        if WORKERS_LOCAL.exists():
            current = json.loads(WORKERS_LOCAL.read_text(encoding="utf-8"))
            if not isinstance(current, dict):
                current = {}
        current[intent] = agent_url
        WORKERS_LOCAL.write_text(json.dumps(current, indent=2) + "\n",
                                 encoding="utf-8")
    except OSError as exc:
        raise HTTPException(status_code=500,
                            detail=f"could not persist workers.local.json: {exc}") from exc
    # Skill registry: learn the new agent's card now.
    catalog = _skills().register_agent(agent_url)
    new_skills = [s for s in catalog.get("skills", [])
                  if s.get("agent_url") == agent_url]
    return {
        "ok": True,
        "intent": intent,
        "agent_url": agent_url,
        "agent": summary["name"],
        "skills_registered": [s["skill"] for s in new_skills],
        "warnings": summary["warnings"],
        "next_steps": [
            "Teach routing: add '" + intent + "' + example queries to "
            "data/intents.yaml, then run: bash scripts/laya-stack.sh train laya",
            "Teach Tier-2: in the Studio Training Data tab "
            "(http://localhost:8081/studio) create the agent spec, Save, Build, "
            "then run: bash scripts/laya-stack.sh train analyzer",
            "Restart the stack so the registration survives reboots "
            "(it is already live in this process).",
        ],
    }

# CORS for the Studio (served from the analyzer domain) calling the
# skill registry directly.
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["content-type"],
)


class QueryRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    session_id: str | None = None


def _router() -> HybridRouter:
    router = _state.get("router")
    if router is None:
        raise HTTPException(status_code=503, detail="router not initialized")
    return router


@app.get("/health")
def health():
    intents = _state.get("intents", [])
    settings = _state.get("settings")
    return {
        "status": "ok",
        "system1": settings.system1_backend if settings else None,
        "system1_checkpoint": settings.laya_checkpoint if settings else None,
        "system2": settings.system2_backend if settings else None,
        "intents": intent_names(intents),
    }


def _skills():
    reg = _state.get("skills")
    if reg is None:
        raise HTTPException(status_code=503,
                            detail="skill registry not initialized")
    return reg


@app.get("/skills")
def list_skills():
    """Unified skill catalog: every skill from every reachable agent,
    in one place."""
    return _skills().catalog()


@app.post("/skills/refresh")
def refresh_skills():
    """Force re-discovery of all agent cards now."""
    return _skills().refresh()


@app.get("/skills/{name}")
def get_skill(name: str):
    """Find which agent provides a skill and how to call it."""
    skill = _skills().find_skill(name)
    if skill is None:
        raise HTTPException(status_code=404,
                            detail=f"no agent provides skill {name!r}")
    return skill


class SkillCallRequest(BaseModel):
    args: dict = Field(default_factory=dict,
                       description="skill arguments, e.g. {'order_id': '48291'}")
    protocol: str = Field(default="auto",
                          description="mcp | a2a | auto (MCP preferred)")


@app.post("/skills/{name}/call")
def call_skill(name: str, req: SkillCallRequest):
    """Execute a skill via the ADK execution plane.

    System 1 (or the caller) decides WHAT; ADK executes HOW —
    MCP tools/call when the providing agent offers it, else A2A.
    """
    skill = _skills().find_skill(name)
    if skill is None:
        raise HTTPException(status_code=404,
                            detail=f"no agent provides skill {name!r}")
    executor = _state.get("executor")
    if executor is None:
        raise HTTPException(status_code=503, detail="executor not initialized")
    return executor.call_skill(name, skill["agent_url"],
                               args=req.args, protocol=req.protocol)


@app.post("/classify")
def classify(req: QueryRequest):
    """System 1 decision; runs System 2 as well when the policy triggers it."""
    router = _router()
    s1 = router.system_one.decide(req.text.strip())
    out = {
        "system1_intent": s1.intent,
        "confidence": s1.confidence,
        "probabilities": s1.probabilities,
        "needs_human": s1.needs_human,
        "utterance_type": s1.utterance_type,
        "guardrail_score": s1.guardrail_score,
        "guardrail_band": s1.guardrail_band,
        "worker_agent": s1.worker_agent,
        "worker_confidence": s1.worker_confidence,
        "skill_required": s1.skill_required,
        "skill_confidence": s1.skill_confidence,
        "needs_rag": s1.needs_rag,
        "needs_more_input": s1.needs_more_input,
        "needs_user_details": s1.needs_user_details,
        "is_multi_turn": s1.is_multi_turn,
        "needs_async": s1.needs_async,
        "system2_used": False,
    }
    needs_s2 = (
        s1.confidence < router.confidence_threshold
        or s1.needs_human > router.human_review_threshold
        or s1.guardrail_score >= router.guardrail_review_score
        or s1.needs_more_input >= MORE_INPUT_REVIEW_THRESHOLD
    ) and s1.guardrail_score < router.guardrail_block_score
    if needs_s2:
        s2 = router.system_two.review(req.text.strip(), s1)
        out.update(
            system2_used=True,
            system2_intent=s2.intent,
            system2_confidence=s2.confidence,
            system2_escalated=s2.escalate_to_human,
            system2_rationale=s2.rationale,
        )
    return out


@app.post("/route")
def route(req: QueryRequest):
    return _router().handle_query(req.text, session_id=req.session_id)


# ---------------------------------------------------------------- RLCD loop
# System 1 low-confidence -> System 2 (Gemma) -> human review -> retrain.


def _feedback() -> FeedbackLogger:
    fb = _router().feedback
    if fb is None:
        raise HTTPException(status_code=503, detail="feedback logging disabled")
    return fb


@app.get("/rlcd/stats")
def rlcd_stats():
    """Feedback counts + the SDR §13 KPI: System 1 fast-path rate."""
    return _feedback().stats()


@app.get("/rlcd/review")
def rlcd_review_queue(limit: int = 50):
    """Pending Gemma decisions awaiting human review (newest first)."""
    return {"pending": _feedback().list_pending(limit=limit)}


class ReviewDecision(BaseModel):
    decision: str = Field(pattern="^(approved|corrected|rejected)$")
    reviewer: str = "human"
    corrections: dict | None = None


@app.post("/rlcd/review/{record_id}")
def rlcd_review(record_id: str, req: ReviewDecision):
    """Human review gate: approve Gemma's labels, correct them, or reject."""
    ok = _feedback().review(record_id, req.decision,
                            reviewer=req.reviewer,
                            corrections=req.corrections)
    if not ok:
        raise HTTPException(status_code=404, detail="record not found")
    return {"record_id": record_id, "decision": req.decision}


class ExportRequest(BaseModel):
    out: str = "train/rlcd_feedback.jsonl"


@app.post("/rlcd/export")
def rlcd_export(req: ExportRequest):
    """Export human-approved records to training JSONL for finetune.py."""
    fb = _feedback()
    result = feedback_to_dataset(fb.path, req.out)
    return result


class TrainRequest(BaseModel):
    dataset: str = "train/rlcd_feedback.jsonl"
    out_dir: str = "models/laya_rlcd_v2"
    mode: str = Field(default="spot-vm", pattern="^(local|spot-vm)$")
    epochs: float = 4.0


@app.post("/rlcd/train")
def rlcd_train(req: TrainRequest):
    """Trigger retraining on the exported approved dataset.

    mode=local runs finetune.py in a background thread (CPU demo only).
    mode=spot-vm returns the gcloud commands for the GPU flow.
    """
    return start_training_job(req.dataset, req.out_dir,
                              mode=req.mode, epochs=req.epochs)


@app.get("/rlcd/train/{job_id}")
def rlcd_train_status(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return job


@app.get("/rlcd/train")
def rlcd_train_jobs():
    return {"jobs": list_jobs()}
