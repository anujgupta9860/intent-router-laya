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
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .a2a_client import A2AClient
from .config import Settings
from .gemma_client import GemmaReviewer
from .hybrid import HybridRouter, SystemOne
from .intents import intent_names, load_intents

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
    router = HybridRouter(
        system_one,
        system_two,
        settings.worker_agents,
        confidence_threshold=settings.confidence_threshold,
        human_review_threshold=settings.human_review_threshold,
        guardrail_review_score=settings.guardrail_review_score,
        guardrail_block_score=settings.guardrail_block_score,
        a2a_client=A2AClient(),
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


class QueryRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)


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
        "system2_used": False,
    }
    needs_s2 = (
        s1.confidence < router.confidence_threshold
        or s1.needs_human > router.human_review_threshold
        or s1.guardrail_score >= router.guardrail_review_score
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
    return _router().handle_query(req.text)
