"""Hybrid intent router: self-hosted Laya is System 1, Gemma is System 2.

Routing policy (every query):

  1. System 1 (Laya, ~25-45 ms local) decides: intent + confidence, Noul
     human-review probability, utterance type, and the Score-based
     guardrail risk (0=safe .. 4=critical).
  2. Guardrail block:  score >= GUARDRAIL_BLOCK_SCORE  -> fallback worker
     immediately. System 2 is not consulted; the risk is too high to
     spend seconds deliberating.
  3. Fast path: confidence >= CONFIDENCE_THRESHOLD and Noul <=
     HUMAN_REVIEW_THRESHOLD and score < GUARDRAIL_REVIEW_SCORE ->
     route straight to the intent's worker. No API cost, no LLM latency.
  4. Slow path: anything else -> System 2 (Gemma) reviews System 1's
     read and makes the final call: confirm/override the intent, or
     escalate to a human (fallback worker).

Nothing here calls a commercial API on the fast path: Laya runs on your
own hardware, so the marginal cost of a System 1 decision is zero.
"""
from __future__ import annotations

import logging
import re
import time

from .a2a_client import A2AClient
from .gemma_client import GemmaReviewer, SystemTwoJudgment
from .intents import Intent
from .laya_client import (
    GUARDRAIL_RUBRIC,
    LayaClient,
    LayaError,
    SystemOneDecision,
)
from .router_util import worker_name_for

log = logging.getLogger(__name__)

#: Keyword lists for the mock System 1. Keep aligned with data/intents.yaml.
_MOCK_KEYWORDS: dict[str, list[str]] = {
    "billing_inquiry": [
        "bill", "billing", "invoice", "charge", "charged", "payment",
        "refund", "receipt", "overcharge", "price",
    ],
    "technical_support": [
        "error", "bug", "crash", "broken", "not working", "failed",
        "failure", "issue", "slow", "login", "log in", "password",
        "reset", "trouble",
    ],
    "sales_question": [
        "pricing", "plan", "quote", "demo", "trial", "buy", "purchase",
        "discount", "enterprise", "upgrade", "features",
    ],
    "account_update": [
        "update", "change", "email", "address", "phone", "profile",
        "account", "settings", "username",
    ],
    "order_status": [
        "order", "shipping", "delivery", "deliver", "track", "tracking",
        "package", "shipped", "arrive",
    ],
    "fallback": [],
}

#: Mock guardrail: risky phrases -> score on the 0..4 rubric scale.
#: A real Laya Score question judges this semantically; the mock uses
#: keywords so the guardrail path is testable offline.
_MOCK_RISK_PHRASES: list[tuple[float, list[str]]] = [
    (3.6, ["delete my account", "close my account", "wire transfer",
           "send money", "social security", "ssn"]),
    (2.2, ["refund", "payment", "charge my card", "cancel my",
           "credit card"]),
    (1.1, ["password", "reset", "login", "log in"]),
]

_IMPERATIVE_RE = re.compile(
    r"^(please\s+)?(cancel|update|change|send|track|give|show|get|set|add|"
    r"remove|delete|open|close|start|stop|book|order|tell|refund|pay)\b",
    re.IGNORECASE,
)


def mock_utterance_type(text: str) -> str:
    t = text.strip()
    if t.endswith("?"):
        return "question"
    if _IMPERATIVE_RE.match(t):
        return "command"
    if len(t.split()) <= 2 or not re.search(r"[a-z]{3,}", t.lower()):
        return "other"
    return "statement"


def mock_guardrail_score(text: str) -> float:
    lowered = text.lower()
    score = 0.2
    for value, phrases in _MOCK_RISK_PHRASES:
        if any(p in lowered for p in phrases):
            score = max(score, value)
    return score


def mock_system_one(text: str, intents: list[Intent]) -> SystemOneDecision:
    """Deterministic System 1 stand-in shaped like a Laya response."""
    lowered = text.lower()
    scores: dict[str, int] = {}
    for intent in intents:
        kws = _MOCK_KEYWORDS.get(intent.name, [])
        scores[intent.name] = sum(1 for kw in kws if kw in lowered)
    total = sum(scores.values())
    utterance = mock_utterance_type(text)
    guardrail_score = mock_guardrail_score(text)
    if total == 0:
        probs = {i.name: 1.0 / len(intents) for i in intents}
        return SystemOneDecision(
            intent="fallback", probabilities=probs, confidence=0.25,
            needs_human=0.85, utterance_type=utterance,
            utterance_confidence=0.6, guardrail_score=guardrail_score,
            model="mock", latency_ms=1.0,
        )
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top_intent, top_hits = ranked[0]
    second_hits = ranked[1][1] if len(ranked) > 1 else 0
    probs = {name: hits / total for name, hits in scores.items()}
    margin = top_hits - second_hits
    confidence = min(0.95, 0.55 + 0.10 * margin + 0.03 * top_hits)
    needs_human = 0.75 if (margin < 2 and top_hits >= 2) else 0.15
    if top_intent == "fallback":
        needs_human = 0.85
    return SystemOneDecision(
        intent=top_intent,
        probabilities={k: round(v, 3) for k, v in probs.items()},
        confidence=round(confidence, 3),
        needs_human=needs_human,
        utterance_type=utterance,
        utterance_confidence=0.7,
        guardrail_score=guardrail_score,
        model="mock",
        latency_ms=1.0,
    )


class SystemOne:
    """System 1 front-end: self-hosted Laya, a self-trained encoder, or mock."""

    def __init__(
        self,
        intents: list[Intent],
        *,
        backend: str = "mock",
        checkpoint: str = "",
        device: str = "auto",
        preload: bool = False,
    ) -> None:
        if backend not in ("mock", "laya", "encoder"):
            raise ValueError(f"unknown System 1 backend {backend!r}")
        self.intents = intents
        self.backend = backend
        self.client: LayaClient | None = None
        self.encoder = None
        if backend == "laya":
            from .laya_client import DEFAULT_CHECKPOINT

            self.client = LayaClient(
                checkpoint or DEFAULT_CHECKPOINT,
                device=device,
                preload=preload,
            )
        elif backend == "encoder":
            from .encoder_client import EncoderClient

            if not checkpoint:
                raise ValueError(
                    "LAYA_CHECKPOINT must point at a trained checkpoint "
                    "when SYSTEM1_BACKEND=encoder (see train/finetune.py)"
                )
            self.encoder = EncoderClient(checkpoint)

    def decide(self, text: str) -> SystemOneDecision:
        if self.backend == "mock":
            return mock_system_one(text, self.intents)
        try:
            if self.backend == "laya":
                assert self.client is not None
                return self.client.decide(text, self.intents)
            assert self.encoder is not None
            return self.encoder.decide(text, self.intents)
        except Exception as exc:
            # Local engine failure: fail SAFE. Maximum guardrail score
            # forces the block path -> fallback worker.
            log.error("System 1 (%s) failed: %s", self.backend, exc)
            probs = {i.name: 1.0 / len(self.intents) for i in self.intents}
            return SystemOneDecision(
                intent="fallback", probabilities=probs, confidence=0.2,
                needs_human=0.9, utterance_type=mock_utterance_type(text),
                utterance_confidence=0.5,
                guardrail_score=float(len(GUARDRAIL_RUBRIC) - 1),
                model=f"{self.backend}-error", latency_ms=0.0,
            )


class HybridRouter:
    """Orchestrates System 1 -> policy -> optional System 2 -> A2A dispatch."""

    def __init__(
        self,
        system_one: SystemOne,
        system_two: GemmaReviewer,
        worker_agents: dict[str, str],
        *,
        confidence_threshold: float = 0.6,
        human_review_threshold: float = 0.5,
        guardrail_review_score: float = 1.5,
        guardrail_block_score: float = 3.0,
        a2a_client: A2AClient | None = None,
    ) -> None:
        self.system_one = system_one
        self.system_two = system_two
        self.worker_agents = worker_agents
        self.confidence_threshold = confidence_threshold
        self.human_review_threshold = human_review_threshold
        self.guardrail_review_score = guardrail_review_score
        self.guardrail_block_score = guardrail_block_score
        self.a2a = a2a_client or A2AClient()

    # ------------------------------------------------------------------ API
    def resolve_worker(self, intent: str) -> str:
        if intent in self.worker_agents:
            return self.worker_agents[intent]
        if "fallback" in self.worker_agents:
            return self.worker_agents["fallback"]
        return next(iter(self.worker_agents.values()))

    def handle_query(self, text: str) -> dict:
        started = time.monotonic()
        text = (text or "").strip()
        if not text:
            return self._respond(
                text, self._empty_system_one(), None,
                path="empty", started=started,
            )

        s1 = self.system_one.decide(text)

        # --- policy ---
        if s1.guardrail_score >= self.guardrail_block_score:
            return self._respond(
                text, s1, None, path="guardrail_block", started=started,
            )

        fast_path = (
            s1.confidence >= self.confidence_threshold
            and s1.needs_human <= self.human_review_threshold
            and s1.guardrail_score < self.guardrail_review_score
        )
        s2: SystemTwoJudgment | None = None
        if not fast_path:
            s2 = self.system_two.review(text, s1)

        return self._respond(text, s1, s2, path="fast" if fast_path else "system2",
                             started=started)

    # -------------------------------------------------------------- internals
    def _empty_system_one(self) -> SystemOneDecision:
        return SystemOneDecision(
            intent="fallback", confidence=0.0, needs_human=0.0,
            utterance_type="other", guardrail_score=0.0, model="mock",
        )

    def _respond(
        self,
        text: str,
        s1: SystemOneDecision,
        s2: SystemTwoJudgment | None,
        *,
        path: str,
        started: float,
    ) -> dict:
        gates = {
            "low_confidence": s1.confidence < self.confidence_threshold,
            "human_review": s1.needs_human > self.human_review_threshold,
            "guardrail_review": s1.guardrail_score >= self.guardrail_review_score,
            "guardrail_block": s1.guardrail_score >= self.guardrail_block_score,
        }

        if path == "guardrail_block":
            intent, routed = "fallback", True
            rationale = (
                f"Guardrail block: risk score {s1.guardrail_score:.2f} "
                f"(band '{s1.guardrail_band}') >= block threshold "
                f"{self.guardrail_block_score}."
            )
        elif path == "system2" and s2 is not None:
            if s2.escalate_to_human:
                intent, routed = "fallback", True
            else:
                intent, routed = s2.intent, False
            rationale = s2.rationale
        elif path == "empty":
            intent, routed, rationale = "fallback", True, "empty query"
        else:  # fast path
            intent, routed = s1.intent, False
            rationale = (
                f"System 1 fast path: confidence {s1.confidence:.2f}, "
                f"risk band '{s1.guardrail_band}'."
            )

        final_intent = "fallback" if routed else intent
        worker_url = self.resolve_worker(final_intent)
        dispatch = self.a2a.dispatch(worker_url, text, final_intent)

        return {
            "path": path,  # fast | system2 | guardrail_block | empty
            "intent": final_intent,
            "system1_intent": s1.intent,
            "confidence": round(s1.confidence, 3),
            "probabilities": {
                k: round(v, 3) for k, v in s1.probabilities.items()
            },
            "needs_human": round(s1.needs_human, 3),
            "utterance_type": s1.utterance_type,
            "guardrail_score": s1.guardrail_score,
            "guardrail_band": s1.guardrail_band,
            "gates": gates,
            "system2_used": s2 is not None,
            "system2_intent": s2.intent if s2 else None,
            "system2_escalated": s2.escalate_to_human if s2 else False,
            "system2_rationale": rationale,
            "routed_to_fallback": routed,
            "worker": worker_url,
            "worker_name": worker_name_for(worker_url),
            "task_id": dispatch["task_id"],
            "dispatch": dispatch,
            "latency_ms": round((time.monotonic() - started) * 1000.0, 1),
            "ts": time.time(),
        }
