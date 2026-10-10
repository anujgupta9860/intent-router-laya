"""Hybrid intent router: self-hosted Laya is System 1, Gemma is System 2.

Routing policy (every query):

  1. System 1 (Laya, ~25-45 ms local) decides eleven typed questions:
     intent + confidence, Noul human-review probability, utterance type,
     the Score-based guardrail risk (0=safe .. 4=critical), plus the
     routing decisions: worker_agent, skill_required, needs_rag,
     needs_more_input, needs_user_details, is_multi_turn, needs_async.
  2. Guardrail block:  score >= GUARDRAIL_BLOCK_SCORE  -> fallback worker
     immediately. System 2 is not consulted; the risk is too high to
     spend seconds deliberating.
  3. Fast path: confidence >= CONFIDENCE_THRESHOLD and Noul <=
     HUMAN_REVIEW_THRESHOLD and score < GUARDRAIL_REVIEW_SCORE and
     needs_more_input < MORE_INPUT_REVIEW_THRESHOLD -> route straight
     to the worker_agent's worker. No API cost, no LLM latency.
  4. Slow path: anything else -> System 2 (Gemma) reviews System 1's
     read and makes the final call: confirm/override the intent, or
     escalate to a human (fallback worker).

The worker is resolved from System 1's worker_agent decision (falling
back to the intent->worker mapping); skill_required and the Noul flags
ride along in the response so workers can adapt (RAG lookup, async
dispatch, clarification).

Nothing here calls a commercial API on the fast path: Laya runs on your
own hardware, so the marginal cost of a System 1 decision is zero.
"""
from __future__ import annotations

import logging
import re
import time

from .a2a_client import A2AClient
from .analyzer_client import AnalyzerClient
from .gemma_client import GemmaReviewer, SystemTwoJudgment
from .feedback import FeedbackLogger
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

#: Intent -> worker agent (mirrors data/intents.yaml worker: mapping).
_MOCK_INTENT_WORKER = {
    "billing_inquiry": "billing",
    "technical_support": "support",
    "sales_question": "sales",
    "account_update": "account",
    "order_status": "orders",
    "fallback": "fallback",
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

_MOCK_ORDER_NUMBER_RE = re.compile(r"#\d+|\border\s*#?\s*\d{4,}|\b\d{5,}\b", re.I)
_MOCK_FOLLOWUP_RE = re.compile(
    r"\b(what about|how about|and then|and the|the (second|first|other) one)\b"
    r"|^(yes|no|yeah|nope|okay|ok)[,.]?\s+(that|this|the|it)\b"
    r"|\bit (still|doesn|don't|does not)\b",
    re.I,
)
_MOCK_HOWTO_RE = re.compile(r"\bhow (do|can|to)\b|\bwhere are\b", re.I)


def mock_routing_decisions(text: str, intent: str) -> dict:
    """Deterministic stand-ins for the seven new System 1 decisions."""
    lowered = text.lower()
    worker_agent = _MOCK_INTENT_WORKER.get(intent, "fallback")
    if intent == "billing_inquiry":
        skill = ("refund_process" if "refund" in lowered or "overcharg" in lowered
                 else "billing_lookup")
    elif intent == "technical_support":
        skill = "knowledge_search" if _MOCK_HOWTO_RE.search(lowered) else "none"
    elif intent == "sales_question":
        skill = ("knowledge_search" if re.search(
            r"pricing|plans|features|trial|discount|offer", lowered) else "none")
    elif intent == "account_update":
        skill = ("knowledge_search" if _MOCK_HOWTO_RE.search(lowered)
                 else "account_modify")
    elif intent == "order_status":
        skill = "order_tracking"
    else:
        skill = "escalation"

    needs_rag = 1.0 if skill == "knowledge_search" else 0.0
    if intent == "order_status" and not _MOCK_ORDER_NUMBER_RE.search(text):
        needs_more_input = 0.9
    elif _MOCK_FOLLOWUP_RE.search(text):
        needs_more_input = 0.85
    elif len(text.split()) <= 3 and intent == "fallback":
        needs_more_input = 0.8
    else:
        needs_more_input = 0.1

    if intent in ("account_update", "billing_inquiry", "order_status"):
        needs_user_details = 0.9
    elif intent == "technical_support" and re.search(
            r"log ?in|password|my account", lowered):
        needs_user_details = 0.9
    else:
        needs_user_details = 0.1

    is_multi_turn = 0.9 if _MOCK_FOLLOWUP_RE.search(text) else 0.05
    needs_async = 0.9 if skill == "refund_process" else 0.05

    return {
        "worker_agent": worker_agent,
        "worker_confidence": 0.8,
        "skill_required": skill,
        "skill_confidence": 0.75,
        "needs_rag": needs_rag,
        "needs_more_input": needs_more_input,
        "needs_user_details": needs_user_details,
        "is_multi_turn": is_multi_turn,
        "needs_async": needs_async,
    }

_IMPERATIVE_RE = re.compile(
    r"^(please\s+)?(cancel|update|change|send|track|give|show|get|set|add|"
    r"remove|delete|open|close|start|stop|book|order|tell|refund|pay)\b",
    re.IGNORECASE,
)

#: System 1 needs_more_input above this -> slow path so System 2 can
#: decide whether to ask a clarifying question before dispatching.
MORE_INPUT_REVIEW_THRESHOLD = 0.8


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
            **mock_routing_decisions(text, "fallback"),
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
        **mock_routing_decisions(text, top_intent),
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
    """Orchestrates System 1 (all decisions) -> policy ->
    optional System 2 (tokens only) -> ADK execution."""

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
        feedback_logger: FeedbackLogger | None = None,
        analyzer_client: AnalyzerClient | None = None,
        executor: object | None = None,
    ) -> None:
        self.system_one = system_one
        self.system_two = system_two
        self.worker_agents = worker_agents
        self.confidence_threshold = confidence_threshold
        self.human_review_threshold = human_review_threshold
        self.guardrail_review_score = guardrail_review_score
        self.guardrail_block_score = guardrail_block_score
        self.a2a = a2a_client or A2AClient()
        # Execution plane: ADK runs every skill invocation. System 1
        # decides WHAT; the executor handles HOW.
        if executor is None:
            from .adk_execution import AdkSkillExecutor
            executor = AdkSkillExecutor()
        self.executor = executor
        # Tier-2 unified intent analyzer (per-agent System 1). None or a
        # disabled client = routing works exactly as before.
        self.analyzer = analyzer_client
        # RLCD feedback loop: log every System 2 review for human review
        # and future retraining. None disables it.
        self.feedback = feedback_logger
        # Task framework (2026-10-10): System 1 is invoked ONLY when the
        # decision gate says a genuine decision is needed. Otherwise the
        # active task's locked intent carries the turn with no System 1.
        from .task_manager import TaskManager
        self.tasks = TaskManager()
        # Workflow engine (2026-10-10): agent → workflow → analyzer
        # (workflow level) → decision mechanism → steps. Definitions
        # come from the analyzer service (hot-reloaded, Studio-edited).
        from .workflow_engine import WorkflowEngine
        analyzer_base = getattr(analyzer_client, "base_url", "") or ""
        self.workflows = WorkflowEngine(analyzer_base)
        # worker_agent ("billing", "support", ...) -> worker URL, derived
        # from the intent taxonomy's worker: mapping and the intent->URL map.
        self._worker_agent_urls: dict[str, str] = {}
        for intent in system_one.intents:
            url = worker_agents.get(intent.name)
            if url and intent.worker not in self._worker_agent_urls:
                self._worker_agent_urls[intent.worker] = url

    # ------------------------------------------------------------------ API
    def resolve_worker(self, intent: str) -> str:
        if intent in self.worker_agents:
            return self.worker_agents[intent]
        if "fallback" in self.worker_agents:
            return self.worker_agents["fallback"]
        return next(iter(self.worker_agents.values()))

    def resolve_worker_for(self, s1: SystemOneDecision) -> str:
        """Resolve the worker URL, preferring System 1's worker_agent call."""
        url = self._worker_agent_urls.get(s1.worker_agent)
        if url:
            return url
        return self.resolve_worker(s1.intent)

    def handle_query(self, text: str, session_id: str | None = None) -> dict:
        started = time.monotonic()
        text = (text or "").strip()
        session_id = session_id or f"ses-{uuid.uuid4().hex[:8]}"

        # --- Decision gate (deterministic; no System 1 here) ---
        # Layer 1: active workflow? Continue its current step with
        # stored slots, or abort/switch.
        wf_state = self.workflows.get_state(session_id)
        if wf_state is not None:
            need_wf, wf_reason = self.workflows.needs_workflow_decision(
                text, wf_state)
            if not need_wf:
                if wf_reason == "workflow_abort":
                    return self._abort_workflow(text, wf_state, session_id,
                                               started=started)
                return self._continue_workflow(text, wf_state, wf_reason,
                                               session_id, started=started)
            # Switch/abort fall through to a fresh decision below.
            self.workflows.abort(session_id)

        # Layer 2: active single task? Locked intent carries the turn.
        task = self.tasks.get_task(session_id)
        need, reason = self.tasks.needs_decision(text, task)
        if not need:
            return self._continue_task(text, task, reason, session_id,
                                       started=started)

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

        # --- System 1, decision 2: workflow or single task? ---
        # The analyzer service identifies the workflow (trigger match
        # until the Laya workflow head is trained). A workflow means:
        # analyze at workflow level, then run its steps deterministically.
        wf_def = None
        if self.analyzer is not None and self.analyzer.enabled:
            wf_def = self.analyzer.identify_workflow(text, s1.worker_agent)
        if wf_def is not None:
            return self._start_workflow(text, s1, wf_def, session_id,
                                        reason, started=started)

        fast_path = (
            s1.confidence >= self.confidence_threshold
            and s1.needs_human <= self.human_review_threshold
            and s1.guardrail_score < self.guardrail_review_score
            and s1.needs_more_input < MORE_INPUT_REVIEW_THRESHOLD
        )
        s2: SystemTwoJudgment | None = None
        if not fast_path:
            s2 = self.system_two.review(text, s1)
            # --- RLCD feedback: log System 2's judgment for human review.
            # Usable as a training label only when Gemma actually decided
            # (not escalated) — the human review gate filters the rest. ---
            if self.feedback is not None and s2 is not None:
                self.feedback.log(
                    query=text,
                    s1={
                        "intent": s1.intent, "confidence": s1.confidence,
                        "worker_agent": s1.worker_agent,
                        "worker_confidence": s1.worker_confidence,
                        "skill_required": s1.skill_required,
                        "skill_confidence": s1.skill_confidence,
                        "needs_human": s1.needs_human,
                        "utterance_type": s1.utterance_type,
                        "guardrail_score": s1.guardrail_score,
                        "needs_rag": s1.needs_rag,
                        "needs_more_input": s1.needs_more_input,
                        "needs_user_details": s1.needs_user_details,
                        "is_multi_turn": s1.is_multi_turn,
                        "needs_async": s1.needs_async,
                    },
                    s2={
                        "intent": s2.intent, "confidence": s2.confidence,
                        "escalate_to_human": s2.escalate_to_human,
                        "worker_agent": s2.worker_agent,
                        "skill_required": s2.skill_required,
                        "needs_rag": s2.needs_rag,
                        "needs_more_input": s2.needs_more_input,
                        "needs_user_details": s2.needs_user_details,
                        "is_multi_turn": s2.is_multi_turn,
                        "needs_async": s2.needs_async,
                        "rationale": s2.rationale,
                    },
                    usable_label=not s2.escalate_to_human,
                )

        resp_path = "fast" if fast_path else "system2"
        resp = self._respond(text, s1, s2, path=resp_path,
                             started=started)
        resp["session_id"] = session_id
        resp["decision_made"] = True
        resp["decision_reason"] = reason
        # A fresh decision starts a task (unless it was blocked or fell
        # back — those carry no actionable intent to lock).
        if resp_path not in ("guardrail_block", "empty") and not resp.get(
                "routed_to_fallback"):
            action = (resp.get("analyzer") or {}).get("order_action")
            new_task = self.tasks.start_task(
                session_id, resp["intent"], resp["worker"],
                action=action, slots=self._extract_slots(text))
            self._update_task_from_result(new_task, resp)
            resp["task"] = new_task.to_dict()
        else:
            resp["task"] = None
        return resp

    # ------------------------------------------------------ task framework
    # Required slots per worker action. The worker stays the source of
    # truth for what it needs; this is the router's local copy so it can
    # track pending slots without another round-trip.
    ACTION_REQUIRED_SLOTS = {
        "track": ["order_id"], "cancel": ["order_id"],
        "modify": ["order_id"], "return": ["order_id"],
        "reorder": ["order_id"], "estimate": ["order_id"],
        "place": ["items", "address"], "list": [], "order_info": [],
    }

    @staticmethod
    def _extract_slots(text: str) -> dict:
        from .task_manager import SLOT_PATTERNS
        slots: dict = {}
        for name, pattern in SLOT_PATTERNS.items():
            m = pattern.search(text or "")
            if m:
                slots[name] = m.group(1)
        return slots

    def _fill_slots(self, text: str, task) -> None:
        for name, value in self._extract_slots(text).items():
            task.slots[name] = value

    def _update_task_from_result(self, task, resp: dict) -> None:
        status = (resp.get("dispatch") or {}).get("result", {}).get("status")
        if status == "completed":
            self.tasks.complete_task(task.session_id)
        elif status == "needs_input":
            required = self.ACTION_REQUIRED_SLOTS.get(task.action or "", [])
            task.pending_slots = [s for s in required
                                  if s not in task.slots]
            # stays active — the next turn continues it without System 1

    def _continue_task(self, text: str, task, reason: str,
                       session_id: str, *, started: float) -> dict:
        """A turn that needs no decision: locked intent, no System 1."""
        if reason == "task_control_abort":
            aborted = self.tasks.abort_task(session_id)
            return {
                "path": "task_continuation",
                "intent": task.intent,
                "system1_intent": None,  # System 1 was NOT consulted
                "session_id": session_id,
                "decision_made": False,
                "decision_reason": reason,
                "task": aborted.to_dict() if aborted else None,
                "answer": "Got it — I've dropped that. "
                          "What would you like to do?",
                "dispatch": None,
                "latency_ms": round((time.monotonic() - started) * 1000.0, 1),
                "ts": time.time(),
            }
        self._fill_slots(text, task)
        # The worker gets the locked action + session slots; no analyzer
        # call — there is nothing new to decide.
        context: dict = {
            "session": {"task_id": task.task_id, "slots": task.slots,
                        "action": task.action},
        }
        if task.action:
            context["analyzer"] = {"order_action": task.action}
        dispatch = self.executor.dispatch(task.worker_agent, text,
                                          task.intent, context=context)
        self._update_task_from_result(task, {"dispatch": dispatch})
        result = dispatch.get("result", {})
        return {
            "path": "task_continuation",
            "intent": task.intent,
            "system1_intent": None,  # System 1 was NOT consulted
            "session_id": session_id,
            "decision_made": False,
            "decision_reason": reason,
            "task": task.to_dict(),
            "worker": task.worker_agent,
            "dispatch": dispatch,
            "answer": result.get("message"),
            "latency_ms": round((time.monotonic() - started) * 1000.0, 1),
            "ts": time.time(),
        }

    # -------------------------------------------------- workflow level
    def _start_workflow(self, text: str, s1, wf_def: dict,
                        session_id: str, reason: str,
                        *, started: float) -> dict:
        """System 1 identified agent + workflow. Now: analyze at
        workflow level, then run the first step as a locked task."""
        # Analyzer at workflow level: /decide with the workflow set.
        wf_decision = None
        if self.analyzer is not None and self.analyzer.enabled:
            wf_decision = self.analyzer.decide_at_workflow(
                text, s1.worker_agent, wf_def["name"],
                router_context={"intent": s1.intent,
                                "confidence": round(s1.confidence, 3)})
        # Start the workflow session; store any details from this message.
        st = self.workflows.start(session_id, wf_def)
        self.workflows.store_slots(session_id, text)
        return self._run_workflow_step(text, st, s1, session_id, reason,
                                       wf_decision, started=started)

    def _run_workflow_step(self, text: str, st, s1, session_id: str,
                           reason: str, wf_decision: dict | None,
                           *, started: float) -> dict:
        step = self.workflows.current_step(st)
        missing = self.workflows.missing_slots(st)
        worker_url = self.resolve_worker_for(s1)
        if missing:
            # Step can't run yet — ask for the missing detail and wait.
            # Still no System 1: the workflow owns this turn.
            return {
                "path": "workflow",
                "intent": s1.intent,
                "system1_intent": s1.intent,
                "system1_workflow": st.workflow_name,
                "session_id": session_id,
                "decision_made": True,
                "decision_reason": reason,
                "workflow": st.to_dict(),
                "task": None,
                "worker": worker_url,
                "workflow_decision": (wf_decision or {}).get("decision"),
                "answer": (f"To continue '{st.workflow_name}' I need: "
                           f"{', '.join(missing)}."),
                "needs_slots": missing,
                "dispatch": None,
                "latency_ms": round((time.monotonic() - started) * 1000.0, 1),
                "ts": time.time(),
            }
        # All slots filled — execute the step as a locked task.
        self.workflows.begin_step(session_id)  # started → inprogress
        context = {
            "analyzer": {"order_action": step["action"]},
            "session": {"task_id": f"wf-{st.workflow_name}-{step['id']}",
                        "slots": st.slots, "action": step["action"],
                        "workflow": st.workflow_name,
                        "step": step["id"]},
        }
        dispatch = self.executor.dispatch(worker_url, text, s1.intent,
                                          context=context)
        result = dispatch.get("result", {})
        status = result.get("status")
        # Merge worker-returned slots (e.g. items the worker parsed).
        worker_slots = (result.get("data") or {}).get("slots")
        if worker_slots:
            self.workflows.store_slots(session_id, "", extra=worker_slots)
        if status == "completed":
            transition = self.workflows.advance(session_id, "success")
        elif status == "needs_input":
            # Step is waiting on the user — workflow stays active.
            transition = f"next:{step['id']}"
        else:
            transition = self.workflows.advance(session_id, "failure")
        fresh = self.workflows.get_state(session_id)
        return {
            "path": "workflow",
            "intent": s1.intent,
            "system1_intent": s1.intent,
            "system1_workflow": st.workflow_name,
            "session_id": session_id,
            "decision_made": True,
            "decision_reason": reason,
            "workflow": fresh.to_dict() if fresh else st.to_dict(),
            "workflow_step": step["id"],
            "workflow_transition": transition,
            "task": None,
            "worker": worker_url,
            "workflow_decision": (wf_decision or {}).get("decision"),
            "dispatch": dispatch,
            "answer": result.get("message"),
            "latency_ms": round((time.monotonic() - started) * 1000.0, 1),
            "ts": time.time(),
        }

    def _continue_workflow(self, text: str, st, reason: str,
                           session_id: str, *, started: float) -> dict:
        """A turn inside an active workflow: store details, run/await
        the current step. System 1 is NOT consulted."""
        # Store whatever details the user just gave.
        self.workflows.store_slots(session_id, text)
        # Re-resolve the worker from the stored definition's agent.
        worker_url = self._worker_agent_urls.get(
            st.definition.get("agent"), next(iter(self.worker_agents.values())))
        # Build a minimal s1-like namespace for _run_workflow_step.
        s1 = type("S1", (), {"intent": "order_status",
                             "confidence": 1.0,
                             "worker_agent": st.definition.get("agent")})()
        # If the message looks like a new workflow trigger for a
        # DIFFERENT workflow → switch: abort and re-identify.
        new_wf = None
        if self.analyzer is not None and self.analyzer.enabled:
            new_wf = self.analyzer.identify_workflow(
                text, st.definition.get("agent", ""))
        if new_wf and new_wf["name"] != st.workflow_name:
            self.workflows.abort(session_id)
            return self._start_workflow(text, s1, new_wf, session_id,
                                        "workflow_switch", started=started)
        return self._run_workflow_step(text, st, s1, session_id, reason,
                                       None, started=started)

    def _abort_workflow(self, text: str, st, session_id: str,
                        *, started: float) -> dict:
        """User aborted the workflow → clear it. The next message
        starts identifying a new workflow from scratch."""
        self.workflows.abort(session_id)
        return {
            "path": "workflow_abort",
            "intent": None,
            "system1_intent": None,  # System 1 was NOT consulted
            "session_id": session_id,
            "decision_made": False,
            "decision_reason": "workflow_abort",
            "workflow": {"workflow": st.workflow_name, "state": "aborted"},
            "answer": (f"OK, I've cancelled the '{st.workflow_name}' "
                       f"workflow. What would you like to do?"),
            "dispatch": None,
            "latency_ms": round((time.monotonic() - started) * 1000.0, 1),
            "ts": time.time(),
        }

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
            "needs_more_input": s1.needs_more_input >= MORE_INPUT_REVIEW_THRESHOLD,
        }

        if path == "guardrail_block":
            intent, routed = "fallback", True
            rationale = (
                f"Guardrail block: risk score {s1.guardrail_score:.2f} "
                f"(band '{s1.guardrail_band}') >= block threshold "
                f"{self.guardrail_block_score}."
            )
        elif path == "system2" and s2 is not None:
            # ARCHITECTURE: every decision is System 1's. System 2
            # generates the rationale (tokens) but never overrides a
            # routing decision. Escalation comes from System 1's
            # needs_human gate, not from System 2.
            if s1.needs_human > self.human_review_threshold:
                intent, routed = "fallback", True
            else:
                intent, routed = s1.intent, False
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
        # Routed-to-fallback always goes to the fallback worker; otherwise
        # prefer System 1's worker_agent decision over the intent mapping.
        worker_url = (self.resolve_worker("fallback") if routed
                      else self.resolve_worker_for(s1))
        # --- Tier-2: unified intent analyzer (per-agent System 1) ---
        # Once System 1 has picked a worker_agent, ask the analyzer for
        # that agent's domain-specific decisions and hand them to the
        # worker with the dispatch. Skipped for fallback-routed queries;
        # never blocks routing when the analyzer is down.
        analyzer_result: dict | None = None
        if not routed and self.analyzer is not None:
            analyzer_result = self.analyzer.analyze(
                text, s1.worker_agent,
                router_context={
                    "intent": s1.intent,
                    "confidence": round(s1.confidence, 3),
                    "skill_required": s1.skill_required,
                })
        # --- Execution plane (ADK): System 1 decided WHAT (intent,
        # worker, skill); ADK executes HOW. The analyzer's Tier-2
        # decisions ride along in the dispatch context.
        dispatch = self.executor.dispatch(
            worker_url, text, final_intent,
            context={"analyzer": analyzer_result} if analyzer_result else None)

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
            # --- System 1 routing decisions (2026-10-08) ---
            "worker_agent": s1.worker_agent,
            "worker_confidence": round(s1.worker_confidence, 3),
            "skill_required": s1.skill_required,
            "skill_confidence": round(s1.skill_confidence, 3),
            "needs_rag": round(s1.needs_rag, 3),
            "needs_more_input": round(s1.needs_more_input, 3),
            "needs_user_details": round(s1.needs_user_details, 3),
            "is_multi_turn": round(s1.is_multi_turn, 3),
            "needs_async": round(s1.needs_async, 3),
            "system2_used": s2 is not None,
            "system2_intent": s2.intent if s2 else None,
            "system2_escalated": s2.escalate_to_human if s2 else False,
            "system2_rationale": rationale,
            "routed_to_fallback": routed,
            "worker": worker_url,
            "worker_name": worker_name_for(worker_url),
            # --- Tier-2 analyzer decisions for the chosen worker_agent ---
            "analyzer": analyzer_result,
            "task_id": dispatch["task_id"],
            "dispatch": dispatch,
            "latency_ms": round((time.monotonic() - started) * 1000.0, 1),
            "ts": time.time(),
        }
