"""System 1: self-hosted Laya open-weights decision model.

Laya (Convai Innovations, Apache-2.0) is an open-weights "System One"
decision model: like Jev it answers typed questions (choice / noul /
score) in a single forward pass with no text generation - but the
weights are yours. No API key, no per-call billing, no data leaving
your network. ~25-45 ms per decision on local hardware.

One Laya ``predict`` call carries the query plus eleven questions:

  * ``intent``           - Choice: which declared intent matches best.
  * ``human_review``     - Noul: probability a human should review.
  * ``utterance_type``   - Choice: question / command / statement / other.
  * ``guardrail_risk``   - Score: expected risk over the ordered rubric
                           ["safe", "low", "medium", "high", "critical"].
  * ``worker_agent``     - Choice: which worker agent handles this
                           (billing / support / sales / account / orders /
                           fallback).
  * ``skill_required``   - Choice: what tool/skill the worker needs
                           (none / billing_lookup / refund_process /
                           order_tracking / account_modify /
                           knowledge_search / escalation).
  * ``needs_rag``        - Noul: answer needs the knowledge base?
  * ``needs_more_input`` - Noul: query missing details needed to act?
  * ``needs_user_details`` - Noul: handling needs the user's account data?
  * ``is_multi_turn``    - Noul: follow-up in an ongoing conversation?
  * ``needs_async``      - Noul: needs long-running background work?

Question phrasing is the single source of truth in
``src/system1_questions.py`` — training (train/convert_dataset.py) and
serving both import it verbatim. Prompt mismatch between training and
serving silently degrades confidence, so never rephrase either side.

Checkpoints (``convaiinnovations/`` on Hugging Face):
  * ``laya-typed-decisions`` - fine-tuned for typed decisions (default;
    vendor-reported 0.766 accuracy on their 2,000-decision benchmark).
  * ``laya``                - English base (ModernBERT-large, 421M).
  * ``laya-multilingual``   - 100+ languages (mmBERT-base, 322M).

The same four-question contract as the Jev hybrid, so the routing
policy in ``hybrid.py`` is unchanged - only the engine behind System 1
is different.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from .intents import Intent
from .system1_questions import (
    INTENT_DESCRIPTIONS,
    NOUL_QUESTIONS,
    QUESTIONS,
    SKILL_DESCRIPTIONS,
    UTTERANCE_DESCRIPTIONS,
    WORKER_DESCRIPTIONS,
)

log = logging.getLogger(__name__)

#: Ordered guardrail rubric. The Score answer is the expected position on
#: this scale: 0.0 = safe ... 4.0 = critical.
GUARDRAIL_RUBRIC = ["safe", "low", "medium", "high", "critical"]

_UTTERANCE_TYPES = {"question", "command", "statement", "other"}

DEFAULT_CHECKPOINT = "convaiinnovations/laya-typed-decisions"

#: Valid worker-agent and skill option sets (from the shared spec).
_WORKER_AGENTS = set(WORKER_DESCRIPTIONS)
_SKILLS = set(SKILL_DESCRIPTIONS)


@dataclass
class SystemOneDecision:
    """Everything System 1 decided about one query."""
    intent: str
    probabilities: dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0
    needs_human: float = 0.0
    utterance_type: str = "other"
    utterance_confidence: float = 0.0
    guardrail_score: float = 0.0
    model: str = ""
    latency_ms: float = 0.0
    # --- routing decisions (added 2026-10-08) ---
    worker_agent: str = "fallback"
    worker_confidence: float = 0.0
    skill_required: str = "none"
    skill_confidence: float = 0.0
    needs_rag: float = 0.0
    needs_more_input: float = 0.0
    needs_user_details: float = 0.0
    is_multi_turn: float = 0.0
    needs_async: float = 0.0

    @property
    def guardrail_band(self) -> str:
        """Nearest rubric label for the expected score."""
        idx = int(round(self.guardrail_score))
        idx = max(0, min(len(GUARDRAIL_RUBRIC) - 1, idx))
        return GUARDRAIL_RUBRIC[idx]


def build_questions(intents: list[Intent]) -> dict:
    """Build the eleven typed questions from the intent taxonomy.

    Uses the EXACT phrasing the fine-tuned checkpoint was trained on —
    imported verbatim from ``src/system1_questions.py`` (the single
    source of truth shared with train/convert_dataset.py). Prompt
    mismatch between training and serving silently degrades confidence,
    so keep these in sync: never rephrase here.
    """
    # Only include intents present in the taxonomy, in taxonomy order.
    intent_criteria = {
        i.name: INTENT_DESCRIPTIONS.get(i.name, i.description)
        for i in intents
        if i.name in INTENT_DESCRIPTIONS
    }
    questions = {}
    for qid, q in QUESTIONS.items():
        entry: dict = {"type": q["type"], "instructions": q["instructions"]}
        if qid == "intent":
            entry["criteria"] = intent_criteria
        elif "criteria" in q:
            entry["criteria"] = q["criteria"]
        questions[qid] = entry
    return questions


class LayaError(RuntimeError):
    """Raised when the local Laya engine fails."""


class LayaClient:
    """System 1 backed by a self-hosted Laya checkpoint.

    The model loads once (lazily on first ``decide`` unless
    ``preload=True``) and stays resident for ~25-45 ms decisions.
    """

    def __init__(
        self,
        checkpoint: str = DEFAULT_CHECKPOINT,
        *,
        device: str = "auto",
        preload: bool = False,
    ) -> None:
        self.checkpoint = checkpoint
        self.device = device
        self._agent = None
        if preload:
            self._ensure_loaded()

    # ------------------------------------------------------------------ API
    def decide(self, text: str, intents: list[Intent]) -> SystemOneDecision:
        agent = self._ensure_loaded()
        started = time.monotonic()
        try:
            result = agent.predict(text, build_questions(intents))
        except Exception as exc:
            raise LayaError(f"Laya predict failed: {exc}") from exc
        latency_ms = (time.monotonic() - started) * 1000.0
        return parse_decision(
            result, intents, model=self.checkpoint, latency_ms=latency_ms
        )

    # -------------------------------------------------------------- internals
    def _ensure_loaded(self):
        if self._agent is None:
            try:
                import laya
            except ImportError as exc:
                raise LayaError(
                    "the 'laya' package is required for the laya backend; "
                    "pip install laya  (plus a torch build for your platform)"
                ) from exc
            log.info("loading Laya checkpoint %s ...", self.checkpoint)
            started = time.monotonic()
            try:
                self._agent = laya.load(self.checkpoint)
            except Exception as exc:
                raise LayaError(
                    f"could not load Laya checkpoint {self.checkpoint!r}: {exc}"
                ) from exc
            log.info(
                "Laya checkpoint loaded in %.1fs",
                time.monotonic() - started,
            )
            # Warm-up so the first real decision isn't the slow one.
            try:
                self._agent.warmup()
            except Exception:
                pass
        return self._agent


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def parse_decision(
    result: dict,
    intents: list[Intent],
    *,
    model: str = "",
    latency_ms: float = 0.0,
) -> SystemOneDecision:
    """Parse a Laya ``predict`` result defensively.

    Unknown intent names fall back to ``"fallback"`` when declared; the
    guardrail score is clamped to the rubric range [0, 4].
    """
    allowed = {i.name for i in intents}
    answers = (result or {}).get("answers", {})

    # --- intent (Choice) ---
    intent_answer = answers.get("intent", {})
    choice = str(intent_answer.get("choice", "fallback"))
    if choice not in allowed:
        log.warning("Laya returned unknown intent %r; using 'fallback'", choice)
        choice = "fallback" if "fallback" in allowed else choice
    try:
        confidence = _clamp01(float(intent_answer.get("confidence", 0.0)))
    except (TypeError, ValueError):
        confidence = 0.0
    probabilities: dict[str, float] = {}
    raw_probs = intent_answer.get("probabilities", {})
    if isinstance(raw_probs, dict):
        for k, v in raw_probs.items():
            if k in allowed:
                try:
                    probabilities[str(k)] = _clamp01(float(v))
                except (TypeError, ValueError):
                    continue

    # --- needs_human (Noul) ---
    human_answer = answers.get("needs_human", {})
    try:
        needs_human = _clamp01(float(human_answer.get("noul", 0.0)))
    except (TypeError, ValueError):
        needs_human = 0.0

    # --- utterance_type (Choice) ---
    utter_answer = answers.get("utterance_type", {})
    utterance_type = str(utter_answer.get("choice", "other"))
    if utterance_type not in _UTTERANCE_TYPES:
        log.warning(
            "Laya returned unknown utterance type %r; using 'other'",
            utterance_type,
        )
        utterance_type = "other"
    try:
        utterance_confidence = _clamp01(float(utter_answer.get("confidence", 0.0)))
    except (TypeError, ValueError):
        utterance_confidence = 0.0

    # --- guardrail (Score) ---
    guard_answer = answers.get("guardrail", {})
    raw_score = guard_answer.get("score", guard_answer.get("value", 0.0))
    try:
        guardrail_score = float(raw_score)
    except (TypeError, ValueError):
        guardrail_score = 0.0
    guardrail_score = max(0.0, min(float(len(GUARDRAIL_RUBRIC) - 1), guardrail_score))

    # --- worker_agent (Choice) ---
    worker_agent, worker_confidence = _parse_choice(
        answers.get("worker_agent", {}), _WORKER_AGENTS, "fallback")

    # --- skill_required (Choice) ---
    skill_required, skill_confidence = _parse_choice(
        answers.get("skill_required", {}), _SKILLS, "none")

    # --- noul routing decisions ---
    nouls = {}
    for qid in ("needs_rag", "needs_more_input", "needs_user_details",
                "is_multi_turn", "needs_async"):
        ans = answers.get(qid, {})
        try:
            nouls[qid] = _clamp01(float(ans.get("noul", 0.0)))
        except (TypeError, ValueError):
            nouls[qid] = 0.0

    return SystemOneDecision(
        intent=choice,
        probabilities=probabilities,
        confidence=confidence,
        needs_human=needs_human,
        utterance_type=utterance_type,
        utterance_confidence=utterance_confidence,
        guardrail_score=round(guardrail_score, 3),
        model=model,
        latency_ms=latency_ms,
        worker_agent=worker_agent,
        worker_confidence=worker_confidence,
        skill_required=skill_required,
        skill_confidence=skill_confidence,
        needs_rag=nouls["needs_rag"],
        needs_more_input=nouls["needs_more_input"],
        needs_user_details=nouls["needs_user_details"],
        is_multi_turn=nouls["is_multi_turn"],
        needs_async=nouls["needs_async"],
    )


def _parse_choice(answer: dict, allowed: set[str], default: str,
                  ) -> tuple[str, float]:
    """Parse a Choice answer defensively: (choice, confidence)."""
    choice = str(answer.get("choice", default))
    if choice not in allowed:
        log.warning("Laya returned unknown choice %r; using %r",
                    choice, default)
        choice = default
    try:
        confidence = _clamp01(float(answer.get("confidence", 0.0)))
    except (TypeError, ValueError):
        confidence = 0.0
    return choice, confidence
