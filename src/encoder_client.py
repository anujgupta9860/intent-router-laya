"""System 1 backend for a self-trained encoder checkpoint.

Loads a checkpoint exported by train/finetune.py (ModernBERT +
classification head + calibration.json with the fitted temperature) and
serves it behind the same SystemOneDecision contract as Laya.

Honest scope: the encoder predicts the *intent* (with temperature-scaled,
calibrated confidence). The other signals are derived, not learned:
  * needs_human  <- 1 - confidence (uncertain => human should look)
  * utterance_type <- lightweight heuristic (same as the mock)
  * guardrail_score <- keyword heuristic (same as the mock)
  * worker_agent / skill_required / needs_rag / needs_more_input /
    needs_user_details / is_multi_turn / needs_async <- keyword
    heuristics (same as the mock, see hybrid.mock_routing_decisions)

If you need learned guardrails, fine-tune Laya itself with its RLCD flow
(see train/README.md) - its Score head is trained, not heuristic.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from .hybrid import mock_guardrail_score, mock_routing_decisions, mock_utterance_type
from .intents import Intent
from .laya_client import GUARDRAIL_RUBRIC, SystemOneDecision

log = logging.getLogger(__name__)


class EncoderError(RuntimeError):
    """Raised when the encoder checkpoint can't be loaded or run."""


class EncoderClient:
    """Self-trained intent classifier served as System 1."""

    def __init__(self, checkpoint: str) -> None:
        self.checkpoint = checkpoint
        self._tokenizer = None
        self._model = None
        self._temperature = 1.0
        self._labels: list[str] = []

    # ------------------------------------------------------------------ API
    def decide(self, text: str, intents: list[Intent]) -> SystemOneDecision:
        self._ensure_loaded()
        import torch

        started = time.monotonic()
        inputs = self._tokenizer(
            text, return_tensors="pt", truncation=True, max_length=128
        )
        with torch.no_grad():
            logits = self._model(**inputs).logits / self._temperature
            probs_t = torch.softmax(logits, dim=-1)[0]
        latency_ms = (time.monotonic() - started) * 1000.0

        allowed = {i.name for i in intents}
        probabilities = {
            label: round(float(probs_t[i]), 4)
            for i, label in enumerate(self._labels)
            if label in allowed
        }
        top_idx = int(probs_t.argmax())
        intent = self._labels[top_idx] if self._labels[top_idx] in allowed else "fallback"
        confidence = round(float(probs_t[top_idx]), 3)
        # Derived signals (see module docstring for the honesty note).
        needs_human = round(max(0.0, min(1.0, 1.0 - confidence + 0.15)), 3)
        if intent == "fallback":
            needs_human = max(needs_human, 0.85)

        return SystemOneDecision(
            intent=intent,
            probabilities=probabilities,
            confidence=confidence,
            needs_human=needs_human,
            utterance_type=mock_utterance_type(text),
            utterance_confidence=0.6,
            guardrail_score=mock_guardrail_score(text),
            model=f"encoder:{self.checkpoint}",
            latency_ms=round(latency_ms, 1),
            **mock_routing_decisions(text, intent),
        )

    # -------------------------------------------------------------- internals
    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )
        except ImportError as exc:
            raise EncoderError(
                "transformers is required for the encoder backend; "
                "pip install transformers torch"
            ) from exc
        ckpt = Path(self.checkpoint)
        if not ckpt.exists():
            raise EncoderError(
                f"encoder checkpoint not found: {ckpt} "
                "(train one with train/finetune.py)"
            )
        log.info("loading encoder checkpoint %s ...", ckpt)
        self._tokenizer = AutoTokenizer.from_pretrained(ckpt)
        self._model = AutoModelForSequenceClassification.from_pretrained(ckpt)
        self._model.eval()
        calib_path = ckpt / "calibration.json"
        if calib_path.exists():
            calib = json.loads(calib_path.read_text())
            self._temperature = float(calib.get("temperature", 1.0))
            self._labels = list(calib.get("labels", []))
        if not self._labels:
            # fall back to the model's own label map
            id2label = getattr(self._model.config, "id2label", {})
            self._labels = [id2label[i] for i in sorted(id2label)]
        log.info(
            "encoder loaded (%d labels, temperature %.3f)",
            len(self._labels), self._temperature,
        )
