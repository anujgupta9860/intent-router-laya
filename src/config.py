"""Runtime configuration for the intent-router-laya service.

Every setting comes from an environment variable so the same container image
can run locally, in docker-compose, and on GKE via Helm with no code changes.

Two model systems, both self-hostable - no commercial API required:
  * System 1 (Laya):  SYSTEM1_BACKEND=mock|laya, LAYA_CHECKPOINT, ...
  * System 2 (Gemma): SYSTEM2_BACKEND=mock|ollama|vertex, GEMMA_MODEL, ...
"""
from __future__ import annotations

import json
import os


def _get(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _float(name: str, default: float) -> float:
    try:
        return float(_get(name, str(default)))
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc


class Settings:
    """All runtime configuration, read from the environment."""

    #: Default A2A worker URLs used when WORKER_AGENTS_JSON is not set.
    #: These match the docker-compose demo (worker-a on :8001, worker-b on :8002).
    DEFAULT_WORKERS: dict[str, str] = {
        "billing_inquiry": "http://worker-a:8001",
        "technical_support": "http://worker-a:8001",
        "sales_question": "http://worker-a:8001",
        "account_update": "http://worker-b:8002",
        "order_status": "http://worker-b:8002",
        "fallback": "http://worker-b:8002",
    }

    def __init__(self) -> None:
        # --- System 1 (Laya, self-hosted) ---
        s1 = _get("SYSTEM1_BACKEND", "mock").strip().lower()
        if s1 not in ("mock", "laya", "encoder"):
            raise ValueError(
                f"SYSTEM1_BACKEND must be one of mock|laya|encoder, got {s1!r}"
            )
        self.system1_backend: str = s1
        self.laya_checkpoint: str = _get(
            "LAYA_CHECKPOINT", "convaiinnovations/laya-typed-decisions"
        )
        self.laya_device: str = _get("LAYA_DEVICE", "auto")
        self.laya_preload: bool = _get("LAYA_PRELOAD", "true").strip().lower() in (
            "1", "true", "yes",
        )

        # --- System 2 (Gemma) ---
        s2 = _get("SYSTEM2_BACKEND", "mock").strip().lower()
        if s2 not in ("mock", "ollama", "vertex"):
            raise ValueError(
                f"SYSTEM2_BACKEND must be one of mock|ollama|vertex, got {s2!r}"
            )
        self.system2_backend: str = s2
        self.gemma_model: str = _get("GEMMA_MODEL", "gemma-3-4b-it")
        self.ollama_url: str = _get("OLLAMA_URL", "http://localhost:11434")
        self.vertex_project: str = _get("VERTEX_PROJECT", "")
        self.vertex_region: str = _get("VERTEX_REGION", "us-central1")
        self.vertex_endpoint_id: str = _get("VERTEX_ENDPOINT_ID", "")

        # --- routing policy ---
        self.confidence_threshold: float = _float("CONFIDENCE_THRESHOLD", 0.6)
        self.human_review_threshold: float = _float("HUMAN_REVIEW_THRESHOLD", 0.5)
        # Guardrail Score bands (0=safe .. 4=critical on the Laya rubric).
        self.guardrail_review_score: float = _float("GUARDRAIL_REVIEW_SCORE", 1.5)
        self.guardrail_block_score: float = _float("GUARDRAIL_BLOCK_SCORE", 3.0)
        if not self.guardrail_block_score > self.guardrail_review_score:
            raise ValueError(
                "GUARDRAIL_BLOCK_SCORE must be greater than GUARDRAIL_REVIEW_SCORE"
            )

        self.worker_agents: dict[str, str] = self._load_worker_agents()
        self.log_level: str = _get("LOG_LEVEL", "INFO").upper()

    @classmethod
    def _load_worker_agents(cls) -> dict[str, str]:
        raw = os.environ.get("WORKER_AGENTS_JSON", "").strip()
        if not raw:
            return dict(cls.DEFAULT_WORKERS)
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"WORKER_AGENTS_JSON is not valid JSON: {exc}") from exc
        if not isinstance(data, dict) or not data:
            raise ValueError(
                "WORKER_AGENTS_JSON must be a non-empty JSON object mapping intent -> agent URL"
            )
        return {str(k): str(v) for k, v in data.items()}

    def worker_url(self, intent: str) -> str:
        """Resolve the A2A agent URL for an intent, falling back gracefully."""
        if intent in self.worker_agents:
            return self.worker_agents[intent]
        if "fallback" in self.worker_agents:
            return self.worker_agents["fallback"]
        return next(iter(self.worker_agents.values()))
