"""Client for the unified intent analyzer (Tier 2, per-agent System 1).

After the router's System 1 picks a worker_agent, this service asks the
unified analyzer (intent-analyzer-unified) for that agent's domain-specific
typed decisions — e.g. billing's billing_action + noul flags — and hands
them to the worker with the A2A dispatch.

The analyzer is best-effort: when ANALYZER_URL is unset or the service is
unreachable, analyze() returns None and routing proceeds unchanged. A
down analyzer must never break routing.
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)


class AnalyzerClient:
    """HTTP client for the unified intent analyzer service."""

    def __init__(self, base_url: str = "", timeout_s: float = 10.0) -> None:
        self.base_url = (base_url or "").rstrip("/")
        self.timeout_s = timeout_s

    @property
    def enabled(self) -> bool:
        return bool(self.base_url)

    def analyze(self, query: str, worker_agent: str,
                router_context: dict | None = None) -> dict | None:
        """Ask the analyzer for the worker agent's typed decisions.

        Returns the analyzer's decision dict, or None when disabled or
        on any failure (never raises).
        """
        if not self.enabled:
            return None
        try:
            import httpx
        except ImportError:
            log.warning("httpx not installed; skipping analyzer call")
            return None
        try:
            resp = httpx.post(
                f"{self.base_url}/analyze",
                json={
                    "query": query,
                    "worker_agent": worker_agent,
                    "router_context": router_context or {},
                },
                timeout=self.timeout_s,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            log.warning("unified analyzer call failed (%s); continuing "
                        "without analyzer decisions", exc)
            return None

    def decide_at_workflow(self, query: str, worker_agent: str,
                           workflow: str,
                           router_context: dict | None = None) -> dict | None:
        """Analyzer at workflow level: /decide with the workflow set.

        Returns analyzer output + policy decision + workflow definition.
        Never raises (None on failure).
        """
        if not self.enabled:
            return None
        try:
            import httpx
        except ImportError:
            return None
        try:
            resp = httpx.post(
                f"{self.base_url}/decide",
                json={
                    "query": query,
                    "worker_agent": worker_agent,
                    "router_context": router_context or {},
                    "workflow": workflow,
                },
                timeout=self.timeout_s,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            log.warning("workflow-level decide failed (%s)", exc)
            return None

    def identify_workflow(self, query: str,
                          worker_agent: str) -> dict | None:
        """System 1's workflow decision via the analyzer service."""
        if not self.enabled:
            return None
        try:
            import httpx
            resp = httpx.post(
                f"{self.base_url}/workflows/identify",
                json={"query": query, "worker_agent": worker_agent,
                      "router_context": {}},
                timeout=self.timeout_s,
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("definition")
        except Exception as exc:
            log.warning("workflow identify failed (%s)", exc)
            return None
