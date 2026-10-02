"""A2A dispatch client for worker agents.

Uses the official ``a2a-sdk`` when it is installed. When it is not
installed (or the SDK call fails), it falls back to a documented stub that
logs the dispatch and returns a synthetic task id, so the router stays
runnable everywhere while the real worker fleet is being built.
"""
from __future__ import annotations

import logging
import uuid

log = logging.getLogger(__name__)

try:
    import a2a  # noqa: F401  -- official A2A Python SDK
    _A2A_SDK_AVAILABLE = True
except ImportError:
    _A2A_SDK_AVAILABLE = False


class A2AClient:
    def __init__(self) -> None:
        self.sdk_available = _A2A_SDK_AVAILABLE
        if self.sdk_available:
            log.info("a2a-sdk is installed; real A2A dispatch enabled")
        else:
            log.info("a2a-sdk not installed; using stub dispatch (set pip install a2a-sdk)")

    def _stub_task(self, agent_url: str, text: str, intent: str, reason: str) -> dict:
        task_id = f"task-{uuid.uuid4().hex[:12]}"
        log.info(
            "stub A2A dispatch: task=%s intent=%s agent=%s reason=%s",
            task_id, intent, agent_url, reason,
        )
        return {
            "task_id": task_id,
            "status": "submitted",
            "agent_url": agent_url,
            "intent": intent,
            "mode": "stub",
            "reason": reason,
        }

    def _dispatch_via_sdk(self, agent_url: str, text: str, intent: str) -> dict:
        # Imported lazily so a missing/broken SDK never breaks module import.
        from a2a.client import A2AClient as _SDKClient  # type: ignore

        client = _SDKClient(url=agent_url)
        send = getattr(client, "send_message", None)
        if send is None:
            raise RuntimeError("installed a2a-sdk client exposes no send_message(); API drift")
        # The SDK accepts an A2A Message; keep the payload minimal and let the
        # worker's agent card describe its skills.
        result = send({"text": text, "intent": intent})
        task_id = getattr(result, "task_id", None) or f"task-{uuid.uuid4().hex[:12]}"
        return {
            "task_id": task_id,
            "status": getattr(result, "status", "submitted"),
            "agent_url": agent_url,
            "intent": intent,
            "mode": "a2a-sdk",
        }

    def dispatch(self, agent_url: str, text: str, intent: str) -> dict:
        """Dispatch a classified query to a worker agent over A2A.

        Returns a dict with at least ``task_id``, ``status``, ``agent_url``,
        ``intent`` and ``mode`` (``"a2a-sdk"`` or ``"stub"``).
        """
        if self.sdk_available:
            try:
                return self._dispatch_via_sdk(agent_url, text, intent)
            except Exception as exc:  # SDK present but unusable -> stub, loudly
                log.warning("a2a-sdk dispatch failed (%s); using stub path", exc)
                return self._stub_task(agent_url, text, intent, reason=f"sdk error: {exc}")
        return self._stub_task(agent_url, text, intent, reason="a2a-sdk not installed")

    def agent_card(self, agent_url: str) -> dict:
        """Fetch a worker's A2A agent card (stub when SDK is missing)."""
        import httpx

        url = agent_url.rstrip("/") + "/.well-known/agent-card.json"
        try:
            resp = httpx.get(url, timeout=10.0)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            log.warning("could not fetch agent card from %s: %s", url, exc)
            return {"url": agent_url, "error": str(exc)}
