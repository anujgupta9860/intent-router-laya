"""Central skill registry: one place for every agent's skills.

Discovers all configured worker agents, fetches each one's
/.well-known/agent-card.json, and serves a unified skill catalog:

  GET  /skills          - every skill from every reachable agent
  POST /skills/refresh  - force re-discovery now

Entries look like:
  {"skill": "track_order", "agent": "order-management-agent",
   "agent_url": "https://...", "description": "...",
   "protocols": ["a2a", "mcp"]}

The catalog is cached (TTL) and refreshes in the background; an
unreachable agent keeps its last-known skills marked stale rather
than vanishing from the catalog.
"""
from __future__ import annotations

import logging
import threading
import time

log = logging.getLogger(__name__)

CACHE_TTL_S = 300


class SkillRegistry:
    def __init__(self, worker_urls: dict[str, str]) -> None:
        # intent -> url; dedupe to unique agent base URLs
        seen: dict[str, str] = {}
        for url in worker_urls.values():
            base = url.rstrip("/")
            seen.setdefault(base, base)
        self.agent_urls: list[str] = sorted(seen)
        self._lock = threading.Lock()
        self._catalog: list[dict] = []
        self._agents: dict = {}
        self._last_refresh: float | None = None
        self._stale_agents: list[str] = []

    # ------------------------------------------------------------------ API
    def catalog(self) -> dict:
        self._ensure_fresh()
        with self._lock:
            return {
                "skills": list(self._catalog),
                "agents": dict(self._agents),
                "skill_count": len(self._catalog),
                "agent_count": len(self._agents),
                "last_refresh": self._last_refresh,
                "stale_agents": list(self._stale_agents),
            }

    def refresh(self) -> dict:
        self._discover()
        return self.catalog()

    def find_skill(self, name: str) -> dict | None:
        self._ensure_fresh()
        with self._lock:
            for s in self._catalog:
                if s["skill"] == name:
                    return s
        return None

    # -------------------------------------------------------------- internals
    def _ensure_fresh(self) -> None:
        if (self._last_refresh is None
                or time.time() - self._last_refresh > CACHE_TTL_S):
            self._discover()

    def _discover(self) -> None:
        import httpx

        skills: list[dict] = []
        agents: dict = {}
        stale: list[str] = []
        for base in self.agent_urls:
            card = self._fetch_card(base)
            if card is None:
                stale.append(base)
                continue
            name = card.get("name", base)
            agents[base] = {
                "name": name,
                "description": card.get("description", ""),
                "version": card.get("version", ""),
                "skills": [s.get("name") for s in card.get("skills", [])],
                "protocols": card.get("protocols", {}),
                "card_url": base + "/.well-known/agent-card.json",
            }
            protocols = list(card.get("protocols", {}).keys()) or ["a2a"]
            for s in card.get("skills", []):
                skills.append({
                    "skill": s.get("name"),
                    "agent": name,
                    "agent_url": base,
                    "description": s.get("description", ""),
                    "protocols": protocols,
                })
        # keep last-known skills for stale agents instead of dropping them
        with self._lock:
            fresh_urls = set(agents)
            kept = [dict(s, stale=True) for s in self._catalog
                    if s["agent_url"] not in fresh_urls]
            self._catalog = sorted(skills + kept,
                                   key=lambda s: s["skill"])
            self._agents = agents
            self._stale_agents = stale
            self._last_refresh = time.time()
        log.info("skill registry: %d skills from %d agents (%d stale)",
                 len(skills), len(agents), len(stale))

    @staticmethod
    def _fetch_card(base: str) -> dict | None:
        import httpx

        url = base + "/.well-known/agent-card.json"
        try:
            resp = httpx.get(url, timeout=10.0)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            log.warning("skill registry: no card from %s (%s)", url, exc)
            return None
