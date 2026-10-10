"""Tests for the central skill registry."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.skill_registry import SkillRegistry

CARD = {
    "name": "order-management-agent",
    "description": "orders",
    "version": "0.1.0",
    "skills": [
        {"name": "track_order", "description": "track"},
        {"name": "list_orders", "description": "list"},
    ],
    "protocols": {"a2a": {}, "mcp": {}},
}


class _Resp:
    def __init__(self, payload=None):
        self._payload = payload

    def raise_for_status(self):
        if self._payload is None:
            raise RuntimeError("unreachable")

    def json(self):
        return self._payload


def _fake_get(url, timeout=10.0):
    if "good" in url:
        return _Resp(CARD)
    return _Resp(None)


def test_catalog_extracts_all_skills():
    reg = SkillRegistry({"order_status": "https://good.example.com",
                         "fallback": "https://good.example.com"})
    with patch("httpx.get", side_effect=_fake_get):
        cat = reg.refresh()
    assert cat["skill_count"] == 2
    assert cat["agent_count"] == 1  # deduped by URL
    names = [s["skill"] for s in cat["skills"]]
    assert names == ["list_orders", "track_order"]  # sorted
    assert cat["skills"][0]["agent"] == "order-management-agent"
    assert cat["skills"][0]["protocols"] == ["a2a", "mcp"]


def test_unreachable_agent_marked_stale():
    reg = SkillRegistry({"order_status": "https://good.example.com"})
    with patch("httpx.get", side_effect=_fake_get):
        reg.refresh()
    assert reg.catalog()["stale_agents"] == []
    reg2 = SkillRegistry({"order_status": "https://down.example.com"})
    # seed last-known skills, then fail to reach
    reg2._catalog = reg.catalog()["skills"]
    reg2._agents = reg.catalog()["agents"]
    with patch("httpx.get", side_effect=_fake_get):
        cat = reg2.refresh()
    assert cat["stale_agents"] == ["https://down.example.com"]
    # last-known skills are kept, marked stale
    assert cat["skill_count"] == 2
    assert all(s.get("stale") for s in cat["skills"])


def test_find_skill():
    reg = SkillRegistry({"order_status": "https://good.example.com"})
    with patch("httpx.get", side_effect=_fake_get):
        reg.refresh()
    found = reg.find_skill("track_order")
    assert found["agent_url"] == "https://good.example.com"
    assert reg.find_skill("nope") is None


def test_cache_ttl_avoids_refetch():
    reg = SkillRegistry({"order_status": "https://good.example.com"})
    calls = []

    def counting(url, timeout=10.0):
        calls.append(url)
        return _fake_get(url, timeout)

    with patch("httpx.get", side_effect=counting):
        reg.catalog()
        reg.catalog()
    assert len(calls) == 1  # second call served from cache
