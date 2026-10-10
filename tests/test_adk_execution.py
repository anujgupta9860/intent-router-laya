"""Tests: System 1 owns all decisions; System 2 generates tokens only;
execution runs through ADK."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.adk_execution import AdkSkillExecutor
from src.hybrid import HybridRouter, SystemOneDecision
from src.gemma_client import SystemTwoJudgment


def _s1(**kw):
    base = dict(intent="order_status", confidence=0.9, needs_human=0.1,
                utterance_type="statement", guardrail_score=0.2,
                worker_agent="orders",
                worker_confidence=0.8, skill_required="track_order",
                skill_confidence=0.75, needs_rag=0.0, needs_more_input=0.1,
                needs_user_details=0.1, is_multi_turn=0.0, needs_async=0.0,
                probabilities={"order_status": 1.0}, model="mock")
    base.update(kw)
    return SystemOneDecision(**base)


def _s2(**kw):
    base = dict(intent="billing_inquiry", confidence=0.9,
                escalate_to_human=True,
                rationale="system 2 thinks otherwise",
                worker_agent="billing", skill_required="none")
    base.update(kw)
    return SystemTwoJudgment(**base)


def _router():
    r = HybridRouter.__new__(HybridRouter)
    r.confidence_threshold = 0.6
    r.human_review_threshold = 0.5
    r.guardrail_review_score = 1.5
    r.guardrail_block_score = 3.0
    r.worker_agents = {"fallback": "http://fb:1"}
    r._worker_agent_urls = {}
    r.analyzer = None
    from src.adk_execution import AdkSkillExecutor
    r.executor = AdkSkillExecutor()
    return r


def test_system2_cannot_override_intent():
    r = _router()
    out = r._respond("q", _s1(), _s2(), path="system2",
                     started=0.0)
    # System 1's decision stands; System 2's conflicting intent ignored
    assert out["intent"] == "order_status"
    assert out["system1_intent"] == "order_status"
    assert "system 2 thinks otherwise" in out["system2_rationale"]


def test_system2_cannot_force_escalation():
    r = _router()
    out = r._respond("q", _s1(needs_human=0.1), _s2(escalate_to_human=True),
                     path="system2", started=0.0)
    assert out["intent"] == "order_status"  # not fallback
    assert out["routed_to_fallback"] is False


def test_system1_human_gate_still_escalates():
    r = _router()
    out = r._respond("q", _s1(needs_human=0.9), _s2(escalate_to_human=False),
                     path="system2", started=0.0)
    assert out["intent"] == "fallback"
    assert out["routed_to_fallback"] is True


class _Resp:
    def __init__(self, payload):
        self._p = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._p


def test_adk_executor_dispatch_a2a():
    ex = AdkSkillExecutor()

    def fake_post(url, json=None, timeout=None):
        assert url.endswith("/message")
        assert json["text"] == "hello"
        return _Resp({"task_id": "t1", "status": "completed",
                      "action": "track", "message": "ok"})

    with patch("httpx.post", side_effect=fake_post):
        out = ex.dispatch("http://agent:1", "hello", "order_status",
                          context={"a": 1})
    assert out["mode"] == "adk-a2a"
    assert out["status"] == "completed"
    assert out["result"]["action"] == "track"


def test_adk_executor_dispatch_error_shape():
    ex = AdkSkillExecutor()

    def boom(url, json=None, timeout=None):
        raise RuntimeError("down")

    with patch("httpx.post", side_effect=boom):
        out = ex.dispatch("http://agent:1", "hi", "order_status")
    assert out["status"] == "error"
    assert out["mode"] == "adk-a2a"


def test_adk_executor_mcp_preferred():
    class Reg:
        def find_skill(self, name):
            return {"skill": name,
                    "agent_url": "https://agent.example.com",
                    "protocols": ["a2a", "mcp"]}

    ex = AdkSkillExecutor(skill_registry=Reg())

    def fake_post(url, json=None, timeout=None):
        assert url.endswith("/mcp")
        assert json["method"] == "tools/call"
        assert json["params"]["name"] == "track_order"
        return _Resp({"jsonrpc": "2.0", "id": "x",
                      "result": {"content": [{"type": "text",
                                              "text": "shipped"}]}})

    with patch("httpx.post", side_effect=fake_post):
        out = ex.call_skill("track_order", "https://agent.example.com",
                            args={"order_id": "48291"})
    assert out["mode"] == "adk-mcp" and out["ok"] is True


def test_adk_executor_mcp_falls_back_to_a2a():
    class Reg:
        def find_skill(self, name):
            return {"skill": name, "agent_url": "https://a.example.com",
                    "protocols": ["a2a"]}  # no mcp

    ex = AdkSkillExecutor(skill_registry=Reg())

    def fake_post(url, json=None, timeout=None):
        assert url.endswith("/message")
        return _Resp({"task_id": "t9", "status": "completed"})

    with patch("httpx.post", side_effect=fake_post):
        out = ex.call_skill("track_order", "https://a.example.com",
                            args={"text": "track #1"})
    assert out["mode"] == "adk-a2a" and out["ok"] is True
