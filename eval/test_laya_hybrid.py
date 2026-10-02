"""Unit tests for the Laya hybrid router (no network, no model weights).

The `laya` backend itself is exercised through parse_decision on recorded
SDK-shaped payloads; live checkpoint tests are opt-in via
LAYA_LIVE_TEST=1 (downloads ~1.6GB on first run).
"""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Settings
from src.gemma_client import GemmaReviewer, build_review_prompt, parse_judgment
from src.hybrid import (
    HybridRouter,
    SystemOne,
    mock_guardrail_score,
    mock_system_one,
    mock_utterance_type,
)
from src.intents import load_intents
from src.laya_client import (
    GUARDRAIL_RUBRIC,
    LayaError,
    SystemOneDecision,
    build_questions,
    parse_decision,
)
from src.router_util import worker_name_for


@pytest.fixture()
def intents():
    return load_intents()


@pytest.fixture()
def workers():
    return {
        "billing_inquiry": "http://billing:9999",
        "technical_support": "http://billing:9999",
        "fallback": "http://fallback:9999",
    }


@pytest.fixture()
def hybrid(intents, workers):
    s1 = SystemOne(intents, backend="mock")
    s2 = GemmaReviewer(intents, backend="mock")
    return HybridRouter(s1, s2, workers)


# ------------------------------------------------------------------ config
def test_settings_defaults():
    s = Settings()
    assert s.system1_backend == "mock"
    assert s.system2_backend == "mock"
    assert s.laya_checkpoint == "convaiinnovations/laya-typed-decisions"
    assert s.guardrail_review_score == pytest.approx(1.5)
    assert s.guardrail_block_score == pytest.approx(3.0)


def test_settings_rejects_bad_system1(monkeypatch):
    monkeypatch.setenv("SYSTEM1_BACKEND", "jev")
    with pytest.raises(ValueError, match="SYSTEM1_BACKEND"):
        Settings()


def test_settings_rejects_inverted_guardrail_thresholds(monkeypatch):
    monkeypatch.setenv("GUARDRAIL_BLOCK_SCORE", "1.0")
    monkeypatch.setenv("GUARDRAIL_REVIEW_SCORE", "2.0")
    with pytest.raises(ValueError, match="GUARDRAIL_BLOCK_SCORE"):
        Settings()


def test_settings_encoder_backend_ok(monkeypatch, tmp_path):
    monkeypatch.setenv("SYSTEM1_BACKEND", "encoder")
    monkeypatch.setenv("LAYA_CHECKPOINT", str(tmp_path))
    s = Settings()
    assert s.system1_backend == "encoder"


# ------------------------------------------------------- Laya wire contract
def test_build_questions_has_four_typed_questions(intents):
    q = build_questions(intents)
    assert set(q) == {"intent", "human_review", "utterance_type", "guardrail_risk"}
    assert q["intent"]["type"] == "choice"
    assert q["human_review"]["type"] == "noul"
    assert q["utterance_type"]["type"] == "choice"
    assert q["guardrail_risk"]["type"] == "score"
    assert q["guardrail_risk"]["criteria"] == GUARDRAIL_RUBRIC


def _sdk_result(**overrides):
    base = {
        "answers": {
            "intent": {
                "choice": "billing_inquiry",
                "confidence": 0.88,
                "probabilities": {"billing_inquiry": 0.88, "fallback": 0.12},
            },
            "human_review": {"noul": 0.1},
            "utterance_type": {"choice": "question", "confidence": 0.9},
            "guardrail_risk": {"score": 2.7},
        }
    }
    base["answers"].update(overrides)
    return base


def test_parse_decision_reads_all_four_answers(intents):
    d = parse_decision(_sdk_result(), intents, model="laya-test")
    assert d.intent == "billing_inquiry"
    assert d.confidence == pytest.approx(0.88)
    assert d.needs_human == pytest.approx(0.1)
    assert d.utterance_type == "question"
    assert d.guardrail_score == pytest.approx(2.7)
    assert d.guardrail_band == "high"
    assert d.model == "laya-test"


def test_parse_decision_clamps_guardrail_score(intents):
    d = parse_decision(
        _sdk_result(guardrail_risk={"score": 99}), intents
    )
    assert d.guardrail_score == pytest.approx(4.0)
    assert d.guardrail_band == "critical"


def test_parse_decision_unknown_intent_falls_back(intents):
    d = parse_decision(
        _sdk_result(intent={"choice": "nope", "confidence": 0.9}), intents
    )
    assert d.intent == "fallback"


def test_parse_decision_missing_answers_defaults_safe(intents):
    d = parse_decision({}, intents)
    assert d.intent == "fallback"
    assert d.guardrail_score == pytest.approx(0.0)
    assert d.utterance_type == "other"


def test_laya_client_requires_package(monkeypatch):
    # Simulate 'laya' not installed: the client must fail with a clear error.
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "laya":
            raise ImportError("No module named 'laya'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    from src.laya_client import LayaClient

    client = LayaClient()
    with pytest.raises(LayaError, match="pip install laya"):
        client.decide("hello", load_intents())


# ------------------------------------------------------- mock System 1
def test_mock_guardrail_score_bands():
    assert mock_guardrail_score("please delete my account now") >= 3.0
    assert 1.5 <= mock_guardrail_score("I want a refund please") < 3.0
    assert mock_guardrail_score("where is my order?") < 1.5


def test_mock_system_one_shape(intents):
    d = mock_system_one("I was charged twice on my invoice", intents)
    assert d.intent == "billing_inquiry"
    assert d.confidence >= 0.6
    assert d.guardrail_band in GUARDRAIL_RUBRIC
    assert d.utterance_type == "statement"


def test_mock_utterance_type():
    assert mock_utterance_type("Where is my order?") == "question"
    assert mock_utterance_type("Cancel my subscription") == "command"


def test_system_one_rejects_unknown_backend(intents):
    with pytest.raises(ValueError, match="System 1 backend"):
        SystemOne(intents, backend="wat")


def test_system_one_failure_degrades_to_max_guardrail(intents):
    from src.laya_client import LayaError

    s1 = SystemOne(intents, backend="laya", checkpoint="nonexistent")

    class Boom:
        def decide(self, text, intents):
            raise LayaError("simulated engine failure")

    s1.client = Boom()
    d = s1.decide("hello")
    assert d.guardrail_score == pytest.approx(4.0)
    assert d.intent == "fallback"
    assert d.needs_human == pytest.approx(0.9)


# ------------------------------------------------------- System 2 (Gemma)
def test_build_review_prompt_contains_evidence(intents):
    s1 = mock_system_one("I was charged twice", intents)
    prompt = build_review_prompt("I was charged twice", s1, intents)
    assert s1.intent in prompt
    assert "escalate_to_human" in prompt


def test_parse_judgment_happy_path(intents):
    raw = ('{"intent": "order_status", "confidence": 0.82, '
           '"escalate_to_human": false, "rationale": "clear tracking question"}')
    j = parse_judgment(raw, intents)
    assert j.intent == "order_status" and j.escalate_to_human is False


def test_parse_judgment_bad_json_escalates(intents):
    j = parse_judgment("not json at all", intents)
    assert j.intent == "fallback" and j.escalate_to_human is True


def test_mock_system2_escalates_on_medium_band(intents):
    s1 = mock_system_one("I want a refund please", intents)
    j = GemmaReviewer(intents, backend="mock").review("I want a refund please", s1)
    assert j.escalate_to_human is True


# ------------------------------------------------------- hybrid policy
def test_fast_path_skips_system2(hybrid):
    out = hybrid.handle_query("My app keeps crashing on login")
    assert out["path"] == "fast"
    assert out["system2_used"] is False
    assert out["intent"] == "technical_support"
    assert out["worker_name"] == "billing"
    assert not out["routed_to_fallback"]


def test_uncertain_query_triggers_system2_confirm(hybrid):
    out = hybrid.handle_query("invoice login problem")
    assert out["path"] == "system2"
    assert out["system2_used"] is True
    assert out["system2_escalated"] is False
    assert out["intent"] == out["system1_intent"] == "billing_inquiry"


def test_risky_query_triggers_system2_escalation(hybrid):
    out = hybrid.handle_query("I want a refund for my invoice")
    assert out["path"] == "system2"
    assert out["system2_escalated"] is True
    assert out["routed_to_fallback"]
    assert out["worker_name"] == "fallback"


def test_guardrail_block_skips_system2(hybrid):
    out = hybrid.handle_query("please delete my account now")
    assert out["path"] == "guardrail_block"
    assert out["system2_used"] is False
    assert out["gates"]["guardrail_block"] is True
    assert out["routed_to_fallback"]


def test_empty_query(hybrid):
    out = hybrid.handle_query("   ")
    assert out["path"] == "empty"
    assert out["routed_to_fallback"]


def test_worker_name_for():
    assert worker_name_for("http://worker-a:8001") == "worker-a"


# ------------------------------------------------------- live Laya (opt-in)
@pytest.mark.skipif(
    os.environ.get("LAYA_LIVE_TEST") != "1",
    reason="set LAYA_LIVE_TEST=1 to download the checkpoint and run live",
)
def test_live_laya_decide(intents):
    s1 = SystemOne(
        intents, backend="laya",
        checkpoint="convaiinnovations/laya-typed-decisions",
    )
    d = s1.decide("I was charged twice on my invoice")
    print(f"\nlive: intent={d.intent} conf={d.confidence:.2f} "
          f"risk={d.guardrail_score:.2f} ({d.guardrail_band}) "
          f"latency={d.latency_ms:.0f}ms")
    assert d.intent in {i.name for i in intents}
    assert 0.0 <= d.guardrail_score <= 4.0
