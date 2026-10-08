"""Build a labeled fine-tuning dataset from the intent taxonomy.

Reads data/intents.yaml (intent descriptions + example queries) and emits
JSONL with one record per training example. Each record carries the four
original System 1 labels plus the seven routing decisions added 2026-10-08:

    {"query": "...", "intent": "billing_inquiry",
     "needs_human": 0.0, "utterance_type": "question",
     "guardrail_band": "safe",
     "worker_agent": "billing", "skill_required": "billing_lookup",
     "needs_rag": 0.0, "needs_more_input": 0.0,
     "needs_user_details": 1.0, "is_multi_turn": 0.0, "needs_async": 0.0}

Labels are derived deterministically from the seed example (BEFORE
template paraphrasing) so the template noise never changes a label.
Synthetic follow-up / how-to examples are appended in-code (they are
training data, not taxonomy — data/intents.yaml is untouched).

Usage:
    python train/build_dataset.py --out train/dataset.jsonl --repeat 14
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

import yaml

# Canonical question spec lives in src/; import it as the single source.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.system1_questions import (  # noqa: E402
    INTENT_DESCRIPTIONS,
    SKILL_DESCRIPTIONS,
    WORKER_DESCRIPTIONS,
)

# Deterministic paraphrase templates: {q} is the seed query.
_TEMPLATES = [
    "{q}",
    "{q} Please help.",
    "Hi, {q}",
    "Hello, {q}",
    "Quick question: {q}",
    "Can you help me? {q}",
]

_RISK_BANDS = {
    "billing_inquiry": "low",
    "technical_support": "safe",
    "sales_question": "safe",
    "account_update": "low",
    "order_status": "safe",
    "fallback": "safe",
}

_INTENT_TO_WORKER = {
    "billing_inquiry": "billing",
    "technical_support": "support",
    "sales_question": "sales",
    "account_update": "account",
    "order_status": "orders",
    "fallback": "fallback",
}

_ORDER_NUMBER_RE = re.compile(r"#\d+|order\s*(number|id|no\.?|#)?\s*\d{4,}|\b\d{5,}\b", re.I)
_PRONOUN_FRAGMENT_RE = re.compile(
    r"^(yes|no|yeah|nope|okay|ok|sure)[,.]?\s+(that|this|the|it)\b"
    r"|\b(what about|how about|and then|and the|the (second|first|other) one)\b",
    re.I,
)
_HOWTO_RE = re.compile(r"\bhow (do|can|to)\b|\bwhere (are|is) the\b", re.I)
_REFUND_RE = re.compile(r"\brefund\b|\bovercharg|\bcharged twice\b|\bfix it\b", re.I)
_DEMO_UPGRADE_RE = re.compile(r"\bdemo\b|\bupgrade\b|\bpurchase\b|\bbuy\b", re.I)
_INFO_RE = re.compile(
    r"\b(pricing|plans|features|trial|discount|offer|available)\b", re.I
)

# Synthetic training examples (follow-ups, how-tos). Kept here — not in
# data/intents.yaml — because they teach *routing decisions*, not intents.
# Format: (query, intent).
_SYNTHETIC = [
    ("What about the pro plan?", "sales_question"),
    ("And the delivery date?", "order_status"),
    ("Yes, that one", "order_status"),
    ("Change it to the new one", "account_update"),
    ("What about my invoice from last month?", "billing_inquiry"),
    ("It still doesn't work", "technical_support"),
    ("No, the other one", "fallback"),
    ("How do I reset my password?", "technical_support"),
    ("And can you refund that too?", "billing_inquiry"),
    ("The second one", "order_status"),
    ("What about annual billing?", "sales_question"),
    ("Thanks, and what about the refund status?", "billing_inquiry"),
    ("How do I enable two-factor authentication?", "technical_support"),
    ("Where are the account settings for notifications?", "account_update"),
]


def _utterance_type(query: str) -> str:
    q = query.strip()
    if q.endswith("?"):
        return "question"
    first = q.split()[0].lower() if q.split() else ""
    if first in {"cancel", "update", "change", "send", "track", "give",
                 "show", "please", "tell", "refund", "pay", "delete"}:
        return "command"
    return "statement"


def _skill_required(intent: str, query: str) -> str:
    q = query.lower()
    if intent == "fallback":
        return "escalation"
    if intent == "billing_inquiry":
        return "refund_process" if _REFUND_RE.search(q) else "billing_lookup"
    if intent == "technical_support":
        return "knowledge_search" if _HOWTO_RE.search(q) else "none"
    if intent == "sales_question":
        if _DEMO_UPGRADE_RE.search(q):
            return "none"
        return "knowledge_search" if _INFO_RE.search(q) else "none"
    if intent == "account_update":
        return "knowledge_search" if _HOWTO_RE.search(q) else "account_modify"
    if intent == "order_status":
        return "order_tracking"
    return "none"


def _needs_more_input(intent: str, query: str) -> float:
    q = query.lower()
    if intent == "order_status" and not _ORDER_NUMBER_RE.search(query):
        return 1.0
    if intent == "billing_inquiry" and re.search(
            r"\bhigher than usual\b|\bseems (high|wrong|off)\b", q):
        return 1.0
    if _PRONOUN_FRAGMENT_RE.search(query):
        return 1.0
    if len(query.split()) <= 3 and intent == "fallback":
        return 1.0
    return 0.0


def _needs_user_details(intent: str, query: str) -> float:
    q = query.lower()
    if intent == "account_update":
        return 1.0
    if intent == "billing_inquiry":
        return 1.0
    if intent == "order_status":
        return 1.0
    if intent == "technical_support" and re.search(
            r"\blog ?in\b|\bpassword\b|\bmy account\b", q):
        return 1.0
    if intent == "sales_question" and _DEMO_UPGRADE_RE.search(q):
        return 1.0
    return 0.0


def _needs_async(intent: str, skill: str, query: str) -> float:
    # Refunds settle asynchronously; everything else in the taxonomy is
    # answerable inline.
    if skill == "refund_process":
        return 1.0
    if re.search(r"\bgenerate\b|\bexport\b.*\breport\b", query.lower()):
        return 1.0
    return 0.0


def _label(intent: str, example: str, synthetic: bool = False) -> dict:
    skill = _skill_required(intent, example)
    return {
        "intent": intent,
        "needs_human": 1.0 if intent == "fallback" else 0.0,
        "utterance_type": _utterance_type(example),
        "guardrail_band": _RISK_BANDS.get(intent, "safe"),
        "worker_agent": _INTENT_TO_WORKER[intent],
        "skill_required": skill,
        "needs_rag": 1.0 if skill == "knowledge_search" else 0.0,
        "needs_more_input": _needs_more_input(intent, example),
        "needs_user_details": _needs_user_details(intent, example),
        "is_multi_turn": 1.0 if (
            synthetic or _PRONOUN_FRAGMENT_RE.search(example)) else 0.0,
        "needs_async": _needs_async(intent, skill, example),
    }


def build(intents_path: Path, repeat: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    data = yaml.safe_load(intents_path.read_text())
    intents = data["intents"] if isinstance(data, dict) else data
    seeds: list[tuple[str, str, bool]] = []
    for intent in intents:
        name = intent["name"]
        examples = intent.get("examples", []) or [intent.get("description", "")]
        for ex in examples:
            seeds.append((name, ex.strip(), False))
    for query, intent in _SYNTHETIC:
        seeds.append((intent, query, True))

    # Sanity: every label value must be a declared option.
    assert set(_INTENT_TO_WORKER.values()) <= set(WORKER_DESCRIPTIONS)
    assert set(INTENT_DESCRIPTIONS) >= {s[0] for s in seeds}

    records: list[dict] = []
    for intent, example, synthetic in seeds:
        labels = _label(intent, example, synthetic)
        assert labels["skill_required"] in SKILL_DESCRIPTIONS, labels
        for i in range(repeat):
            template = _TEMPLATES[(i + rng.randrange(len(_TEMPLATES))) % len(_TEMPLATES)]
            query = template.format(q=example)
            records.append({"query": query, **labels})
    rng.shuffle(records)
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--intents", default="data/intents.yaml")
    parser.add_argument("--out", default="train/dataset.jsonl")
    parser.add_argument("--repeat", type=int, default=14,
                        help="paraphrases per seed example")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = build(Path(args.intents), args.repeat, args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    intents = sorted({r["intent"] for r in records})
    skills = sorted({r["skill_required"] for r in records})
    print(f"wrote {len(records)} records -> {out} "
          f"({len(intents)} intents: {', '.join(intents)})")
    print(f"skills: {', '.join(skills)}")
    for lbl in ("needs_rag", "needs_more_input", "needs_user_details",
                "is_multi_turn", "needs_async"):
        pos = sum(1 for r in records if r[lbl] >= 0.5)
        print(f"  {lbl}: {pos}/{len(records)} positive")


if __name__ == "__main__":
    main()
