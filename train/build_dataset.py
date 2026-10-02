"""Build a labeled fine-tuning dataset from the intent taxonomy.

Reads data/intents.yaml (intent descriptions + example queries) and emits
JSONL with one record per training example:

    {"query": "...", "intent": "billing_inquiry",
     "needs_human": 0.0, "utterance_type": "question",
     "guardrail_band": "safe"}

Augmentation: each seed example is expanded with lightweight template
paraphrases (deterministic, no LLM needed) so a small seed set becomes a
trainable dataset. For production, replace/augment with real labeled
traffic - see train/README.md.

Usage:
    python train/build_dataset.py --out train/dataset.jsonl --repeat 8
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import yaml

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


def _utterance_type(query: str) -> str:
    q = query.strip()
    if q.endswith("?"):
        return "question"
    first = q.split()[0].lower() if q.split() else ""
    if first in {"cancel", "update", "change", "send", "track", "give",
                 "show", "please", "tell", "refund", "pay", "delete"}:
        return "command"
    return "statement"


def build(intents_path: Path, repeat: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    data = yaml.safe_load(intents_path.read_text())
    intents = data["intents"] if isinstance(data, dict) else data
    records: list[dict] = []
    for intent in intents:
        name = intent["name"]
        examples = intent.get("examples", []) or [intent.get("description", "")]
        for ex in examples:
            for i in range(repeat):
                template = _TEMPLATES[(i + rng.randrange(len(_TEMPLATES))) % len(_TEMPLATES)]
                query = template.format(q=ex.strip())
                records.append({
                    "query": query,
                    "intent": name,
                    "needs_human": 1.0 if name == "fallback" else 0.0,
                    "utterance_type": _utterance_type(query),
                    "guardrail_band": _RISK_BANDS.get(name, "safe"),
                })
    rng.shuffle(records)
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--intents", default="data/intents.yaml")
    parser.add_argument("--out", default="train/dataset.jsonl")
    parser.add_argument("--repeat", type=int, default=8,
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
    print(f"wrote {len(records)} records -> {out} "
          f"({len(intents)} intents: {', '.join(intents)})")


if __name__ == "__main__":
    main()
