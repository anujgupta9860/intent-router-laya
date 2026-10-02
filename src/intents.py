"""Intent taxonomy loader.

The taxonomy lives in ``data/intents.yaml`` so new intents can be added
without touching code: add a block to the YAML, rebuild, redeploy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class Intent:
    name: str
    description: str
    worker: str
    examples: list[str] = field(default_factory=list)


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_intents(path: str | Path = "data/intents.yaml") -> list[Intent]:
    p = Path(path)
    if not p.is_absolute():
        p = repo_root() / p
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    intents: list[Intent] = []
    for item in data.get("intents", []):
        intents.append(
            Intent(
                name=item["name"],
                description=item.get("description", ""),
                worker=item.get("worker", "fallback"),
                examples=list(item.get("examples", [])),
            )
        )
    if not intents:
        raise ValueError(f"No intents defined in {p}")
    names = [i.name for i in intents]
    if len(set(names)) != len(names):
        raise ValueError(f"Duplicate intent names in {p}")
    return intents


def intent_names(intents: list[Intent]) -> list[str]:
    return [i.name for i in intents]
