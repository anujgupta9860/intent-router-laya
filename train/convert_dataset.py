"""Convert intent-router-laya's dataset.jsonl to Laya RLCD training format.

Reads:  train/dataset.jsonl  (lines: {query, intent, needs_human,
        utterance_type, guardrail_band, worker_agent, skill_required,
        needs_rag, needs_more_input, needs_user_details, is_multi_turn,
        needs_async})
Writes: train/laya_rlcd.jsonl (lines: {state, questions, gold})

Question phrasing is imported VERBATIM from the repo's
train/system1_questions.py — the single source of truth shared with
serving (src/laya_client.build_questions). Never rephrase here.

The output matches the format of the upstream LocalLLaMA/typed-decisions
dataset used by laya's fine-tuning notebook:
  - state:       the raw query text
  - questions:   {qid: {type, instructions, criteria}}
  - gold:        {qid: {probabilities, label}}

Question types follow the notebook's build_training_item():
  - choice: criteria = {option: description}, probabilities one-hot by key
  - noul:   probabilities = {"false": p, "true": p}
  - score:  criteria = [level names], probabilities = {"0":.., "1":..} by index

Usage (from the repo root on the training VM):
    python train/build_dataset.py --out train/dataset.jsonl --repeat 14
    python train/convert_dataset.py --in train/dataset.jsonl --out train/laya_rlcd.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# The shared spec lives in the repo at src/system1_questions.py.
# This script may run as train/convert_dataset.py (repo root on sys.path
# is NOT automatic), so add the repo root explicitly.
_here = Path(__file__).resolve().parent
_repo_root = _here.parent if _here.name == "train" else _here
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
from src.system1_questions import (  # type: ignore  # noqa: E402
    GUARDRAIL_LEVELS,
    INTENT_DESCRIPTIONS,
    QUESTIONS,
    SKILL_DESCRIPTIONS,
    UTTERANCE_DESCRIPTIONS,
    WORKER_DESCRIPTIONS,
)

#: dataset.jsonl label key -> RLCD question id for the noul questions.
_NOUL_LABELS = {
    "needs_human": "needs_human",
    "needs_rag": "needs_rag",
    "needs_more_input": "needs_more_input",
    "needs_user_details": "needs_user_details",
    "is_multi_turn": "is_multi_turn",
    "needs_async": "needs_async",
}


def _choice_gold(options: dict, value: str) -> dict:
    if value not in options:
        value = next(iter(options))
    return {
        "probabilities": {k: (1.0 if k == value else 0.0) for k in options},
        "label": value,
    }


def _noul_gold(p: float) -> dict:
    p = max(0.0, min(1.0, float(p)))
    return {
        "probabilities": {"false": 1.0 - p, "true": p},
        "label": "true" if p >= 0.5 else "false",
    }


def to_rlcd(rec: dict) -> dict:
    query = rec["query"]

    questions = {qid: dict(q) for qid, q in QUESTIONS.items()}
    # criteria dicts are shared references; copy to be safe.
    for q in questions.values():
        if isinstance(q.get("criteria"), dict):
            q["criteria"] = dict(q["criteria"])
        elif isinstance(q.get("criteria"), list):
            q["criteria"] = list(q["criteria"])

    band = rec.get("guardrail_band", "safe")
    band_idx = GUARDRAIL_LEVELS.index(band) if band in GUARDRAIL_LEVELS else 0
    gold = {
        "intent": _choice_gold(INTENT_DESCRIPTIONS, rec["intent"]),
        "utterance_type": _choice_gold(
            UTTERANCE_DESCRIPTIONS, rec.get("utterance_type", "statement")),
        "worker_agent": _choice_gold(
            WORKER_DESCRIPTIONS, rec.get("worker_agent", "fallback")),
        "skill_required": _choice_gold(
            SKILL_DESCRIPTIONS, rec.get("skill_required", "none")),
        "guardrail": {
            "probabilities": {str(i): (1.0 if i == band_idx else 0.0)
                              for i in range(len(GUARDRAIL_LEVELS))},
            "label": band_idx,
        },
    }
    for label_key, qid in _NOUL_LABELS.items():
        gold[qid] = _noul_gold(rec.get(label_key, 0.0))

    # Every emitted question must have gold, and vice versa.
    assert set(questions) == set(gold), (
        f"question/gold mismatch: {set(questions) ^ set(gold)}")
    return {"state": query, "questions": questions, "gold": gold}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="train/dataset.jsonl")
    ap.add_argument("--out", default="train/laya_rlcd.jsonl")
    args = ap.parse_args()

    inp, out = Path(args.inp), Path(args.out)
    n = 0
    with inp.open() as fin, out.open("w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            fout.write(json.dumps(to_rlcd(json.loads(line))) + "\n")
            n += 1
    n_q = len(QUESTIONS)
    print(f"wrote {n} RLCD rows -> {out} ({n} x {n_q} = {n * n_q} sequences)")


if __name__ == "__main__":
    main()
