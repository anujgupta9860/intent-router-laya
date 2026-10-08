"""RLCD feedback loop: log System 2 reviews as future training data.

Every query that falls back to System 2 (Gemma) is, by definition, a case
System 1 found hard. Logging Gemma's judgment on those queries builds the
exact training distribution RLCD needs: hard examples with trusted labels.

Records are appended as JSONL (one JSON object per line) so the log is
crash-safe and streamable. A record looks like::

    {"ts": 1728..., "query": "...",
     "s1": {"intent": ..., "confidence": ..., "worker_agent": ...,
            "skill_required": ..., "needs_human": ..., ... (all 11)},
     "s2": {"intent": ..., "confidence": ..., "escalate_to_human": ...,
            "worker_agent": ..., "skill_required": ..., "needs_rag": ...,
            ... (all 11), "rationale": ...},
     "usable_label": true}

``usable_label`` is false when System 2 escalated to a human (no trustworthy
label) or when the query was a guardrail block / empty. Only usable records
are exported for training.

``feedback_to_dataset`` converts usable records into the training JSONL
format consumed by ``train/finetune.py`` (same schema as
``train/build_dataset.py`` output).
"""
from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path

log = logging.getLogger(__name__)

#: System 1 decision keys captured per record (all 11).
_S1_KEYS = (
    "intent", "confidence", "worker_agent", "worker_confidence",
    "skill_required", "skill_confidence", "needs_human", "utterance_type",
    "guardrail_score", "needs_rag", "needs_more_input", "needs_user_details",
    "is_multi_turn", "needs_async",
)

#: System 2 judgment keys captured per record (all 11 + rationale).
_S2_KEYS = (
    "intent", "confidence", "escalate_to_human", "worker_agent",
    "skill_required", "needs_rag", "needs_more_input", "needs_user_details",
    "is_multi_turn", "needs_async", "rationale",
)


class FeedbackLogger:
    """Thread-safe JSONL appender for System 2 fallback records."""

    def __init__(self, log_dir: str | Path = "feedback", enabled: bool = True):
        self.dir = Path(log_dir)
        self.enabled = enabled
        self._lock = threading.Lock()
        if self.enabled:
            self.dir.mkdir(parents=True, exist_ok=True)
            self.path = self.dir / "fallbacks.jsonl"
        else:
            self.path = None

    def log(self, query: str, s1: dict, s2: dict | None,
            usable_label: bool) -> bool:
        """Append one feedback record. Returns True if written."""
        if not self.enabled or self.path is None:
            return False
        import uuid
        record = {
            "id": uuid.uuid4().hex[:12],
            "ts": time.time(),
            "query": query,
            "s1": {k: s1.get(k) for k in _S1_KEYS},
            "s2": ({k: s2.get(k) for k in _S2_KEYS} if s2 else None),
            "usable_label": bool(usable_label and s2),
            # --- human review gate (2026-10-08): nothing trains until a
            # human approves or corrects the record ---
            "review_status": "pending",   # pending | approved | corrected | rejected
            "reviewed_by": None,
            "reviewed_ts": None,
            "corrections": None,          # human-supplied label overrides
        }
        line = json.dumps(record, ensure_ascii=False)
        try:
            with self._lock:
                with self.path.open("a", encoding="utf-8") as f:
                    f.write(line + "\n")
            return True
        except OSError as exc:
            log.warning("feedback log write failed: %s", exc)
            return False

    def stats(self) -> dict:
        """Count records; approved ones are training candidates."""
        total = usable = pending = approved = 0
        if self.path and self.path.exists():
            try:
                with self.path.open(encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        total += 1
                        if rec.get("usable_label") and rec.get("s2"):
                            usable += 1
                            status = rec.get("review_status", "pending")
                            if status == "pending":
                                pending += 1
                            elif status in ("approved", "corrected"):
                                approved += 1
            except OSError as exc:
                log.warning("feedback stats read failed: %s", exc)
        return {
            "feedback_log": str(self.path) if self.path else None,
            "total_fallbacks": total,
            "usable_labels": usable,
            "pending_review": pending,
            "approved_for_training": approved,
        }

    def list_pending(self, limit: int = 50) -> list[dict]:
        """Return pending-review records (newest first) for the review UI."""
        out: list[dict] = []
        if not (self.path and self.path.exists()):
            return out
        try:
            with self.path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if (rec.get("usable_label") and rec.get("s2")
                            and rec.get("review_status", "pending") == "pending"):
                        out.append(rec)
        except OSError as exc:
            log.warning("feedback pending read failed: %s", exc)
        out.sort(key=lambda r: r.get("ts", 0), reverse=True)
        return out[:limit]

    def review(self, record_id: str, decision: str,
               reviewer: str = "human",
               corrections: dict | None = None) -> bool:
        """Human review gate: approve | correct | reject a record.

        - approved: Gemma's labels stand.
        - corrected: ``corrections`` holds human-supplied label overrides
          (any subset of the s2 keys); these win at export time.
        - rejected: record is excluded from training.
        Returns True if the record was found and updated.
        """
        if decision not in ("approved", "corrected", "rejected"):
            raise ValueError(f"bad review decision: {decision!r}")
        if not (self.path and self.path.exists()):
            return False
        found = False
        with self._lock:
            try:
                lines = self.path.read_text(encoding="utf-8").splitlines()
            except OSError as exc:
                log.warning("feedback review read failed: %s", exc)
                return False
            new_lines: list[str] = []
            for line in lines:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    new_lines.append(line)
                    continue
                if rec.get("id") == record_id:
                    rec["review_status"] = (
                        "corrected" if decision == "corrected"
                        else decision)
                    rec["reviewed_by"] = reviewer
                    rec["reviewed_ts"] = time.time()
                    rec["corrections"] = corrections or None
                    found = True
                    line = json.dumps(rec, ensure_ascii=False)
                new_lines.append(line)
            if found:
                try:
                    self.path.write_text("\n".join(new_lines) + "\n",
                                         encoding="utf-8")
                except OSError as exc:
                    log.warning("feedback review write failed: %s", exc)
                    return False
        return found


def _utterance_type(query: str) -> str:
    q = query.strip()
    if q.endswith("?"):
        return "question"
    first = q.split()[0].lower() if q.split() else ""
    if first in {"cancel", "update", "change", "send", "track", "give",
                 "show", "please", "tell", "refund", "pay", "delete"}:
        return "command"
    return "statement"


_GUARDRAIL_BANDS = ["safe", "low", "medium", "high", "critical"]


def feedback_to_dataset(feedback_path: str | Path,
                        out_path: str | Path) -> dict:
    """Convert HUMAN-APPROVED feedback records into training JSONL.

    Only records with review_status ``approved`` or ``corrected`` are
    exported — pending/rejected records never train. For ``corrected``
    records, the human's ``corrections`` override Gemma's labels.

    Output schema matches ``train/build_dataset.py`` so it feeds straight
    into ``train/finetune.py``::

        {"query": ..., "intent": ..., "worker_agent": ...,
         "skill_required": ..., "needs_human": ..., "utterance_type": ...,
         "guardrail_band": ..., "needs_rag": ..., "needs_more_input": ...,
         "needs_user_details": ..., "is_multi_turn": ..., "needs_async": ...}

    Noul fields are probabilities (0.0/1.0 from boolean judgments).
    Guardrail band comes from System 1's score — System 2 only overrides
    on escalate, which is excluded as unusable.
    """
    feedback_path = Path(feedback_path)
    out_path = Path(out_path)
    written = skipped = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as out:
        if feedback_path.exists():
            with feedback_path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        skipped += 1
                        continue
                    if not rec.get("usable_label") or not rec.get("s2"):
                        skipped += 1
                        continue
                    # --- human review gate: only approved/corrected train ---
                    if rec.get("review_status") not in ("approved", "corrected"):
                        skipped += 1
                        continue
                    s2 = dict(rec["s2"])
                    # human corrections win over Gemma's labels
                    for k, v in (rec.get("corrections") or {}).items():
                        if k in s2:
                            s2[k] = v
                    s1 = rec.get("s1", {}) or {}
                    score = float(s1.get("guardrail_score", 0.0) or 0.0)
                    band = _GUARDRAIL_BANDS[
                        max(0, min(4, int(round(score))))]
                    out.write(json.dumps({
                        "query": rec.get("query", ""),
                        "intent": s2.get("intent", "fallback"),
                        "worker_agent": s2.get("worker_agent", "fallback"),
                        "skill_required": s2.get("skill_required", "none"),
                        "needs_human": 1.0 if s2.get("escalate_to_human") else 0.0,
                        "utterance_type": _utterance_type(rec.get("query", "")),
                        "guardrail_band": band,
                        "needs_rag": float(s2.get("needs_rag", 0.0) or 0.0),
                        "needs_more_input": float(s2.get("needs_more_input", 0.0) or 0.0),
                        "needs_user_details": float(s2.get("needs_user_details", 0.0) or 0.0),
                        "is_multi_turn": float(s2.get("is_multi_turn", 0.0) or 0.0),
                        "needs_async": float(s2.get("needs_async", 0.0) or 0.0),
                    }, ensure_ascii=False) + "\n")
                    written += 1
    return {"written": written, "skipped": skipped,
            "dataset": str(out_path)}
