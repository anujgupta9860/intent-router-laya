"""Task-scoped intent persistence with a deterministic decision gate.

The rule: System 1 is invoked ONLY when a genuine decision is needed.
Once a decision is made and a task starts, the intent stays locked and
System 1 stays out of the picture until the task completes, aborts, or
the user switches tasks.

Flow per message:
    needs_decision(text, active_task)?
      ├─ NO  → continue the task: locked intent/worker, fill slots,
      │        dispatch with session context. System 1 NOT called.
      └─ YES → System 1 decides → start a new task (aborting the old
              one if the user switched).

The gate itself is deterministic (patterns, not a model) — that is the
entire point: deciding "does this need a decision" must not itself
require System 1.
"""
from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass, field

# ---------------------------------------------------------------- slots
# Slot name -> pattern that fills it. Workers can advertise more later;
# the router fills what it recognizes and passes the rest through.
SLOT_PATTERNS = {
    "order_id": re.compile(r"#?(\d{5,})"),
}

# ---------------------------------------------------------------- gate
# Phrases that control the task itself (abort), not a new decision.
TASK_CONTROL_ABORT = (
    "never mind", "forget it", "forget about it", "stop", "abort",
    "cancel that", "nevermind",
)

# "I can't give you the ID" → worker recovery, not a new decision.
NO_ID_PHRASES = (
    "don't have", "dont have", "do not have",
    "don't know", "dont know", "do not know",
    "forgot", "lost it", "can't find", "cant find", "no idea",
)

# Action verbs per task action — a *different* action verb than the
# active task's means the user switched tasks → decision needed.
ACTION_VERBS = {
    "track": ("track", "where is", "status of", "locate", "find my"),
    "cancel": ("cancel", "delete my order", "delete order"),
    "modify": ("change", "modify", "update", "edit"),
    "return": ("return", "refund", "send back"),
    "reorder": ("reorder", "buy again", "order again"),
    "place": ("place an order", "place order", "purchase"),
    "list": ("list", "show me", "all orders", "my orders"),
    "estimate": ("eta", "estimate", "when will", "delivery date"),
}


@dataclass
class Task:
    task_id: str
    session_id: str
    intent: str
    worker_agent: str
    action: str | None = None
    state: str = "active"  # active | completed | aborted
    slots: dict = field(default_factory=dict)
    pending_slots: list = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "state": self.state,
            "intent": self.intent,
            "action": self.action,
            "slots": self.slots,
            "pending_slots": self.pending_slots,
        }


class TaskManager:
    """In-memory session → active task map.

    NOTE: in-memory. Cloud Run scale-to-zero drops sessions; a
    persistent store (Firestore) is the production follow-up.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, Task] = {}

    # ------------------------------------------------------------ tasks
    def get_task(self, session_id: str | None) -> Task | None:
        if not session_id:
            return None
        task = self._sessions.get(session_id)
        return task if task and task.state == "active" else None

    def start_task(self, session_id: str, intent: str, worker_agent: str,
                   action: str | None = None,
                   slots: dict | None = None) -> Task:
        old = self._sessions.get(session_id)
        if old and old.state == "active":
            old.state = "aborted"  # user switched tasks
        task = Task(task_id=f"task-{uuid.uuid4().hex[:8]}",
                    session_id=session_id, intent=intent,
                    worker_agent=worker_agent, action=action,
                    slots=dict(slots or {}))
        self._sessions[session_id] = task
        return task

    def complete_task(self, session_id: str) -> None:
        task = self._sessions.get(session_id)
        if task:
            task.state = "completed"

    def abort_task(self, session_id: str) -> Task | None:
        task = self._sessions.get(session_id)
        if task and task.state == "active":
            task.state = "aborted"
        return task

    # ------------------------------------------------------------- gate
    def needs_decision(self, text: str,
                       task: Task | None) -> tuple[bool, str]:
        """Deterministic gate. Returns (decision_needed, reason).

        No model is consulted here — by design.
        """
        t = (text or "").lower().strip()
        if task is None:
            return True, "no_active_task"
        if not t:
            return False, "empty"

        # Task control: abort the task, no decision needed.
        if any(p in t for p in TASK_CONTROL_ABORT):
            return False, "task_control_abort"

        # Bare slot fill: "48291" / "#48291" → continue, fill the slot.
        if re.fullmatch(r"#?\d{5,}", t):
            return False, "slot_fill"

        # "I don't have it" → worker recovery path, not a new decision.
        if any(p in t for p in NO_ID_PHRASES):
            return False, "slot_recovery"

        # A *different* action verb → user switched tasks → decide.
        for action, verbs in ACTION_VERBS.items():
            if action != task.action and any(v in t for v in verbs):
                return True, f"task_switch:{action}"

        # Same task, but a *different* entity ("track #99999" while
        # tracking #48291) → new decision.
        m = SLOT_PATTERNS["order_id"].search(t)
        if m and task.slots.get("order_id") and \
                m.group(1) != task.slots["order_id"]:
            return True, "new_entity"

        # Default: ambiguous → a decision IS needed (System 1 decides).
        return True, "ambiguous"
