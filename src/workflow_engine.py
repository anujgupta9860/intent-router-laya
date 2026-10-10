"""Workflow-level decision making.

Pipeline (per the architecture):
    1. System 1 identifies the AGENT (worker_agent)      [existing]
    2. System 1 identifies the WORKFLOW or task           [new `workflow` decision]
    3. Within the workflow: ANALYZE at workflow level     [analyzer /decide + workflow]
    4. DECISION MECHANISM at workflow level               [policy + step transitions]
    5. Steps execute as locked-intent tasks (no System 1 per step)

If the workflow aborts, its state is cleared and the next message
goes back to step 2 (identify a new workflow).

Slot storage: every detail the user enters within a workflow
(order_id, address, items, ...) is extracted and STORED in the
session's workflow state, shared across steps. A step runs only when
its needs_slots are all filled; otherwise the workflow asks for the
missing slot and waits — still no System 1.

Workflow definitions live in the analyzer service (hot-reloaded there,
edited in the Studio). This engine fetches + caches them (TTL).
"""
from __future__ import annotations

import logging
import re
import time
import uuid
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

# ---------------------------------------------------------------- slots
# Generic slot extractors. Domain specifics (catalog items) are the
# worker's job; the worker returns them in result.data["slots"] and the
# engine merges them into the session state.
SLOT_PATTERNS = {
    "order_id": re.compile(r"#?(\d{5,})"),
    "address": re.compile(
        r"(?:ship to|deliver to|delivery address|address:?)\s+(.+)",
        re.IGNORECASE),
    "email": re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),
    "phone": re.compile(r"\+?1?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}"),
}


def extract_slots(text: str) -> dict:
    slots: dict = {}
    for name, pattern in SLOT_PATTERNS.items():
        m = pattern.search(text or "")
        if m:
            slots[name] = m.group(1).strip().rstrip(".")
    return slots


@dataclass
class WorkflowState:
    session_id: str
    workflow_name: str
    definition: dict
    current_step_id: str
    slots: dict = field(default_factory=dict)   # stored user details
    step_history: list = field(default_factory=list)
    messages: list = field(default_factory=list)  # all user messages in session
    # Lifecycle: started → inprogress → completed | aborted
    state: str = "started"

    @property
    def active(self) -> bool:
        return self.state in ("started", "inprogress")

    def to_dict(self) -> dict:
        return {
            "workflow": self.workflow_name,
            "state": self.state,
            "current_step": self.current_step_id,
            "slots": self.slots,
            "steps_done": [h["step"] for h in self.step_history],
            "messages": len(self.messages),
        }


class WorkflowEngine:
    def __init__(self, analyzer_url: str, ttl_s: int = 60) -> None:
        self.analyzer_url = (analyzer_url or "").rstrip("/")
        self._ttl = ttl_s
        self._defs: dict[str, dict] = {}
        self._fetched_at: float = 0.0
        self._sessions: dict[str, WorkflowState] = {}

    # ------------------------------------------------------- definitions
    def refresh_definitions(self, force: bool = False) -> dict[str, dict]:
        """Fetch workflow definitions from the analyzer (TTL-cached)."""
        now = time.time()
        if not force and now - self._fetched_at < self._ttl and self._defs:
            return self._defs
        if not self.analyzer_url:
            return self._defs
        try:
            import httpx
            r = httpx.get(f"{self.analyzer_url}/workflows", timeout=10.0)
            r.raise_for_status()
            defs = {w["name"]: w for w in r.json().get("workflows", [])}
            self._defs = defs
            self._fetched_at = now
            log.info("workflow engine: loaded %d workflow definitions",
                     len(defs))
        except Exception as exc:
            log.warning("workflow engine: definition refresh failed: %s", exc)
        return self._defs

    def identify(self, text: str, agent: str) -> dict | None:
        """System 1's workflow decision (trigger match until the Laya
        workflow head is trained). Returns the workflow definition or
        None for a single task."""
        defs = self.refresh_definitions()
        t = (text or "").lower()
        for wf in defs.values():
            if wf.get("agent") != agent:
                continue
            for trigger in wf.get("triggers", []):
                if trigger.lower() in t:
                    return wf
        return None

    # ------------------------------------------------------------ states
    def get_state(self, session_id: str | None) -> WorkflowState | None:
        if not session_id:
            return None
        st = self._sessions.get(session_id)
        return st if st and st.active else None

    def start(self, session_id: str, wf_def: dict,
              slots: dict | None = None) -> WorkflowState:
        old = self._sessions.get(session_id)
        if old and old.active:
            old.state = "aborted"
        first_step = wf_def["steps"][0]["id"]
        st = WorkflowState(session_id=session_id,
                           workflow_name=wf_def["name"],
                           definition=wf_def,
                           current_step_id=first_step,
                           slots=dict(slots or {}))
        self._sessions[session_id] = st
        return st

    def begin_step(self, session_id: str) -> None:
        """Mark the workflow inprogress when its first step starts."""
        st = self._sessions.get(session_id)
        if st and st.state == "started":
            st.state = "inprogress"

    def abort(self, session_id: str) -> WorkflowState | None:
        st = self._sessions.get(session_id)
        if st and st.active:
            st.state = "aborted"
        return st

    # ----------------------------------------------------- slot storage
    def store_slots(self, session_id: str, text: str,
                    extra: dict | None = None) -> dict:
        """Extract details from the message (and worker-returned slots)
        and STORE them in the session's workflow state. Returns all slots.

        Already-filled slots are NOT overwritten — a zip code inside an
        address must not clobber the stored order_id."""
        st = self._sessions.get(session_id)
        if st is None:
            return dict(extra or {})
        found = extract_slots(text)
        if extra:
            found.update(extra)
        for k, v in found.items():
            if k not in st.slots:
                st.slots[k] = v
        return dict(st.slots)

    # ------------------------------------------------------- transitions
    def current_step(self, st: WorkflowState) -> dict:
        for s in st.definition["steps"]:
            if s["id"] == st.current_step_id:
                return s
        raise KeyError(f"unknown step {st.current_step_id!r}")

    def missing_slots(self, st: WorkflowState) -> list[str]:
        step = self.current_step(st)
        return [s for s in step.get("needs_slots", [])
                if s not in st.slots]

    def advance(self, session_id: str, outcome: str) -> str:
        """Deterministic step transition. outcome: success|failure.

        Returns "next:<step_id>" | "complete" | "abort" | "retry".
        No System 1 involved — the definition decides.
        """
        st = self._sessions.get(session_id)
        if st is None or not st.active:
            return "abort"
        step = self.current_step(st)
        st.step_history.append({"step": step["id"], "outcome": outcome,
                                "ts": time.time()})
        target = step["on_success"] if outcome == "success" \
            else step["on_failure"]
        if target == "complete":
            st.state = "completed"
            return "complete"
        if target == "abort":
            st.state = "aborted"
            return "abort"
        if target == "retry":
            return f"next:{step['id']}"
        st.current_step_id = target
        return f"next:{target}"

    # ------------------------------------------------------------- gate
    ABORT_PHRASES = ("never mind", "forget it", "stop", "abort",
                     "cancel the workflow", "cancel workflow")

    def needs_workflow_decision(self, text: str,
                                st: WorkflowState) -> tuple[bool, str]:
        """Does this message need a NEW workflow-level decision?

        No → continue the current step with stored slots (no System 1).
        """
        t = (text or "").lower().strip()
        if any(p in t for p in self.ABORT_PHRASES):
            return False, "workflow_abort"
        # New workflow trigger while one is active → switch → decide.
        # (Checked by the caller via identify(); here: different action
        # verbs signal the user left the workflow.)
        return False, "workflow_continue"
