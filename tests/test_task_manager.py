"""Tests for the task framework: decision gate + intent persistence."""
import pytest

from src.task_manager import TaskManager


@pytest.fixture
def tm():
    return TaskManager()


def test_no_task_needs_decision(tm):
    need, reason = tm.needs_decision("track my order", None)
    assert need is True and reason == "no_active_task"


def test_slot_fill_continues_without_decision(tm):
    task = tm.start_task("s1", "order_status", "http://worker",
                         action="track")
    need, reason = tm.needs_decision("48291", task)
    assert need is False and reason == "slot_fill"
    need, reason = tm.needs_decision("#48291", task)
    assert need is False


def test_no_id_phrase_continues(tm):
    task = tm.start_task("s1", "order_status", "http://worker",
                         action="track")
    need, reason = tm.needs_decision("i dont have the order number", task)
    assert need is False and reason == "slot_recovery"


def test_task_control_abort_needs_no_decision(tm):
    task = tm.start_task("s1", "order_status", "http://worker",
                         action="track")
    need, reason = tm.needs_decision("never mind", task)
    assert need is False and reason == "task_control_abort"


def test_different_action_verb_triggers_decision(tm):
    task = tm.start_task("s1", "order_status", "http://worker",
                         action="track", slots={"order_id": "48291"})
    need, reason = tm.needs_decision("actually cancel my order", task)
    assert need is True and reason == "task_switch:cancel"


def test_new_entity_triggers_decision(tm):
    task = tm.start_task("s1", "order_status", "http://worker",
                         action="track", slots={"order_id": "48291"})
    need, reason = tm.needs_decision("track #99999", task)
    assert need is True and reason == "new_entity"


def test_same_entity_continues_or_decides(tm):
    # "track #48291" while tracking #48291 — same entity, same action.
    # Not a bare slot fill (has verb), not a switch → ambiguous → decide.
    task = tm.start_task("s1", "order_status", "http://worker",
                         action="track", slots={"order_id": "48291"})
    need, _ = tm.needs_decision("track #48291", task)
    assert need is True  # ambiguous → System 1 confirms


def test_ambiguous_goes_to_system1(tm):
    task = tm.start_task("s1", "order_status", "http://worker",
                         action="track")
    need, reason = tm.needs_decision("hmm what about tomorrow", task)
    assert need is True and reason == "ambiguous"


def test_new_task_aborts_old(tm):
    tm.start_task("s1", "order_status", "http://worker", action="track")
    new = tm.start_task("s1", "order_status", "http://worker",
                        action="cancel")
    assert new.state == "active"
    assert tm.get_task("s1").task_id == new.task_id


def test_completed_task_needs_new_decision(tm):
    tm.start_task("s1", "order_status", "http://worker", action="track")
    tm.complete_task("s1")
    assert tm.get_task("s1") is None
    need, _ = tm.needs_decision("thanks", None)
    assert need is True
