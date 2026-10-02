"""Stub A2A worker agent for local demos and integration tests.

Run one instance per worker, picking its identity with WORKER_NAME:

    WORKER_NAME=worker-a uvicorn src.workers.stub_worker:app --port 8001

Exposes:
  GET  /.well-known/agent-card.json  - A2A agent card
  POST /a2a/tasks                    - accept a routed task, return a canned response
"""
from __future__ import annotations

import os
import time
import uuid

from fastapi import FastAPI
from pydantic import BaseModel, Field

WORKER_NAME = os.environ.get("WORKER_NAME", "worker-a")

app = FastAPI(title=f"Stub worker: {WORKER_NAME}", version="0.1.0")


class TaskRequest(BaseModel):
    text: str = ""
    intent: str = ""
    task_id: str | None = None


class TaskResponse(BaseModel):
    task_id: str
    worker: str
    status: str
    intent: str
    response: str
    ts: float = Field(default_factory=time.time)


@app.get("/.well-known/agent-card.json")
def agent_card():
    return {
        "name": WORKER_NAME,
        "description": f"Stub worker agent {WORKER_NAME} for intent-router demos",
        "url": f"http://{WORKER_NAME}:8000",
        "version": "0.1.0",
        "capabilities": {"streaming": False, "pushNotifications": False},
        "skills": [
            {
                "id": "handle_task",
                "name": "Handle routed task",
                "description": "Accepts a user query routed by the intent router "
                               "and returns a canned response.",
                "tags": ["routing", "stub"],
            }
        ],
    }


@app.post("/a2a/tasks", response_model=TaskResponse)
def create_task(req: TaskRequest):
    task_id = req.task_id or f"task-{uuid.uuid4().hex[:12]}"
    snippet = req.text[:140]
    return TaskResponse(
        task_id=task_id,
        worker=WORKER_NAME,
        status="completed",
        intent=req.intent,
        response=f"[{WORKER_NAME}] handled intent '{req.intent}': {snippet}",
    )


@app.get("/health")
def health():
    return {"status": "ok", "worker": WORKER_NAME}
