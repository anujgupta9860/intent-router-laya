"""Execution plane: ADK runs every skill invocation.

Architecture:
  System 1 (Laya) ... decides WHAT — intent, worker_agent, skill_required,
                      needs_human, guardrail (every decision, typed outputs)
  System 2 (Gemma) ... generates TOKENS — rationales, messages (no decisions)
  ADK (this module) .. executes HOW — every skill call runs as an ADK
                       FunctionTool via run_async + ToolContext.

There are no bare httpx calls in the routing path. System 1's decision
(skill + target agent) enters here; ADK's tool machinery executes it,
whether the skill lives behind A2A (/message) or MCP (/mcp tools/call).
"""
from __future__ import annotations

import asyncio
import logging
import uuid

log = logging.getLogger(__name__)


def _tool_context():
    """Minimal ADK ToolContext for programmatic (non-LLM) tool execution."""
    from google.adk.tools.tool_context import ToolContext
    return ToolContext.__new__(ToolContext)


def _run(coro):
    """Run an ADK coroutine from sync router code."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    # Already inside a loop (shouldn't happen in the sync path, but safe):
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


# ------------------------------------------------------------------ tools
def _a2a_tool_fn(agent_url: str, text: str, intent: str,
                 context: dict | None):
    """Build the sync function behind the A2A dispatch FunctionTool."""
    def a2a_dispatch() -> dict:
        """Dispatch a message to a remote A2A worker agent."""
        import httpx
        url = agent_url.rstrip("/") + "/message"
        resp = httpx.post(url, json={"text": text, "intent": intent,
                                    "context": context or {}},
                          timeout=30.0)
        resp.raise_for_status()
        return resp.json()
    a2a_dispatch.__name__ = "a2a_dispatch"
    return a2a_dispatch


def _mcp_tool_fn(mcp_url: str, tool_name: str, args: dict):
    """Build the sync function behind an MCP tool-call FunctionTool."""
    def mcp_call() -> dict:
        """Call a tool on a remote MCP server."""
        import httpx
        payload = {"jsonrpc": "2.0", "id": f"adk-{uuid.uuid4().hex[:8]}",
                   "method": "tools/call",
                   "params": {"name": tool_name, "arguments": args}}
        resp = httpx.post(mcp_url, json=payload, timeout=30.0)
        resp.raise_for_status()
        data = resp.json()
        if "error" in data:
            raise RuntimeError(f"MCP error: {data['error']}")
        return data.get("result", {})
    mcp_call.__name__ = f"mcp_{tool_name}"
    return mcp_call


# -------------------------------------------------------------- executor
class AdkSkillExecutor:
    """Executes System 1's skill decisions through ADK's tool machinery."""

    def __init__(self, skill_registry=None) -> None:
        self.registry = skill_registry

    # ---------------------------------------------------------- sync API
    def dispatch(self, agent_url: str, text: str, intent: str,
                 context: dict | None = None) -> dict:
        """Route-path dispatch: query -> worker agent, executed via ADK.

        Returns the same shape as the old A2AClient.dispatch
        (task_id, status, agent_url, intent, mode, result).
        """
        return _run(self._dispatch_async(agent_url, text, intent, context))

    def call_skill(self, skill: str, agent_url: str,
                   args: dict | None = None,
                   protocol: str = "auto") -> dict:
        """Direct skill invocation. MCP is preferred when the agent
        offers the skill over MCP; otherwise falls back to A2A."""
        return _run(self._call_skill_async(skill, agent_url, args or {},
                                           protocol))

    # -------------------------------------------------------- async core
    async def _dispatch_async(self, agent_url: str, text: str,
                              intent: str,
                              context: dict | None) -> dict:
        from google.adk.tools import FunctionTool
        tool = FunctionTool(_a2a_tool_fn(agent_url, text, intent, context))
        try:
            data = await tool.run_async(args={}, tool_context=_tool_context())
        except Exception as exc:
            log.warning("ADK A2A dispatch to %s failed: %s", agent_url, exc)
            return {
                "task_id": f"task-{uuid.uuid4().hex[:12]}",
                "status": "error",
                "agent_url": agent_url, "intent": intent,
                "mode": "adk-a2a",
                "error": str(exc),
            }
        return {
            "task_id": data.get("task_id", f"task-{uuid.uuid4().hex[:12]}"),
            "status": data.get("status", "completed"),
            "agent_url": agent_url, "intent": intent,
            "mode": "adk-a2a",
            "result": data,
        }

    async def _call_skill_async(self, skill: str, agent_url: str,
                                args: dict, protocol: str) -> dict:
        from google.adk.tools import FunctionTool
        base = agent_url.rstrip("/")
        use_mcp = protocol in ("mcp", "auto") and self._offers_mcp(skill, base)
        if use_mcp:
            tool = FunctionTool(_mcp_tool_fn(base + "/mcp", skill, args))
            mode = "adk-mcp"
        else:
            text = args.get("text", skill)
            tool = FunctionTool(_a2a_tool_fn(
                base, text, args.get("intent", ""), args.get("context")))
            mode = "adk-a2a"
        try:
            data = await tool.run_async(args={}, tool_context=_tool_context())
            return {"skill": skill, "agent_url": base, "mode": mode,
                    "ok": True, "result": data}
        except Exception as exc:
            log.warning("ADK skill call %s on %s failed: %s",
                        skill, base, exc)
            return {"skill": skill, "agent_url": base, "mode": mode,
                    "ok": False, "error": str(exc)}

    # ---------------------------------------------------------- internals
    def _offers_mcp(self, skill: str, agent_url: str) -> bool:
        if self.registry is None:
            return False
        try:
            found = self.registry.find_skill(skill)
        except Exception:
            return False
        return bool(found and found["agent_url"].rstrip("/") == agent_url
                    and "mcp" in found.get("protocols", []))
