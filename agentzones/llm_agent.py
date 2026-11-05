"""Generic tool-calling agent on the zone's local model.

Tools: the MCP servers in ``MCP_SERVERS`` (JSON list, filtered by ``TOOLS``
patterns such as ``proxy.*``) plus two zone tools that go through the gateway:
``zone.list_agents`` and ``zone.send_message``.

Payload: ``{"type": "ask", "question": ..., "context": {...}, "history": [{"role", "content"}]}``
or a plain string. ``history`` carries earlier turns of a chat (the agent itself keeps no state).

Questions from other zones are answered, but with only the tools listed in ``LLM_REMOTE_TOOLS``
(default: none), so another zone can use the model but cannot drive this zone's tools. Set
``LLM_ACCEPT_REMOTE=0`` to refuse other zones altogether. ``hops`` in the payload limits agent-to-agent
chains to ``MAX_HOPS``.
"""

from __future__ import annotations

import contextvars
import json
import logging
from typing import Any

from .agent import Agent, Incoming
from .config import mcp_servers_from_env, model_from_env
from .llm import LLM, LLMUnavailable
from .tools import Tool, ToolRegistry

log = logging.getLogger(__name__)

MAX_HOPS = 2
_hops: contextvars.ContextVar[int] = contextvars.ContextVar("hops", default=0)


def _short(obj: Any, n: int = 6000) -> str:
    s = obj if isinstance(obj, str) else json.dumps(obj, default=str)
    return s if len(s) <= n else s[:n] + "...[truncated]"


class LLMAgent(Agent):
    capabilities = ["llm", "chat"]
    description = "General assistant using this zone's tools and agents"
    accepts = {"ask": "{question, context?} -> {answer, ...}; other zones only if the assistant allows it"}

    async def start(self) -> None:
        c = self.config
        self.description = c.get("AGENT_DESCRIPTION", self.description)
        self.system_prompt = c.get("SYSTEM_PROMPT", "You are a helpful agent inside an Agent Zone. "
                                                   "Use tools to check facts before answering.")
        self.max_steps = int(c.get("MAX_STEPS", "8"))
        self.accept_remote = c.get("LLM_ACCEPT_REMOTE", "1") not in ("0", "false", "no")
        self.llm = LLM(model_from_env(c))
        self.tools = ToolRegistry()
        self._register_zone_tools()
        await self.tools.start_mcp(mcp_servers_from_env(c.get("MCP_SERVERS", "")))
        patterns = [p.strip() for p in c.get("TOOLS", "").split(",") if p.strip()]
        self.selected = self.tools.select(patterns + ["zone.*"]) if patterns else list(self.tools.tools.values())
        remote = [p.strip() for p in c.get("LLM_REMOTE_TOOLS", "").split(",") if p.strip()]
        self.remote_tools = [t for t in self.selected if t in self.tools.select(remote)] if remote else []
        self.model_name = self.llm.cfg.model

    async def stop(self) -> None:
        await self.tools.close()
        await self.llm.close()

    def _register_zone_tools(self) -> None:
        t = self.tools

        @t.builtin("list_agents", "List agents in all zones, optionally only those with a capability. "
                   "Each agent's 'accepts' lists the message types it handles and what they contain.", {
            "type": "object", "properties": {"capability": {"type": "string"}}})
        async def _list(args):
            return await self.gw.directory(args.get("capability") or None)

        @t.builtin("send_message", "Send a message to an agent (in any zone) and return its reply. "
                   "Use a payload type from the agent's 'accepts' (see list_agents), e.g. "
                   "{\"type\": \"ask\", \"question\": ...}.", {
                       "type": "object", "required": ["zone", "agent", "payload"], "properties": {
                           "zone": {"type": "string"}, "agent": {"type": "string"},
                           "payload": {"type": "object"}}})
        async def _send(args):
            hops = _hops.get() + 1
            if hops > MAX_HOPS:
                return {"error": f"hop limit ({MAX_HOPS}) reached; answer with what you have"}
            payload = args.get("payload")
            if isinstance(payload, dict):
                payload = {**payload, "hops": hops}
            return await self.gw.send(args["zone"], args["agent"], payload, timeout=120)

    async def handle(self, msg: Incoming) -> Any:
        p = msg.payload if isinstance(msg.payload, dict) else {"type": "ask", "question": str(msg.payload)}
        if p.get("type", "ask") != "ask":
            return self.unknown(p.get("type"))
        if not msg.local and not self.accept_remote:
            return {"error": f"{self.name} only answers its own zone"}
        _hops.set(int(p.get("hops", 0)))
        return await self.ask(str(p.get("question", "")), p.get("context"),
                              None if msg.local else msg.from_zone, p.get("history"))

    async def ask(self, question: str, context: Any = None, remote_zone: str | None = None,
                  history: Any = None) -> dict[str, Any]:
        tools: list[Tool] = self.remote_tools if remote_zone else self.selected
        sys_prompt = self.system_prompt
        if remote_zone:
            sys_prompt += f"\nThis question comes from zone '{remote_zone}'. Answer concisely." + (
                "" if tools else " You have no tools for this question; answer from what you know.")
        user = question if not context else f"{question}\n\nContext:\n{_short(context, 3000)}"
        messages: list[dict[str, Any]] = [{"role": "system", "content": sys_prompt}]
        for turn in (history or [])[-8:] if isinstance(history, list) else []:  # earlier turns of the chat
            if isinstance(turn, dict) and turn.get("role") in ("user", "assistant") and turn.get("content"):
                messages.append({"role": turn["role"], "content": str(turn["content"])[:4000]})
        messages.append({"role": "user", "content": user})
        steps: list[dict[str, Any]] = []
        info = {"model": self.model_name, "tools": [t.name for t in tools], "remote": bool(remote_zone)}
        for _ in range(self.max_steps):
            try:
                reply = await self.llm.chat(messages, [t.openai() for t in tools] or None)
            except LLMUnavailable as exc:
                return {"answer": None, "error": str(exc), "steps": steps, **info}
            if not reply.tool_calls:
                return {"answer": reply.content, "steps": steps, **info}
            messages.append({"role": "assistant", "content": reply.content or "", "tool_calls": [
                {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": json.dumps(c.arguments)}}
                for c in reply.tool_calls]})
            for call in reply.tool_calls:
                tool = self.tools.by_llm_name(call.name)
                if tool is None or tool not in tools:
                    result: Any = {"error": f"tool {call.name} not available"}
                else:
                    try:
                        result = await tool.handler(call.arguments)
                    except Exception as exc:
                        result = {"error": f"{type(exc).__name__}: {exc}"}
                steps.append({"tool": tool.name if tool else call.name, "args": call.arguments,
                              "result": _short(result, 1500)})
                messages.append({"role": "tool", "tool_call_id": call.id, "content": _short(result)})
        return {"answer": "Stopped: step limit reached.", "steps": steps, **info}
