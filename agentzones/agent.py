"""Agents hosted by a zone.

* :class:`LLMAgent`  - generic tool-calling loop on the zone's local model.
* custom agents      - subclass :class:`Agent`, configure ``kind: python`` and
                       ``impl: package.module:ClassName``.

Every agent answers :meth:`Agent.ask`. The same entry point serves local
users (CLI / HTTP) and peer zones; ``ctx`` tells them apart and carries the
hop count used to stop question loops across zones.
"""

from __future__ import annotations

import importlib
import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .config import AgentConfig
from .llm import LLMUnavailable

if TYPE_CHECKING:
    from .zone import Zone

log = logging.getLogger(__name__)


@dataclass
class AskContext:
    origin: str = "local"  # zone name of the original asker
    hops: int = 0  # how many zone boundaries the question has crossed
    trace: list[str] = field(default_factory=list)  # zones visited
    request_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"origin": self.origin, "hops": self.hops, "trace": self.trace, "request_id": self.request_id}


class Agent:
    def __init__(self, zone: "Zone", cfg: AgentConfig):
        self.zone, self.cfg = zone, cfg

    @property
    def name(self) -> str:
        return self.cfg.name

    def card(self) -> dict[str, Any]:
        return {"name": self.cfg.name, "description": self.cfg.description, "kind": self.cfg.kind}

    async def start(self) -> None:
        pass

    async def ask(self, question: str, context: dict[str, Any] | None, ctx: AskContext) -> dict[str, Any]:
        raise NotImplementedError

    def tools(self):
        tools = self.zone.tools.select(self.cfg.tools)
        if self.cfg.can_ask_peers:
            names = {t.name for t in tools}
            for extra in ("zone.ask_peers", "zone.list_peers", "zone.call_peer_tool"):
                if extra not in names and extra in self.zone.tools.tools:
                    tools.append(self.zone.tools.tools[extra])
        return tools


def _short(obj: Any, n: int = 6000) -> str:
    s = obj if isinstance(obj, str) else json.dumps(obj, default=str)
    return s if len(s) <= n else s[:n] + "...[truncated]"


async def run_tool_loop(zone: "Zone", messages: list[dict[str, Any]], tools: list, max_steps: int = 8,
                        json_mode: bool = False) -> tuple[str | None, list[dict[str, Any]], str | None]:
    """Generic LLM tool-calling loop. Returns (final_text, steps, error)."""
    steps: list[dict[str, Any]] = []
    allowed = {t.name for t in tools}
    for _ in range(max_steps):
        try:
            reply = await zone.llm.chat(messages, [t.openai() for t in tools] or None, json_mode=json_mode and not tools)
        except LLMUnavailable as exc:
            return None, steps, str(exc)
        if not reply.tool_calls:
            return reply.content, steps, None
        messages.append({"role": "assistant", "content": reply.content or "", "tool_calls": [
            {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": json.dumps(c.arguments)}}
            for c in reply.tool_calls]})
        for call in reply.tool_calls:
            tool = zone.tools.by_llm_name(call.name)
            if tool is None or tool.name not in allowed:
                result: Any = {"error": f"tool {call.name} not available"}
            else:
                try:
                    result = await tool.handler(call.arguments)
                except Exception as exc:
                    result = {"error": f"{type(exc).__name__}: {exc}"}
            steps.append({"tool": tool.name if tool else call.name, "args": call.arguments,
                          "result": _short(result, 1500)})
            messages.append({"role": "tool", "tool_call_id": call.id, "content": _short(result)})
    return None, steps, "step limit reached"


class LLMAgent(Agent):
    async def ask(self, question: str, context: dict[str, Any] | None, ctx: AskContext) -> dict[str, Any]:
        sys_prompt = self.cfg.system_prompt
        if ctx.origin != "local":
            sys_prompt += f"\nThis question comes from peer zone '{ctx.origin}'. Answer concisely."
        user = question if not context else f"{question}\n\nContext:\n{_short(context, 3000)}"
        messages: list[dict[str, Any]] = [{"role": "system", "content": sys_prompt},
                                          {"role": "user", "content": user}]
        answer, steps, err = await run_tool_loop(self.zone, messages, self.tools(), self.cfg.max_steps)
        if err == "step limit reached":
            return {"answer": "Stopped: step limit reached.", "steps": steps}
        if err:
            return {"answer": None, "error": err, "steps": steps}
        return {"answer": answer, "steps": steps}


def build_agent(zone: "Zone", cfg: AgentConfig) -> Agent:
    if cfg.kind == "llm":
        return LLMAgent(zone, cfg)
    if not cfg.impl:
        raise ValueError(f"agent {cfg.name}: kind=python needs impl: 'module:Class'")
    mod, _, cls = cfg.impl.partition(":")
    return getattr(importlib.import_module(mod), cls)(zone, cfg)
