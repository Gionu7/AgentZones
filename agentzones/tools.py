"""Tool registry of a zone: MCP tools + built-in zone tools.

Qualified names are ``<server>.<tool>`` (e.g. ``proxy.check_https``).
For the LLM we use ``<server>__<tool>`` because OpenAI-style function names
may not contain dots.
"""

from __future__ import annotations

import fnmatch
import json
import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from .mcp_lite import HttpMCPClient, StdioMCPClient, _BaseClient
from .config import MCPServerConfig

log = logging.getLogger(__name__)

Handler = Callable[[dict[str, Any]], Awaitable[Any]]


@dataclass
class Tool:
    name: str  # qualified
    description: str
    input_schema: dict[str, Any]
    handler: Handler

    @property
    def llm_name(self) -> str:
        return self.name.replace(".", "__")

    def openai(self) -> dict[str, Any]:
        return {"type": "function", "function": {
            "name": self.llm_name, "description": self.description[:1000], "parameters": self.input_schema,
        }}


class ToolRegistry:
    def __init__(self) -> None:
        self.tools: dict[str, Tool] = {}
        self.clients: list[_BaseClient] = []

    def add(self, tool: Tool) -> None:
        self.tools[tool.name] = tool

    def builtin(self, name: str, description: str, schema: dict[str, Any] | None = None):
        def deco(fn: Handler) -> Handler:
            self.add(Tool(f"zone.{name}", description, schema or {"type": "object", "properties": {}}, fn))
            return fn
        return deco

    async def start_mcp(self, servers: list[MCPServerConfig]) -> None:
        for s in servers:
            client: _BaseClient
            if s.command:
                client = StdioMCPClient(s.name, s.command, s.args, s.env)
            else:
                client = HttpMCPClient(s.name, s.url or "", s.headers)
            try:
                await client.start()
            except Exception as exc:
                log.error("MCP server %s failed to start: %s", s.name, exc)
                await client.close()
                continue
            self.clients.append(client)
            for t in client.tools:
                self.add(Tool(f"{s.name}.{t.name}", t.description, t.input_schema, self._mcp_handler(client, t.name)))
            log.info("MCP server %s: %d tools", s.name, len(client.tools))

    @staticmethod
    def _mcp_handler(client: _BaseClient, tool_name: str) -> Handler:
        async def h(args: dict[str, Any]) -> Any:
            text, is_err = await client.call_tool(tool_name, args)
            if is_err:
                raise RuntimeError(text)
            try:
                return json.loads(text)
            except (json.JSONDecodeError, TypeError):
                return text
        return h

    def select(self, patterns: list[str]) -> list[Tool]:
        if not patterns:
            return list(self.tools.values())
        return [t for t in self.tools.values() if any(fnmatch.fnmatch(t.name, p) for p in patterns)]

    def by_llm_name(self, llm_name: str) -> Tool | None:
        return self.tools.get(llm_name.replace("__", ".", 1))

    async def call(self, name: str, args: dict[str, Any]) -> Any:
        tool = self.tools.get(name)
        if not tool:
            raise KeyError(f"unknown tool {name}")
        return await tool.handler(args or {})

    async def close(self) -> None:
        for c in self.clients:
            await c.close()
