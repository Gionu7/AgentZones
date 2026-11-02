"""Minimal, dependency-free Model Context Protocol (MCP) client and server.

Why not the official SDK? Its Python API changes between major versions
(v1 ``FastMCP`` vs v2 ``MCPServer``), while the wire protocol is stable.
A zone only needs ``initialize``, ``tools/list`` and ``tools/call``; this
module implements those over the two standard transports:

* stdio            - newline-delimited JSON-RPC 2.0 on a child process
* streamable HTTP  - JSON-RPC POSTed to one URL, JSON or SSE responses

Servers written with :class:`MiniMCPServer` are regular MCP servers and also
work with Claude Desktop, VS Code, etc. Servers written with any official SDK
work with :class:`StdioMCPClient` / :class:`HttpMCPClient`.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import sys
import traceback
import typing
from dataclasses import dataclass, field
from typing import Any, Callable

import httpx

PROTOCOL_VERSION = "2025-06-18"
CLIENT_INFO = {"name": "agentzones", "version": "0.1.0"}


@dataclass
class MCPTool:
    name: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=lambda: {"type": "object", "properties": {}})


class MCPError(RuntimeError):
    pass


def _result_text(result: dict[str, Any]) -> str:
    parts = []
    for item in result.get("content", []) or []:
        if item.get("type") == "text":
            parts.append(item.get("text", ""))
        else:
            parts.append(json.dumps(item))
    if not parts and "structuredContent" in result:
        parts.append(json.dumps(result["structuredContent"]))
    return "\n".join(parts)


# --------------------------------------------------------------------------- client


class _BaseClient:
    name: str
    tools: list[MCPTool]

    async def start(self) -> None: ...
    async def close(self) -> None: ...
    async def _request(self, method: str, params: dict | None = None) -> dict: ...

    async def list_tools(self) -> list[MCPTool]:
        res = await self._request("tools/list", {})
        self.tools = [
            MCPTool(t["name"], t.get("description", ""), t.get("inputSchema") or {"type": "object", "properties": {}})
            for t in res.get("tools", [])
        ]
        return self.tools

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> tuple[str, bool]:
        """Returns (text, is_error)."""
        res = await self._request("tools/call", {"name": name, "arguments": arguments or {}})
        return _result_text(res), bool(res.get("isError"))


class StdioMCPClient(_BaseClient):
    def __init__(self, name: str, command: str, args: list[str] | None = None, env: dict[str, str] | None = None):
        self.name, self.command, self.args = name, command, args or []
        self.env = {**os.environ, **(env or {})}
        self.proc: asyncio.subprocess.Process | None = None
        self._id = 0
        self._pending: dict[int, asyncio.Future] = {}
        self._reader: asyncio.Task | None = None
        self._lock = asyncio.Lock()
        self.tools = []

    async def start(self) -> None:
        cmd = sys.executable if self.command in ("python", "python3") else self.command
        self.proc = await asyncio.create_subprocess_exec(
            cmd, *self.args, env=self.env,
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=None,
            limit=16 * 1024 * 1024,
        )
        self._reader = asyncio.create_task(self._read_loop())
        await self._request("initialize", {
            "protocolVersion": PROTOCOL_VERSION, "capabilities": {}, "clientInfo": CLIENT_INFO,
        })
        await self._notify("notifications/initialized")
        await self.list_tools()

    async def _read_loop(self) -> None:
        assert self.proc and self.proc.stdout
        while True:
            line = await self.proc.stdout.readline()
            if not line:
                break
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue  # tolerate stray prints
            if "id" in msg and msg["id"] in self._pending and ("result" in msg or "error" in msg):
                fut = self._pending.pop(msg["id"])
                if not fut.done():
                    fut.set_result(msg)
            elif "id" in msg and "method" in msg:  # server->client request (e.g. ping)
                await self._send({"jsonrpc": "2.0", "id": msg["id"], "result": {}})
        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(MCPError(f"MCP server {self.name!r} exited"))

    async def _send(self, msg: dict) -> None:
        assert self.proc and self.proc.stdin
        async with self._lock:
            self.proc.stdin.write((json.dumps(msg) + "\n").encode())
            await self.proc.stdin.drain()

    async def _notify(self, method: str, params: dict | None = None) -> None:
        await self._send({"jsonrpc": "2.0", "method": method, **({"params": params} if params else {})})

    async def _request(self, method: str, params: dict | None = None, timeout: float = 120) -> dict:
        self._id += 1
        rid = self._id
        fut = asyncio.get_running_loop().create_future()
        self._pending[rid] = fut
        await self._send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params or {}})
        msg = await asyncio.wait_for(fut, timeout)
        if "error" in msg:
            raise MCPError(f"{self.name}.{method}: {msg['error'].get('message')}")
        return msg.get("result", {})

    async def close(self) -> None:
        if self.proc and self.proc.returncode is None:
            self.proc.terminate()
            try:
                await asyncio.wait_for(self.proc.wait(), 5)
            except asyncio.TimeoutError:
                self.proc.kill()
        if self._reader:
            self._reader.cancel()


class HttpMCPClient(_BaseClient):
    """Streamable-HTTP transport (single endpoint, JSON or SSE responses)."""

    def __init__(self, name: str, url: str, headers: dict[str, str] | None = None):
        self.name, self.url = name, url
        self.headers = {"Accept": "application/json, text/event-stream", **(headers or {})}
        self.http = httpx.AsyncClient(timeout=120)
        self._id = 0
        self.tools = []

    async def start(self) -> None:
        await self._request("initialize", {
            "protocolVersion": PROTOCOL_VERSION, "capabilities": {}, "clientInfo": CLIENT_INFO,
        })
        await self._post({"jsonrpc": "2.0", "method": "notifications/initialized"})
        await self.list_tools()

    async def _post(self, msg: dict) -> httpx.Response:
        r = await self.http.post(self.url, json=msg, headers=self.headers)
        sid = r.headers.get("mcp-session-id")
        if sid:
            self.headers["Mcp-Session-Id"] = sid
        return r

    async def _request(self, method: str, params: dict | None = None) -> dict:
        self._id += 1
        rid = self._id
        r = await self._post({"jsonrpc": "2.0", "id": rid, "method": method, "params": params or {}})
        r.raise_for_status()
        msgs: list[dict] = []
        if r.headers.get("content-type", "").startswith("text/event-stream"):
            for line in r.text.splitlines():
                if line.startswith("data:"):
                    try:
                        msgs.append(json.loads(line[5:].strip()))
                    except json.JSONDecodeError:
                        pass
        else:
            body = r.json()
            msgs = body if isinstance(body, list) else [body]
        for msg in msgs:
            if msg.get("id") == rid:
                if "error" in msg:
                    raise MCPError(f"{self.name}.{method}: {msg['error'].get('message')}")
                return msg.get("result", {})
        raise MCPError(f"{self.name}.{method}: no response")

    async def close(self) -> None:
        await self.http.aclose()


# --------------------------------------------------------------------------- server

_PY_TO_JSON = {str: "string", int: "integer", float: "number", bool: "boolean", dict: "object", list: "array"}


def _schema_for(fn: Callable) -> dict[str, Any]:
    sig = inspect.signature(fn)
    hints = typing.get_type_hints(fn)
    props, required = {}, []
    for pname, p in sig.parameters.items():
        t = hints.get(pname, str)
        origin = typing.get_origin(t)
        if origin is typing.Union or str(origin) == "<class 'types.UnionType'>":
            args = [a for a in typing.get_args(t) if a is not type(None)]
            t = args[0] if args else str
        t = typing.get_origin(t) or t
        prop: dict[str, Any] = {"type": _PY_TO_JSON.get(t, "string")}
        if p.default is inspect.Parameter.empty:
            required.append(pname)
        else:
            prop["default"] = p.default
        props[pname] = prop
    schema: dict[str, Any] = {"type": "object", "properties": props}
    if required:
        schema["required"] = required
    return schema


class MiniMCPServer:
    """Tiny stdio MCP server. Usage::

        srv = MiniMCPServer("proxy-doctor")

        @srv.tool()
        def check_env() -> dict:
            "Describe proxy environment variables."
            ...

        srv.run()
    """

    def __init__(self, name: str, version: str = "0.1.0", instructions: str = ""):
        self.name, self.version, self.instructions = name, version, instructions
        self._tools: dict[str, tuple[Callable, MCPTool]] = {}

    def tool(self, name: str | None = None, description: str | None = None):
        def deco(fn: Callable) -> Callable:
            tname = name or fn.__name__
            desc = description or inspect.getdoc(fn) or ""
            self._tools[tname] = (fn, MCPTool(tname, desc, _schema_for(fn)))
            return fn
        return deco

    # exposed for in-process use / tests
    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name not in self._tools:
            return {"content": [{"type": "text", "text": f"unknown tool {name}"}], "isError": True}
        fn, _ = self._tools[name]
        try:
            out = fn(**(arguments or {}))
            if inspect.isawaitable(out):
                out = asyncio.run(out)
            text = out if isinstance(out, str) else json.dumps(out, indent=2, default=str)
            return {"content": [{"type": "text", "text": text}], "isError": False}
        except Exception as exc:  # report tool errors to the model, don't crash
            return {"content": [{"type": "text", "text": f"{type(exc).__name__}: {exc}"}], "isError": True}

    def handle(self, msg: dict[str, Any]) -> dict[str, Any] | None:
        method, rid = msg.get("method"), msg.get("id")
        if rid is None:
            return None  # notification
        try:
            if method == "initialize":
                result = {
                    "protocolVersion": msg.get("params", {}).get("protocolVersion", PROTOCOL_VERSION),
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": self.name, "version": self.version},
                    **({"instructions": self.instructions} if self.instructions else {}),
                }
            elif method == "ping":
                result = {}
            elif method == "tools/list":
                result = {"tools": [
                    {"name": t.name, "description": t.description, "inputSchema": t.input_schema}
                    for _, t in self._tools.values()
                ]}
            elif method == "tools/call":
                p = msg.get("params", {})
                result = self.call(p.get("name", ""), p.get("arguments") or {})
            else:
                return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": f"method not found: {method}"}}
            return {"jsonrpc": "2.0", "id": rid, "result": result}
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32603, "message": str(exc)}}

    def run(self) -> None:
        # keep stdout clean for protocol traffic
        out = sys.stdout
        sys.stdout = sys.stderr
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            resp = self.handle(msg)
            if resp is not None:
                out.write(json.dumps(resp) + "\n")
                out.flush()
