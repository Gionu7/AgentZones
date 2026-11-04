"""MCP server for a zone: lets any MCP client (Claude Desktop, an IDE, an agent
framework) use the zone through its gateway.

Tools: ``list_agents(capability)`` and ``send_message(zone, agent, payload)``.
Transport: streamable HTTP on ``POST /mcp`` (JSON responses), port 8000,
published on 127.0.0.1 only. ``agentzone mcp --stdio`` serves the same tools
over stdio.
"""

from __future__ import annotations

import os
from typing import Any

import httpx
from fastapi import Body, FastAPI, Response
from fastapi.responses import JSONResponse

from .mcp_lite import MiniMCPServer


def build_server(gateway_url: str, token: str = "", name: str = "mcp") -> MiniMCPServer:
    headers = {"authorization": f"Bearer {token}"} if token else {}
    http = httpx.Client(base_url=gateway_url.rstrip("/"), headers=headers, timeout=310, trust_env=False)
    srv = MiniMCPServer("agent-zone", instructions="Talk to AI agents in this and other agent zones. "
                        "Use list_agents to find agents, then send_message to ask one.")

    @srv.tool()
    def list_agents(capability: str = "") -> list:
        """List agents in all zones. capability filters, e.g. 'proxy-doctor' or 'fl-client:*'."""
        r = http.get("/directory", params={"capability": capability} if capability else None)
        r.raise_for_status()
        return r.json()

    @srv.tool()
    def send_message(zone: str, agent: str, payload: dict, timeout: float = 60) -> Any:
        """Send payload to agent in zone and return the reply. Most agents accept {"type": "ask", "question": "..."};
        federated-learning aggregators accept {"type": "fl.status"}, {"type": "fl.start"}, {"type": "fl.stop"}."""
        r = http.post("/send", json={"from_agent": name, "target_zone": zone, "target_agent": agent,
                                     "payload": payload, "timeout": timeout}, timeout=timeout + 10)
        if r.status_code >= 400:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:500]}")
        return r.json()

    return srv


def http_app(srv: MiniMCPServer) -> FastAPI:
    app = FastAPI(title="agent zone MCP", docs_url=None, redoc_url=None, openapi_url=None)

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.post("/mcp")
    def mcp(body: Any = Body(None)):  # sync on purpose: the tools use a blocking client
        msgs = body if isinstance(body, list) else [body]
        out = [r for r in (srv.handle(m) for m in msgs if isinstance(m, dict)) if r is not None]
        if not out:
            return Response(status_code=202)
        return JSONResponse(out if isinstance(body, list) else out[0])

    return app


def main_from_env() -> MiniMCPServer:
    return build_server(os.environ.get("GATEWAY_URL", "http://gateway:8080"), os.environ.get("ZONE_API_TOKEN", ""),
                        os.environ.get("AGENT_NAME", "mcp"))
