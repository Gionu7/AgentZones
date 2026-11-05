"""MCP server for a zone: lets any MCP client (Claude Desktop, an IDE, an agent
framework) use the zone through its gateway.

Tools:

* ``list_agents(capability)`` and ``send_message(zone, agent, payload)``
* federated learning, typed and narrow: ``fl_list_jobs()``, ``fl_status(job)``,
  ``fl_request_start(job, rounds, reason)`` and ``fl_request_stop(job, reason)``. The last two only
  *ask*: the aggregator keeps the request pending until an operator approves it (FL_APPROVAL), caps the
  rounds (FL_MAX_ROUNDS) and records it in its audit log. Jobs are the ones defined in the zone files;
  no plugin, data or hyperparameter can be chosen here.

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
        """List agents in all zones. capability filters, e.g. 'proxy-doctor' or 'fl-client:*'.
        Each agent's 'accepts' lists the message types it handles and what they contain."""
        r = http.get("/directory", params={"capability": capability} if capability else None)
        r.raise_for_status()
        return r.json()

    @srv.tool()
    def send_message(zone: str, agent: str, payload: dict, timeout: float = 60) -> Any:
        """Send payload to agent in zone and return the reply. Use a message type from the agent's 'accepts'
        (see list_agents), e.g. {"type": "ask", "question": "..."}. For federated learning use the fl_* tools instead."""
        r = http.post("/send", json={"from_agent": name, "target_zone": zone, "target_agent": agent,
                                     "payload": payload, "timeout": timeout}, timeout=timeout + 10)
        if r.status_code >= 400:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:500]}")
        return r.json()

    def _aggregator(job: str) -> tuple[str, str]:
        r = http.get("/directory", params={"capability": f"fl-aggregator:{job}"})
        r.raise_for_status()
        found = r.json()
        if not found:
            raise ValueError(f"no federated-learning job {job!r}; call fl_list_jobs")
        return found[0]["zone"], found[0]["agent"]

    def _fl(job: str, payload: dict) -> dict:
        zone, agent = _aggregator(job)
        r = http.post("/send", json={"from_agent": name, "target_zone": zone, "target_agent": agent,
                                     "payload": payload, "timeout": 20}, timeout=30)
        if r.status_code >= 400:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
        out = r.json()
        if isinstance(out, dict) and out.get("error"):
            raise RuntimeError(out["error"])
        return out

    def _summary(st: dict) -> dict:
        hist = st.get("history") or []
        last = {k: v for k, v in (hist[-1] if hist else {}).items() if k != "contrib"}
        return {"job": st.get("job"), "state": st.get("state"), "round": st.get("round"), "rounds": st.get("rounds"),
                "error": st.get("error"), "last_round": last, "pending_requests": st.get("pending"),
                "limits": st.get("limits")}

    @srv.tool()
    def fl_list_jobs() -> list:
        """List federated-learning jobs: name, where the aggregator runs, and the clients."""
        r = http.get("/directory")
        r.raise_for_status()
        entries = r.json()
        jobs = []
        for e in entries:
            for c in e["capabilities"]:
                if c.startswith("fl-aggregator:"):
                    job = c.split(":", 1)[1]
                    jobs.append({"job": job, "aggregator": f"{e['agent']}@{e['zone']}",
                                 "clients": [f"{x['agent']}@{x['zone']}" for x in entries
                                             if f"fl-client:{job}" in x["capabilities"]]})
        return jobs

    @srv.tool()
    def fl_status(job: str) -> dict:
        """State, current round, the latest round's metrics, pending requests and limits of a job."""
        return _summary(_fl(job, {"type": "fl.status"}))

    @srv.tool()
    def fl_request_start(job: str, rounds: int, reason: str) -> dict:
        """Ask to start a training run of `job` for `rounds` rounds. Explain why in `reason`.
        An operator must approve it; rounds above the job's limit are capped."""
        out = _fl(job, {"type": "fl.request", "action": "start", "rounds": rounds, "reason": reason})
        return {"pending": out.get("pending"), "message": out.get("message") or "started", **_summary(out)}

    @srv.tool()
    def fl_request_stop(job: str, reason: str) -> dict:
        """Ask to stop the running training run of `job`. Explain why in `reason`. An operator must approve it."""
        out = _fl(job, {"type": "fl.request", "action": "stop", "reason": reason})
        return {"pending": out.get("pending"), "message": out.get("message") or "stopped", **_summary(out)}

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
