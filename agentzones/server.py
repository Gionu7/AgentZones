"""HTTP API of a zone.

Local API (for users, CLI, dashboard)          Peer API (/p2p/*, HMAC-signed)
  GET  /                dashboard                POST /p2p/announce     gossip
  GET  /health                                   POST /p2p/ask          ask a public agent
  GET  /api/info        zone card + status       POST /p2p/tools/call   call an exposed tool
  GET  /api/peers                                POST /p2p/fixes        receive a verified fix
  GET  /api/tools                                GET  /p2p/fl/weights   trained FL snapshot
  POST /api/tools/call
  POST /api/agents/{name}/ask
  GET  /api/fixes
  GET  /api/events
  GET  /api/fl  POST /api/fl/{train|aggregate|round}

Peer calls must carry a valid HMAC signature when ``security.network_secret``
is set. Local API calls from non-loopback addresses must carry
``Authorization: Bearer <security.api_token>`` when it is set. See docs/SECURITY.md.
"""

from __future__ import annotations

import hmac
import json
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from .config import ZoneConfig
from .security import ip_allowed, verify
from .zone import Zone

STATIC = Path(__file__).parent / "static"


class AskBody(BaseModel):
    question: str
    context: dict[str, Any] | None = None


class ToolCallBody(BaseModel):
    tool: str
    arguments: dict[str, Any] = {}


def _is_loopback(ip: str | None) -> bool:
    return ip in ("127.0.0.1", "::1", "localhost", "testclient") or (ip or "").startswith("127.")


def create_app(cfg: ZoneConfig, zone: Zone | None = None) -> FastAPI:
    zone = zone or Zone(cfg)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await zone.start()
        try:
            yield
        finally:
            await zone.stop()

    app = FastAPI(title=f"Agent Zone {cfg.name}", lifespan=lifespan)
    app.state.zone = zone
    sec = cfg.security

    @app.middleware("http")
    async def guard(request: Request, call_next):
        path = request.url.path
        ip = request.client.host if request.client else None
        if path in ("/health",):
            return await call_next(request)
        if not ip_allowed(ip, sec.allowed_cidrs) and not _is_loopback(ip):
            return JSONResponse({"error": "source address not allowed"}, status_code=403)
        if path.startswith("/api/") and sec.api_token and not _is_loopback(ip):
            supplied = request.headers.get("authorization", "").removeprefix("Bearer ").strip()
            if not hmac.compare_digest(supplied, sec.api_token):
                return JSONResponse({"error": "unauthorized: api token required"}, status_code=401)
        if path.startswith("/p2p/") and sec.network_secret:
            body = await request.body()
            reason = verify(sec.network_secret, request.method, path, body, request.headers, sec.max_clock_skew_s)
            if reason:
                return JSONResponse({"error": f"unauthorized: {reason}"}, status_code=401)
        return await call_next(request)

    def caller(request: Request) -> str:
        return request.headers.get("x-az-zone", "unknown")

    # ---------------------------------------------------------------- local API
    @app.get("/")
    async def dashboard():
        return FileResponse(STATIC / "index.html")

    @app.get("/health")
    async def health():
        return {"ok": True, "zone": zone.name}

    @app.get("/api/info")
    async def info():
        return {**zone.card(), "uptime_s": round(time.time() - zone.started),
                "llm": await zone.llm.health(), "peers": len(zone.peers.alive()),
                "fixes": len(zone.fixes.fixes), "fl": zone.fl.status() if zone.fl else None}

    @app.get("/api/peers")
    async def peers():
        return [{"name": p.name, "url": p.url, "last_seen": p.last_seen, "via": p.via, "card": p.card}
                for p in zone.peers.alive()]

    @app.get("/api/tools")
    async def tools():
        return [{**t.spec(), "exposed": t.exposed, "server": t.server} for t in zone.tools.tools.values()]

    @app.post("/api/tools/call")
    async def call_tool(body: ToolCallBody):
        try:
            return {"result": await zone.tools.call(body.tool, body.arguments)}
        except KeyError as exc:
            raise HTTPException(404, str(exc))
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}

    @app.get("/api/agents")
    async def agents():
        return [a.card() for a in zone.agents.values()]

    @app.post("/api/agents/{name}/ask")
    async def ask(name: str, body: AskBody):
        if name not in zone.agents:
            raise HTTPException(404, f"no agent {name}")
        return await zone.ask(name, body.question, body.context)

    @app.get("/api/fixes")
    async def fixes():
        return zone.fixes.all()

    @app.get("/api/events")
    async def events(since: float = 0):
        return [e for e in zone.events if e["t"] > since]

    @app.get("/api/fl")
    async def fl_status():
        if not zone.fl:
            raise HTTPException(404, "federated learning not enabled")
        return zone.fl.status()

    @app.post("/api/fl/{phase}")
    async def fl_phase(phase: str):
        fn = {"train": zone.fl_train, "aggregate": zone.fl_aggregate, "round": zone.fl_round}.get(phase)
        if not fn:
            raise HTTPException(404, "phase must be train|aggregate|round")
        try:
            return await fn()
        except RuntimeError as exc:
            raise HTTPException(400, str(exc))

    # ---------------------------------------------------------------- peer API
    @app.post("/p2p/announce")
    async def announce(request: Request):
        card = json.loads(await request.body() or b"{}")
        return zone.on_announce(card, request.client.host if request.client else None)

    @app.post("/p2p/ask")
    async def p2p_ask(request: Request):
        return await zone.on_peer_ask(json.loads(await request.body() or b"{}"))

    @app.post("/p2p/tools/call")
    async def p2p_tool(request: Request):
        return await zone.on_peer_tool_call(json.loads(await request.body() or b"{}"), caller(request))

    @app.post("/p2p/fixes")
    async def p2p_fix(request: Request):
        return zone.on_peer_fix(json.loads(await request.body() or b"{}"), caller(request))

    @app.get("/p2p/fl/weights")
    async def p2p_weights():
        if not zone.fl:
            raise HTTPException(404, "federated learning not enabled")
        return zone.fl.shared()

    return app
