"""The zone gateway: the only component an agent talks to.

Local API (zone network only, published on 127.0.0.1)::

    POST /agents/register  {name, url, capabilities, description}   agents call this every 10 s
    GET  /directory?capability=fl-client:*                          agents in all zones
    POST /send  {target_zone, target_agent, payload, timeout?}      request -> reply payload
    GET  /status, /health

Hub API (role ``hub`` only, a separate port that spokes reach)::

    POST /hub/register  GET /hub/poll?wait=20  POST /hub/ack  POST /hub/send  GET /hub/directory

Every hub call is HMAC-signed with the spoke's own token (see security.py).

A spoke only ever opens outbound connections: it long-polls the hub for its
mailbox, delivers messages to local agents and posts the replies back.
"""

from __future__ import annotations

import asyncio
import contextlib
import fnmatch
import json
import logging
import os
import socket
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response

from . import __version__
from .hub import Hub, Message, RoutingError
from .security import RequestVerifier, parse_tokens, sign_request, sign_response, verify_response

log = logging.getLogger(__name__)

DEV_TOKEN_PREFIX = "dev-token-"


@dataclass
class GatewaySettings:
    zone: str
    role: Literal["hub", "spoke", "standalone"] = "standalone"
    host: str = "0.0.0.0"
    port: int = 8080                     # local API
    hub_host: str = "0.0.0.0"
    hub_port: int = 9090                 # hub API (role hub)
    spoke_tokens: dict[str, str] = field(default_factory=dict)  # hub: zone -> token
    hub_url: str = ""                    # spoke
    hub_token: str = ""                  # spoke
    hub_proxy: str = ""                  # spoke: HTTP proxy for hub traffic
    api_token: str = ""                  # optional bearer token for the local API
    poll_wait_s: float = 20.0
    agent_ttl_s: float = 30.0
    register_interval_s: float = 10.0
    visibility_s: float = 30.0
    default_timeout_s: float = 60.0
    max_timeout_s: float = 3600.0

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "GatewaySettings":
        e = dict(os.environ if env is None else env)
        g = e.get
        return cls(
            zone=g("ZONE") or socket.gethostname(), role=(g("ROLE") or "standalone").lower(),  # type: ignore[arg-type]
            host=g("GATEWAY_HOST", "0.0.0.0"), port=int(g("GATEWAY_PORT", "8080")),
            hub_host=g("HUB_HOST", "0.0.0.0"), hub_port=int(g("HUB_PORT", "9090")),
            spoke_tokens=parse_tokens(g("SPOKE_TOKENS", "")), hub_url=g("HUB_URL", "").rstrip("/"),
            hub_token=g("HUB_TOKEN", ""), hub_proxy=g("HUB_PROXY", ""), api_token=g("ZONE_API_TOKEN", ""),
            poll_wait_s=float(g("POLL_WAIT_S", "20")), visibility_s=float(g("ACK_TIMEOUT_S", "30")),
            default_timeout_s=float(g("SEND_TIMEOUT_S", "60")),
        )

    def check(self) -> list[str]:
        """Return warnings; raise on settings that cannot work."""
        if self.role not in ("hub", "spoke", "standalone"):
            raise ValueError(f"ROLE must be hub, spoke or standalone, not {self.role!r}")
        warnings = []
        if self.role == "spoke":
            if not (self.hub_url and self.hub_token):
                raise ValueError("a spoke needs HUB_URL and HUB_TOKEN")
            if self.hub_token.startswith(DEV_TOKEN_PREFIX):
                warnings.append("HUB_TOKEN is a dev-token-* default; set a random token per spoke")
            if self.hub_url.startswith("http://"):
                warnings.append("hub traffic is plain HTTP: requests are signed, but anyone on the path can read them")
        if self.role == "hub":
            if not self.spoke_tokens:
                warnings.append("SPOKE_TOKENS is empty: no spoke can connect")
            if any(t.startswith(DEV_TOKEN_PREFIX) for t in self.spoke_tokens.values()):
                warnings.append("SPOKE_TOKENS contains dev-token-* defaults; replace them")
            if len(set(self.spoke_tokens.values())) < len(self.spoke_tokens):
                warnings.append("two spokes share a token; use a different token for each spoke")
        return warnings


class HubClient:
    """Signed requests from a spoke to its hub, optionally through an HTTP proxy."""

    def __init__(self, s: GatewaySettings):
        self.s = s
        self.http = httpx.AsyncClient(base_url=s.hub_url, proxy=s.hub_proxy or None, trust_env=False,
                                      timeout=httpx.Timeout(s.poll_wait_s + 15, connect=10))

    async def call(self, method: str, path: str, body: Any = None) -> Any:
        raw = b"" if body is None else json.dumps(body, separators=(",", ":")).encode()
        headers = sign_request(self.s.hub_token, self.s.zone, method, path, raw)
        if raw:
            headers["content-type"] = "application/json"
        r = await self.http.request(method, path, content=raw, headers=headers)
        if r.status_code == 401:
            raise PermissionError(f"hub rejected zone {self.s.zone}: {r.text[:200]}")
        if not verify_response(self.s.hub_token, headers["X-AZ-Nonce"], r.content, r.headers.get("x-az-signature")):
            raise PermissionError(f"hub response to {path} is not signed with our token (HTTP {r.status_code})")
        data = r.json()
        if r.status_code == 403:
            raise RoutingError(str(data.get("detail", data)))
        if r.status_code >= 400:
            raise RuntimeError(f"hub {path}: HTTP {r.status_code}: {data.get('detail', data)}")
        return data

    async def close(self) -> None:
        await self.http.aclose()


class Gateway:
    def __init__(self, s: GatewaySettings):
        self.s = s
        self.zone = s.zone
        self.agents: dict[str, dict[str, Any]] = {}
        self.pending: dict[str, asyncio.Future] = {}
        self.events: list[dict[str, Any]] = []
        self.http = httpx.AsyncClient(timeout=s.default_timeout_s, trust_env=False)  # zone-local only
        self.hub: Hub | None = Hub(s.zone, list(s.spoke_tokens), self.deliver_local, s.visibility_s) \
            if s.role == "hub" else None
        self.verifier = RequestVerifier(s.spoke_tokens) if s.role == "hub" else None
        self.hub_client = HubClient(s) if s.role == "spoke" else None
        self.hub_state: dict[str, Any] = {"connected": False, "last_ok": None, "last_error": None}
        self._remote_dir: tuple[float, list[dict[str, Any]]] = (0.0, [])
        self._seen: dict[str, float] = {}
        self._tasks: list[asyncio.Task] = []
        self.started = time.time()

    # ------------------------------------------------------------------ lifecycle
    async def start(self) -> None:
        for w in self.s.check():
            log.warning("%s", w)
        if self.hub_client:
            self._tasks += [asyncio.create_task(self._poll_loop()), asyncio.create_task(self._register_loop())]
        self.event("gateway", f"zone {self.zone} started as {self.s.role}")

    async def stop(self) -> None:
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        await self.http.aclose()
        if self.hub_client:
            await self.hub_client.close()

    def event(self, kind: str, msg: str) -> None:
        self.events.append({"t": round(time.time(), 3), "kind": kind, "msg": msg})
        del self.events[:-200]
        log.info("[%s] %s", kind, msg)

    # ------------------------------------------------------------------ local agents
    def register_agent(self, name: str, url: str, capabilities: list[str], description: str = "") -> None:
        if name not in self.agents:
            self.event("agent", f"agent {name} registered ({', '.join(capabilities) or 'no capabilities'})")
        self.agents[name] = {"url": url.rstrip("/"), "capabilities": capabilities,
                             "description": description, "seen": time.time()}

    def live_agents(self) -> dict[str, dict[str, Any]]:
        now = time.time()
        return {n: a for n, a in self.agents.items() if now - a["seen"] <= self.s.agent_ttl_s}

    def local_entries(self) -> list[dict[str, Any]]:
        return [{"zone": self.zone, "agent": n, "capabilities": a["capabilities"], "description": a["description"]}
                for n, a in sorted(self.live_agents().items())]

    async def directory(self, capability: str | None = None) -> list[dict[str, Any]]:
        entries = self.local_entries()
        if self.hub:
            entries += self.hub.remote_entries()
        elif self.hub_client:
            t, cached = self._remote_dir
            if time.time() - t > 2:
                try:
                    cached = [e for e in await self.hub_client.call("GET", "/hub/directory") if e["zone"] != self.zone]
                    self._remote_dir = (time.time(), cached)
                except Exception as exc:
                    log.warning("hub directory unavailable: %s", exc)
            entries += cached
        if capability:
            entries = [e for e in entries if any(fnmatch.fnmatchcase(c, capability) for c in e["capabilities"])]
        return entries

    # ------------------------------------------------------------------ sending
    async def send(self, from_agent: str, target_zone: str, target_agent: str, payload: Any,
                   timeout: float | None = None) -> Any:
        timeout = min(float(timeout or self.s.default_timeout_s), self.s.max_timeout_s)
        msg: Message = {"id": uuid.uuid4().hex, "kind": "request", "from_zone": self.zone,
                        "from_agent": from_agent, "to_zone": target_zone or self.zone, "to_agent": target_agent,
                        "payload": payload, "expires": time.time() + timeout}
        fut = asyncio.get_running_loop().create_future()
        self.pending[msg["id"]] = fut
        try:
            await self.route(msg)
            return await asyncio.wait_for(fut, timeout)
        except asyncio.TimeoutError:
            raise TimeoutError(f"no reply from {target_agent}@{target_zone} within {timeout:.0f}s") from None
        finally:
            self.pending.pop(msg["id"], None)

    async def route(self, msg: Message) -> None:
        if msg["to_zone"] == self.zone:
            await self.deliver_local(msg)
        elif self.hub:
            await self.hub.accept(msg, self.zone)
        elif self.hub_client:
            for attempt in range(4):
                try:
                    await self.hub_client.call("POST", "/hub/send", msg)
                    return
                except (httpx.TransportError, PermissionError) as exc:
                    if attempt == 3:
                        raise ConnectionError(f"hub unreachable: {exc}") from exc
                    await asyncio.sleep(0.5 * 2 ** attempt)
        else:
            raise LookupError(f"zone {msg['to_zone']!r} is not reachable from a standalone zone")

    async def deliver_local(self, msg: Message) -> None:
        if msg.get("kind") == "reply":
            fut = self.pending.get(str(msg.get("reply_to")))
            if fut and not fut.done():
                fut.set_result(msg.get("payload"))
            return
        asyncio.create_task(self._handle_request(msg))

    async def _handle_request(self, msg: Message) -> None:
        reply_payload = await self._call_agent(msg)
        reply: Message = {"id": uuid.uuid4().hex, "kind": "reply", "reply_to": msg["id"], "from_zone": self.zone,
                          "from_agent": msg.get("to_agent"), "to_zone": msg["from_zone"],
                          "to_agent": msg.get("from_agent"), "payload": reply_payload,
                          "expires": max(float(msg.get("expires") or 0), time.time() + 30)}
        try:
            await self.route(reply)
        except Exception as exc:
            log.warning("could not return reply for %s: %s", msg["id"], exc)

    async def _call_agent(self, msg: Message) -> Any:
        agent = self.live_agents().get(str(msg.get("to_agent")))
        if agent is None:
            return {"error": f"no agent {msg.get('to_agent')!r} in zone {self.zone}"}
        if msg["from_zone"] != self.zone:
            ptype = msg.get("payload", {}).get("type") if isinstance(msg.get("payload"), dict) else None
            self.event("message", f"{msg.get('from_agent')}@{msg['from_zone']} -> {msg['to_agent']}"
                                  + (f" ({ptype})" if ptype else ""))
        timeout = max(float(msg.get("expires") or 0) - time.time(), 1.0)
        try:
            r = await self.http.post(agent["url"] + "/message", timeout=timeout, json={
                k: msg.get(k) for k in ("id", "from_zone", "from_agent", "to_agent", "payload", "expires")})
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            return {"error": f"agent {msg.get('to_agent')} failed: {type(exc).__name__}: {exc}"}

    # ------------------------------------------------------------------ spoke loops
    def _first_time(self, mid: str) -> bool:
        now = time.time()
        if len(self._seen) > 5000:
            self._seen = {k: t for k, t in self._seen.items() if now - t < 900}
        if mid in self._seen:
            return False
        self._seen[mid] = now
        return True

    async def _poll_loop(self) -> None:
        assert self.hub_client
        backoff = 1.0
        while True:
            try:
                msgs = await self.hub_client.call("GET", f"/hub/poll?wait={self.s.poll_wait_s:g}")
                self._hub_ok()
                backoff = 1.0
                if msgs:
                    await self.hub_client.call("POST", "/hub/ack", {"ids": [m["id"] for m in msgs]})
                for m in msgs:
                    if self._first_time(m["id"]):
                        await self.deliver_local(m)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._hub_error(exc)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30)

    async def _register_loop(self) -> None:
        assert self.hub_client
        while True:
            try:
                await self.hub_client.call("POST", "/hub/register", {"agents": self.local_entries()})
                self._hub_ok()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._hub_error(exc)
            await asyncio.sleep(self.s.register_interval_s)

    def _hub_ok(self) -> None:
        if not self.hub_state["connected"]:
            self.event("hub", f"connected to hub {self.s.hub_url}" + (f" via {self.s.hub_proxy}" if self.s.hub_proxy else ""))
        self.hub_state.update(connected=True, last_ok=time.time())

    def _hub_error(self, exc: Exception) -> None:
        if self.hub_state["connected"] or self.hub_state["last_error"] is None:
            self.event("hub", f"hub unreachable: {type(exc).__name__}: {exc}")
        self.hub_state.update(connected=False, last_error=f"{type(exc).__name__}: {exc}")

    def status(self) -> dict[str, Any]:
        out: dict[str, Any] = {"zone": self.zone, "role": self.s.role, "version": __version__,
                               "uptime_s": round(time.time() - self.started), "agents": self.local_entries(),
                               "events": self.events[-30:]}
        if self.hub:
            out["spokes"] = self.hub.zones()
        if self.hub_client:
            out["hub"] = {"url": self.s.hub_url, "proxy": self.s.hub_proxy or None, **self.hub_state}
        return out


# ---------------------------------------------------------------------- HTTP apps

def local_app(gw: Gateway) -> FastAPI:
    @contextlib.asynccontextmanager
    async def lifespan(_app):
        await gw.start()
        try:
            yield
        finally:
            await gw.stop()

    app = FastAPI(title=f"agent zone {gw.zone} gateway", version=__version__, lifespan=lifespan)
    app.state.gateway = gw

    @app.middleware("http")
    async def _auth(request: Request, call_next):
        if gw.s.api_token and request.url.path != "/health":
            if request.headers.get("authorization") != f"Bearer {gw.s.api_token}":
                return JSONResponse({"detail": "missing or bad bearer token"}, status_code=401)
        return await call_next(request)

    @app.get("/health")
    async def health():
        return {"ok": True, "zone": gw.zone, "role": gw.s.role}

    @app.get("/status")
    async def status():
        return gw.status()

    @app.post("/agents/register")
    async def register(body: dict[str, Any]):
        name, url = body.get("name"), body.get("url")
        if not name or not url:
            raise HTTPException(400, "name and url are required")
        gw.register_agent(str(name), str(url), [str(c) for c in body.get("capabilities") or []],
                          str(body.get("description") or ""))
        return {"ok": True, "zone": gw.zone, "ttl_s": gw.s.agent_ttl_s}

    @app.get("/directory")
    async def directory(capability: str | None = None):
        return await gw.directory(capability)

    @app.post("/send")
    async def send(body: dict[str, Any]):
        if not body.get("target_agent"):
            raise HTTPException(400, "target_agent is required")
        try:
            return await gw.send(str(body.get("from_agent") or "operator"), str(body.get("target_zone") or gw.zone),
                                 str(body["target_agent"]), body.get("payload"), body.get("timeout"))
        except TimeoutError as exc:
            raise HTTPException(504, str(exc))
        except (RoutingError, LookupError) as exc:
            raise HTTPException(404, str(exc))
        except ConnectionError as exc:
            raise HTTPException(502, str(exc))
        except RuntimeError as exc:
            raise HTTPException(502, str(exc))

    return app


def hub_app(gw: Gateway) -> FastAPI:
    """Spoke-facing API. Every request is verified, every response is signed."""
    assert gw.hub and gw.verifier
    hub, verifier = gw.hub, gw.verifier
    app = FastAPI(title=f"agent zone hub {gw.zone}", version=__version__, docs_url=None, redoc_url=None,
                  openapi_url=None)

    def signed(zone: str, nonce: str, data: Any, status: int = 200) -> Response:
        raw = json.dumps(data, separators=(",", ":")).encode()
        return Response(raw, status_code=status, media_type="application/json",
                        headers={"X-AZ-Signature": sign_response(gw.s.spoke_tokens[zone], nonce, raw)})

    async def authed(request: Request) -> tuple[str, str, bytes]:
        raw = await request.body()
        path = request.url.path + (f"?{request.url.query}" if request.url.query else "")
        zone, reason = verifier.verify(request.method, path, raw, request.headers)
        if not zone:
            gw.event("security", f"rejected {request.method} {request.url.path} from "
                                 f"{request.client.host if request.client else '?'}: {reason}")
            raise HTTPException(401, reason)
        return zone, request.headers["x-az-nonce"], raw

    @app.get("/health")
    async def health():
        return {"ok": True}

    @app.post("/hub/register")
    async def register(request: Request):
        zone, nonce, raw = await authed(request)
        known = zone in hub.directory
        hub.register(zone, (json.loads(raw or b"{}").get("agents") or []))
        if not known:
            gw.event("hub", f"spoke {zone} connected")
        return signed(zone, nonce, {"ok": True})

    @app.get("/hub/directory")
    async def directory(request: Request):
        zone, nonce, _ = await authed(request)
        return signed(zone, nonce, gw.local_entries() + hub.remote_entries())

    @app.get("/hub/poll")
    async def poll(request: Request, wait: float = 20):
        zone, nonce, _ = await authed(request)
        return signed(zone, nonce, await hub.poll(zone, min(max(wait, 0), 55)))

    @app.post("/hub/ack")
    async def ack(request: Request):
        zone, nonce, raw = await authed(request)
        return signed(zone, nonce, {"acked": hub.ack(zone, list(json.loads(raw or b"{}").get("ids") or []))})

    @app.post("/hub/send")
    async def send(request: Request):
        zone, nonce, raw = await authed(request)
        try:
            msg = json.loads(raw)
            if not isinstance(msg, dict) or not msg.get("id"):
                raise ValueError("message needs an id")
            await hub.accept(msg, zone)
        except (RoutingError, ValueError) as exc:
            return signed(zone, nonce, {"detail": str(exc)}, 403 if isinstance(exc, RoutingError) else 400)
        return signed(zone, nonce, {"ok": True})

    return app


async def serve(s: GatewaySettings) -> None:
    """Run the gateway: local API, plus the hub API when role is hub."""
    import uvicorn

    gw = Gateway(s)
    servers = [uvicorn.Server(uvicorn.Config(local_app(gw), host=s.host, port=s.port, log_level="warning"))]
    if gw.hub:
        servers.append(uvicorn.Server(uvicorn.Config(hub_app(gw), host=s.hub_host, port=s.hub_port,
                                                     log_level="warning")))
    log.info("zone %s (%s): local API on %s:%d%s", s.zone, s.role, s.host, s.port,
             f", hub API on {s.hub_host}:{s.hub_port}" if gw.hub else "")
    await asyncio.gather(*(srv.serve() for srv in servers))
