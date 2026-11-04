"""Agents and the process that hosts one.

An agent is a small HTTP service inside a zone. It registers with its zone's
gateway every 10 s and receives messages on ``POST /message``; whatever
:meth:`Agent.handle` returns is the reply. To talk to other agents it only
uses its own gateway (``self.gw.directory`` / ``self.gw.send``), never another
zone directly.

Configuration is plain environment variables, so one Docker image serves every
agent (see zone/base.yml)::

    AGENT_NAME          name in the directory (default: the service host name)
    AGENT_IMPL          module:Class, a subclass of Agent
    AGENT_CAPABILITIES  comma separated, added to the class defaults
    GATEWAY_URL         http://gateway:8080
    AGENT_PORT / AGENT_URL   where the gateway reaches this agent
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import logging
import os
import socket
from dataclasses import dataclass
from typing import Any, Mapping

import httpx
from fastapi import FastAPI

log = logging.getLogger(__name__)


class GatewayError(RuntimeError):
    pass


@dataclass
class Incoming:
    id: str
    from_zone: str
    from_agent: str
    payload: Any
    local: bool  # sent from this agent's own zone

    @property
    def type(self) -> str | None:
        return self.payload.get("type") if isinstance(self.payload, dict) else None


class GatewayClient:
    """What an agent can do: look agents up and send them a message (request -> reply)."""

    def __init__(self, url: str, agent: str, token: str = ""):
        self.url, self.agent = url.rstrip("/"), agent
        headers = {"authorization": f"Bearer {token}"} if token else {}
        self.http = httpx.AsyncClient(base_url=self.url, timeout=30, headers=headers, trust_env=False)
        self.zone: str | None = None

    async def directory(self, capability: str | None = None) -> list[dict[str, Any]]:
        r = await self.http.get("/directory", params={"capability": capability} if capability else None)
        r.raise_for_status()
        return r.json()

    async def send(self, zone: str, agent: str, payload: Any, timeout: float = 60) -> Any:
        try:
            r = await self.http.post("/send", timeout=timeout + 10, json={
                "from_agent": self.agent, "target_zone": zone, "target_agent": agent,
                "payload": payload, "timeout": timeout})
        except httpx.HTTPError as exc:
            raise GatewayError(f"gateway {self.url}: {exc}") from exc
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail", r.text)
            except ValueError:
                detail = r.text
            raise GatewayError(f"send to {agent}@{zone}: HTTP {r.status_code}: {detail}")
        return r.json()

    async def register(self, url: str, capabilities: list[str], description: str) -> None:
        r = await self.http.post("/agents/register", json={
            "name": self.agent, "url": url, "capabilities": capabilities, "description": description})
        r.raise_for_status()
        self.zone = r.json().get("zone")

    async def close(self) -> None:
        await self.http.aclose()


class Agent:
    """Subclass and implement :meth:`handle`. ``self.config`` is the environment."""

    capabilities: list[str] = []
    description: str = ""

    def __init__(self, name: str, gw: GatewayClient, config: Mapping[str, str]):
        self.name, self.gw, self.config = name, gw, dict(config)

    @property
    def zone(self) -> str:
        return self.gw.zone or self.config.get("ZONE", "")

    def all_capabilities(self) -> list[str]:
        extra = [c.strip() for c in self.config.get("AGENT_CAPABILITIES", "").split(",") if c.strip()]
        return list(dict.fromkeys(self.capabilities + extra))

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def handle(self, msg: Incoming) -> Any:
        raise NotImplementedError

    def status(self) -> dict[str, Any]:
        return {}


class EchoAgent(Agent):
    """Replies with what it got. Handy for checking a zone's wiring."""

    capabilities = ["echo"]
    description = "Echoes the payload back"

    async def handle(self, msg: Incoming) -> Any:
        return {"echo": msg.payload, "from": f"{msg.from_agent}@{msg.from_zone}", "zone": self.zone}


def load_agent(impl: str, name: str, gw: GatewayClient, config: Mapping[str, str]) -> Agent:
    mod, _, cls = impl.partition(":")
    klass = getattr(importlib.import_module(mod), cls)
    if not (isinstance(klass, type) and issubclass(klass, Agent)):
        raise TypeError(f"{impl} is not an agentzones.agent.Agent")
    return klass(name, gw, config)


class AgentHost:
    """Runs one agent: HTTP endpoint for the gateway plus the registration loop."""

    def __init__(self, config: Mapping[str, str] | None = None, agent: Agent | None = None):
        cfg = dict(os.environ if config is None else config)
        self.config = cfg
        self.name = cfg.get("AGENT_NAME") or socket.gethostname()
        self.port = int(cfg.get("AGENT_PORT", "9000"))
        self.url = (cfg.get("AGENT_URL") or f"http://{socket.gethostname()}:{self.port}").rstrip("/")
        self.register_interval = float(cfg.get("REGISTER_INTERVAL_S", "10"))
        gw = GatewayClient(cfg.get("GATEWAY_URL", "http://gateway:8080"), self.name, cfg.get("ZONE_API_TOKEN", ""))
        self.agent = agent or load_agent(cfg.get("AGENT_IMPL", "agentzones.agent:EchoAgent"), self.name, gw, cfg)
        self._task: asyncio.Task | None = None

    async def _register_loop(self) -> None:
        ok = False
        while True:
            try:
                await self.agent.gw.register(self.url, self.agent.all_capabilities(), self.agent.description)
                if not ok:
                    log.info("agent %s registered with %s (zone %s)", self.name, self.agent.gw.url, self.agent.zone)
                ok = True
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if ok or self._task is None:
                    log.warning("agent %s: gateway %s unreachable: %s", self.name, self.agent.gw.url, exc)
                ok = False
            await asyncio.sleep(self.register_interval)

    def app(self) -> FastAPI:
        @contextlib.asynccontextmanager
        async def lifespan(_app):
            await self.agent.start()
            self._task = asyncio.create_task(self._register_loop())
            try:
                yield
            finally:
                self._task.cancel()
                await self.agent.stop()
                await self.agent.gw.close()

        app = FastAPI(title=f"agent {self.name}", lifespan=lifespan, docs_url=None, redoc_url=None)
        app.state.host = self

        @app.get("/health")
        async def health():
            return {"ok": True, "agent": self.name, "zone": self.agent.zone}

        @app.post("/message")
        async def message(body: dict[str, Any]):
            zone = str(body.get("from_zone") or "")
            msg = Incoming(str(body.get("id") or ""), zone, str(body.get("from_agent") or ""), body.get("payload"),
                           local=bool(zone) and zone == self.agent.zone)
            try:
                return await self.agent.handle(msg)
            except Exception as exc:
                log.exception("agent %s failed on a message", self.name)
                return {"error": f"{type(exc).__name__}: {exc}"}

        return app

    async def serve(self, host: str = "0.0.0.0") -> None:
        import uvicorn

        await uvicorn.Server(uvicorn.Config(self.app(), host=host, port=self.port, log_level="warning")).serve()
