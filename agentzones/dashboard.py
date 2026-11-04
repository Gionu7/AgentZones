"""Zone dashboard: ``GET /`` (the page) and ``GET /dashboard/data`` (one JSON snapshot).

Served by the gateway's local API, so like that API it is reachable on
127.0.0.1 only. It works on any zone; on the hub it shows the whole network:
spokes, the directory, every message that crossed the hub (type and size, never
contents), federated-learning jobs and Proxy Doctor fixes.

The gateway collects the data itself, with a short cache, so any number of
browser tabs cost the same few messages. Those messages come from the agent
name ``dashboard``; the page hides them from the traffic view by default.
"""

from __future__ import annotations

import asyncio
import time
from importlib import resources
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

if TYPE_CHECKING:
    from .gateway import Gateway

AGENT = "dashboard"


class Dashboard:
    def __init__(self, gw: "Gateway", fl_every_s: float = 1.5, doctors_every_s: float = 8.0):
        self.gw = gw
        self.fl_every_s, self.doctors_every_s = fl_every_s, doctors_every_s
        self._fl: tuple[float, list[dict[str, Any]]] = (0.0, [])
        self._doctors: tuple[float, list[dict[str, Any]]] = (0.0, [])
        self._lock = asyncio.Lock()

    async def _ask(self, entry: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
        try:
            r = await self.gw.send(AGENT, entry["zone"], entry["agent"], payload, timeout=5)
            return r if isinstance(r, dict) else {"error": str(r)}
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}

    async def _fl_jobs(self, directory: list[dict[str, Any]]) -> list[dict[str, Any]]:
        aggs = [e for e in directory if any(c.startswith("fl-aggregator:") for c in e["capabilities"])]
        replies = await asyncio.gather(*(self._ask(e, {"type": "fl.status"}) for e in aggs))
        jobs = []
        for e, r in zip(aggs, replies):
            job = next(c.split(":", 1)[1] for c in e["capabilities"] if c.startswith("fl-aggregator:"))
            clients = [f"{c['agent']}@{c['zone']}" for c in directory if f"fl-client:{job}" in c["capabilities"]]
            jobs.append({"zone": e["zone"], "agent": e["agent"], "job": job, "clients": clients,
                         "status": r if "state" in r else None, "error": r.get("error") if "state" not in r else None})
        return jobs

    async def _doctor_status(self, directory: list[dict[str, Any]]) -> list[dict[str, Any]]:
        docs = [e for e in directory if "proxy-doctor" in e["capabilities"]]
        replies = await asyncio.gather(*(self._ask(e, {"type": "status"}) for e in docs))
        return [{"zone": e["zone"], "agent": e["agent"], "fixes": r.get("fixes"), "model": r.get("model"),
                 "error": r.get("error")} for e, r in zip(docs, replies)]

    async def snapshot(self) -> dict[str, Any]:
        gw = self.gw
        async with self._lock:  # one refresh at a time, whatever the number of tabs
            directory = await gw.directory()
            now = time.time()
            if now - self._fl[0] > self.fl_every_s:
                self._fl = (now, await self._fl_jobs(directory))
            if now - self._doctors[0] > self.doctors_every_s:
                self._doctors = (now, await self._doctor_status(directory))
        out: dict[str, Any] = {"t": time.time(), **gw.status(), "directory": directory, "fl": self._fl[1],
                               "doctors": self._doctors[1], "events": gw.events[-80:]}
        out["poll_wait_s"] = gw.s.poll_wait_s
        if gw.hub:
            out["traffic"] = list(gw.hub.traffic)[-150:]
            out["totals"] = gw.hub.totals
            out["rate"] = [r for r in gw.hub.rate if r[0] >= int(time.time()) - 120]  # [sec, fl, pd, other, bytes]
        return out


def add_dashboard(app: FastAPI, gw: "Gateway") -> None:
    dash = Dashboard(gw)
    page = resources.files("agentzones").joinpath("static/dashboard.html").read_text(encoding="utf-8")

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def index():
        return HTMLResponse(page, headers={"Cache-Control": "no-store"})

    @app.get("/dashboard/data")
    async def data():
        return await dash.snapshot()
