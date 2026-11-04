"""Federated learning client agent.

Advertises ``fl-client:<FL_JOB>`` and answers:

* ``fl.fit``      {parameters, round, config} -> {status: ok, parameters, num_samples, metrics}
                  or {status: busy} while a fit is already running (delivery is at-least-once)
* ``fl.evaluate`` {parameters, config}        -> {status: ok, num_samples, metrics}
* ``fl.model``    {parameters, round}         -> the final global model; saved to FL_MODEL_OUT if set
* ``fl.status``

Environment: FL_JOB, FL_TASK (module:Class, a ClientTask), FL_MODEL_OUT (optional .npz path).
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..agent import Agent, Incoming
from . import ClientTask, decode, encode, load_plugin

log = logging.getLogger(__name__)


class FLClient(Agent):
    description = "Federated learning client"

    def __init__(self, *a: Any, **kw: Any):
        super().__init__(*a, **kw)
        self.job = self.config.get("FL_JOB", "default")
        self.capabilities = [f"fl-client:{self.job}"]
        self.description = f"Federated learning client for job {self.job}"
        self.task: ClientTask | None = None
        self.lock = asyncio.Lock()
        self.history: list[dict[str, Any]] = []

    async def start(self) -> None:
        spec = self.config.get("FL_TASK")
        if not spec:
            raise ValueError("FL_TASK is required (module:Class of a ClientTask)")
        # building the task may load or download data: keep the event loop free
        self.task = await asyncio.to_thread(load_plugin, spec, self.config)

    async def handle(self, msg: Incoming) -> Any:
        p = msg.payload if isinstance(msg.payload, dict) else {}
        t = msg.type
        if t == "fl.status":
            return self.status()
        if t not in ("fl.fit", "fl.evaluate", "fl.model"):
            return {"error": f"unknown message type {t!r}", "accepts": ["fl.fit", "fl.evaluate", "fl.model", "fl.status"]}
        if p.get("job", self.job) != self.job:
            return {"status": "error", "error": f"this client serves job {self.job!r}"}
        if self.lock.locked():
            return {"status": "busy"}
        async with self.lock:
            assert self.task
            params = decode(p["parameters"])
            if t == "fl.fit":
                t0 = time.time()
                new, n, metrics = await asyncio.to_thread(self.task.fit, params, p.get("config") or {})
                rec = {"round": p.get("round"), "num_samples": n, "seconds": round(time.time() - t0, 2),
                       **{k: round(float(v), 4) for k, v in metrics.items()}}
                self.history.append(rec)
                del self.history[:-100]
                log.info("fl.fit round %s from %s@%s: %s", p.get("round"), msg.from_agent, msg.from_zone, rec)
                return {"status": "ok", "parameters": encode(new), "num_samples": int(n), "metrics": metrics}
            if t == "fl.evaluate":
                n, metrics = await asyncio.to_thread(self.task.evaluate, params, p.get("config") or {})
                return {"status": "ok", "num_samples": int(n), "metrics": metrics}
            await asyncio.to_thread(self.task.on_global, params)
            out = self.config.get("FL_MODEL_OUT")
            if out:
                Path(out).parent.mkdir(parents=True, exist_ok=True)
                tmp = Path(out).with_suffix(".tmp.npz")
                np.savez_compressed(tmp, **params)
                tmp.replace(out)
            return {"status": "ok", "saved": bool(out)}

    def status(self) -> dict[str, Any]:
        return {"job": self.job, "task": self.config.get("FL_TASK"), "busy": self.lock.locked(),
                "history": self.history[-10:]}
