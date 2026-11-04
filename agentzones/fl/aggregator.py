"""Federated learning aggregator agent.

Lifecycle of a run::

    idle -> waiting_for_clients -> running -> done | failed | stopped

Each round the aggregator looks up ``fl-client:<FL_JOB>`` in the directory
again, so clients may join or leave between rounds. It sends ``fl.fit`` to all
of them in parallel; the round counts if at least ``FL_MIN_FIT_CLIENTS``
reply in time. A round that falls short is retried (``FL_MAX_ROUND_RETRIES``)
and then fails the run.

Control messages (from any zone unless ``FL_CONTROL_ZONES`` restricts it):
``fl.start`` {rounds?, resume?}, ``fl.stop``, ``fl.status``.

Environment: FL_JOB, FL_STRATEGY (module:Class), FL_ROUNDS (10), FL_MIN_CLIENTS (2),
FL_MIN_FIT_CLIENTS (= FL_MIN_CLIENTS), FL_ROUND_TIMEOUT_S (600), FL_WAIT_TIMEOUT_S (0 = forever),
FL_AUTOSTART (true), FL_MODEL_DIR (/models), FL_MAX_ROUND_RETRIES (3).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..agent import Agent, GatewayError, Incoming
from . import FitResult, Parameters, Strategy, check_shapes, decode, encode, load_plugin

log = logging.getLogger(__name__)


def _bool(v: str | None, default: bool) -> bool:
    return default if v is None or v == "" else v.strip().lower() in ("1", "true", "yes", "on")


class FLAggregator(Agent):
    def __init__(self, *a: Any, **kw: Any):
        super().__init__(*a, **kw)
        c = self.config
        self.job = c.get("FL_JOB", "default")
        self.capabilities = [f"fl-aggregator:{self.job}"]
        self.description = f"Federated learning aggregator for job {self.job}"
        self.rounds = int(c.get("FL_ROUNDS", "10"))
        self.min_clients = int(c.get("FL_MIN_CLIENTS", "2"))
        self.min_fit = int(c.get("FL_MIN_FIT_CLIENTS") or self.min_clients)
        self.round_timeout = float(c.get("FL_ROUND_TIMEOUT_S", "600"))
        self.wait_timeout = float(c.get("FL_WAIT_TIMEOUT_S", "0"))
        self.max_retries = int(c.get("FL_MAX_ROUND_RETRIES", "3"))
        self.poll_s = float(c.get("FL_POLL_S", "2"))
        self.model_dir = Path(c.get("FL_MODEL_DIR", "/models"))
        self.control_zones = {z.strip() for z in c.get("FL_CONTROL_ZONES", "").split(",") if z.strip()}
        self.strategy: Strategy | None = None
        self.state = "idle"
        self.round = 0
        self.parameters: Parameters | None = None
        self.history: list[dict[str, Any]] = []
        self.error: str | None = None
        self.run_id = 0
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        spec = self.config.get("FL_STRATEGY")
        if not spec:
            raise ValueError("FL_STRATEGY is required (module:Class of a Strategy)")
        self.strategy = await asyncio.to_thread(load_plugin, spec, self.config)
        if _bool(self.config.get("FL_AUTOSTART"), True):
            self.start_run()

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()

    # ------------------------------------------------------------------ control
    async def handle(self, msg: Incoming) -> Any:
        t, p = msg.type, (msg.payload if isinstance(msg.payload, dict) else {})
        if t == "fl.status":
            return self.status()
        if t in ("fl.start", "fl.stop") and self.control_zones and msg.from_zone not in self.control_zones:
            return {"error": f"zone {msg.from_zone} may not control job {self.job}"}
        if t == "fl.start":
            if self.state in ("waiting_for_clients", "running"):
                return {"error": f"a run is already {self.state}", **self.status()}
            self.start_run(rounds=p.get("rounds"), resume=bool(p.get("resume")))
            log.info("run %d started by %s@%s", self.run_id, msg.from_agent, msg.from_zone)
            return {"ok": True, **self.status()}
        if t == "fl.stop":
            if self._task and not self._task.done():
                self._task.cancel()
                self.state = "stopped"
                log.info("run %d stopped by %s@%s", self.run_id, msg.from_agent, msg.from_zone)
            return {"ok": True, **self.status()}
        return {"error": f"unknown message type {t!r}", "accepts": ["fl.start", "fl.stop", "fl.status"]}

    def start_run(self, rounds: int | None = None, resume: bool = False) -> None:
        if rounds:
            self.rounds = int(rounds)
        self.run_id += 1
        self.error = None
        self.state = "waiting_for_clients"
        if not resume or self.parameters is None:
            assert self.strategy
            self.parameters = self.strategy.initial_parameters()
            self.round, self.history = 0, []
        self._task = asyncio.create_task(self._run(self.run_id))

    def status(self) -> dict[str, Any]:
        return {"job": self.job, "state": self.state, "run": self.run_id, "round": self.round,
                "rounds": self.rounds, "min_clients": self.min_clients, "min_fit_clients": self.min_fit,
                "error": self.error, "history": self.history[-20:]}

    # ------------------------------------------------------------------ the run
    async def _clients(self) -> list[dict[str, Any]]:
        try:
            return await self.gw.directory(f"fl-client:{self.job}")
        except Exception as exc:
            log.warning("directory lookup failed: %s", exc)
            return []

    async def _wait_for_clients(self) -> list[dict[str, Any]]:
        self.state = "waiting_for_clients"
        t0, last = time.time(), -1
        while True:
            clients = await self._clients()
            if len(clients) != last:
                log.info("job %s: %d/%d clients available", self.job, len(clients), self.min_clients)
                last = len(clients)
            if len(clients) >= self.min_clients:
                self.state = "running"
                return clients
            if self.wait_timeout and time.time() - t0 > self.wait_timeout:
                raise TimeoutError(f"only {len(clients)} of {self.min_clients} clients after {self.wait_timeout:.0f}s")
            await asyncio.sleep(self.poll_s)

    async def _fit_one(self, c: dict[str, Any], rnd: int, blob: str, cfg: dict[str, Any]) -> FitResult | str:
        who = f"{c['agent']}@{c['zone']}"
        try:
            r = await self.gw.send(c["zone"], c["agent"], {"type": "fl.fit", "job": self.job, "round": rnd,
                                                          "parameters": blob, "config": cfg},
                                   timeout=self.round_timeout)
            if not isinstance(r, dict) or r.get("status") != "ok":
                return f"{who}: {r.get('status') or r.get('error') if isinstance(r, dict) else r}"
            params = decode(r["parameters"])
            assert self.parameters is not None
            check_shapes(self.parameters, params)
            return FitResult(c["zone"], c["agent"], params, int(r["num_samples"]), dict(r.get("metrics") or {}))
        except (GatewayError, KeyError, ValueError) as exc:
            return f"{who}: {exc}"

    async def _run(self, run_id: int) -> None:
        assert self.strategy and self.parameters is not None
        try:
            retries = 0
            clients: list[dict[str, Any]] = []
            while self.round < self.rounds:
                clients = await self._wait_for_clients()
                rnd = self.round + 1
                cfg = self.strategy.configure_fit(rnd, self.parameters, clients)
                blob = encode(self.parameters)
                t0 = time.time()
                outcomes = await asyncio.gather(*(self._fit_one(c, rnd, blob, cfg) for c in clients))
                results = [o for o in outcomes if isinstance(o, FitResult)]
                failures = [o for o in outcomes if isinstance(o, str)]
                for f in failures:
                    log.warning("round %d: %s", rnd, f)
                if len(results) < self.min_fit:
                    retries += 1
                    log.warning("round %d: only %d/%d clients replied (attempt %d/%d)",
                                rnd, len(results), self.min_fit, retries, self.max_retries)
                    if retries >= self.max_retries:
                        raise RuntimeError(f"round {rnd}: only {len(results)} of {self.min_fit} required clients "
                                           f"replied, {retries} times")
                    await asyncio.sleep(self.poll_s)
                    continue
                retries = 0
                new, fit_metrics = await asyncio.to_thread(
                    self.strategy.aggregate_fit, rnd, results, failures, self.parameters)
                eval_metrics = await asyncio.to_thread(self.strategy.evaluate, rnd, new)
                self.parameters, self.round = new, rnd
                rec = {"round": rnd, "clients": [f"{r.agent}@{r.zone}" for r in results], "failed": len(failures),
                       "seconds": round(time.time() - t0, 2),
                       **{k: round(float(v), 4) for k, v in {**fit_metrics, **eval_metrics}.items()}}
                self.history.append(rec)
                log.info("job %s round %d/%d: %s", self.job, rnd, self.rounds, rec)
                await asyncio.to_thread(self._save)
            self.state = "done"
            log.info("job %s done after %d rounds", self.job, self.round)
            await self._broadcast(clients)
        except asyncio.CancelledError:
            if run_id == self.run_id:
                self.state = "stopped"
            raise
        except Exception as exc:
            log.exception("job %s failed", self.job)
            self.state, self.error = "failed", f"{type(exc).__name__}: {exc}"

    async def _broadcast(self, clients: list[dict[str, Any]]) -> None:
        """Hand the final global model to the clients (best effort)."""
        assert self.parameters is not None
        blob = encode(self.parameters)

        async def one(c: dict[str, Any]) -> None:
            try:
                await self.gw.send(c["zone"], c["agent"], {"type": "fl.model", "job": self.job, "round": self.round,
                                                          "parameters": blob}, timeout=120)
            except Exception as exc:
                log.warning("final model to %s@%s: %s", c["agent"], c["zone"], exc)
        await asyncio.gather(*(one(c) for c in clients))

    def _save(self) -> None:
        assert self.parameters is not None
        self.model_dir.mkdir(parents=True, exist_ok=True)
        path = self.model_dir / f"{self.job}-latest.npz"
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, **self.parameters)
        tmp.replace(path)
        (self.model_dir / f"{self.job}-latest.json").write_text(json.dumps(
            {"job": self.job, "run": self.run_id, "round": self.round, "history": self.history}, indent=2))
