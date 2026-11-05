"""Federated learning aggregator agent.

Lifecycle of a run::

    idle -> waiting_for_clients -> running -> done | failed | stopped

Each round the aggregator looks up ``fl-client:<FL_JOB>`` in the directory
again, so clients may join or leave between rounds. It sends ``fl.fit`` to all
of them in parallel; the round counts if at least ``FL_MIN_FIT_CLIENTS`` usable
updates come back in time. A round that falls short is retried
(``FL_MAX_ROUND_RETRIES``) and then fails the run.

Who may do what
---------------
* ``fl.status`` - anyone.
* ``fl.start`` {rounds?, resume?, reason?} / ``fl.stop`` - *operators* only: senders named in
  ``FL_OPERATORS`` (default: operator, cli, dashboard-operator).
* ``fl.request`` {action: start|stop, rounds?, reason} - anyone else, e.g. an LLM agent through the
  MCP tools. With ``FL_APPROVAL=required`` (default) it waits in ``pending`` until an operator sends
  ``fl.approve`` {id} or ``fl.reject`` {id}; with ``FL_APPROVAL=auto`` it runs at once.
* ``FL_CONTROL_ZONES`` limits all of the above (except status) to some zones.

Every start, stop, request and decision goes to the audit log (``status.audit`` and
``<FL_MODEL_DIR>/<job>-audit.jsonl``). Limits: ``FL_MAX_ROUNDS`` per start (50),
``FL_COOLDOWN_S`` between runs (0). ``FL_SCHEDULE_S`` > 0 starts a run on its own every N seconds.

Robustness: ``FL_MAX_SAMPLES`` caps the weight one client can claim (0 = no cap) and
``FL_MAX_UPDATE_RATIO`` (default 10) drops an update whose size is that many times the median
(needs 3 or more updates). Each round's history lists every client's share, update size and status.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from ..agent import Agent, GatewayError, Incoming
from . import FitResult, Parameters, Strategy, check_shapes, decode, encode, load_plugin

log = logging.getLogger(__name__)

ACTIVE = ("waiting_for_clients", "running")


def _bool(v: str | None, default: bool) -> bool:
    return default if v is None or v == "" else v.strip().lower() in ("1", "true", "yes", "on")


def _csv(v: str | None) -> set[str]:
    return {x.strip() for x in (v or "").split(",") if x.strip()}


def update_norm(new: Parameters, old: Parameters) -> float:
    return float(np.sqrt(sum(float(np.sum((new[k].astype(np.float64) - old[k]) ** 2)) for k in old)))


class FLAggregator(Agent):
    accepts = {
        "fl.status": "{} -> job, state, round, rounds, history, pending, audit",
        "fl.start": "{rounds?, resume?, reason?}: operators only (FL_OPERATORS) -> status",
        "fl.stop": "{reason?}: operators only -> status",
        "fl.request": "{action: start|stop, rounds?, reason}: anyone; waits for an operator unless "
                      "FL_APPROVAL=auto -> {pending: id}",
        "fl.approve": "{id}: operators only; runs a pending request -> status",
        "fl.reject": "{id, reason?}: operators only -> status",
    }

    def __init__(self, *a: Any, **kw: Any):
        super().__init__(*a, **kw)
        c = self.config
        self.job = c.get("FL_JOB", "default")
        self.capabilities = [f"fl-aggregator:{self.job}"]
        self.description = f"Federated learning aggregator for job {self.job}"
        self.rounds = int(c.get("FL_ROUNDS", "10"))
        self.min_clients = int(c.get("FL_MIN_CLIENTS", "2"))
        self.min_fit = int(c.get("FL_MIN_FIT_CLIENTS") or self.min_clients)
        self.round_timeout = float(c.get("FL_ROUND_TIMEOUT_S", "300"))
        self.wait_timeout = float(c.get("FL_WAIT_TIMEOUT_S", "0"))
        self.max_retries = int(c.get("FL_MAX_ROUND_RETRIES", "3"))
        self.poll_s = float(c.get("FL_POLL_S", "2"))
        self.model_dir = Path(c.get("FL_MODEL_DIR", "/models"))
        self.control_zones = _csv(c.get("FL_CONTROL_ZONES"))
        self.operators = _csv(c.get("FL_OPERATORS", "operator,cli,dashboard-operator"))
        self.approval = c.get("FL_APPROVAL", "required").strip().lower()
        self.max_rounds = int(c.get("FL_MAX_ROUNDS", "50"))
        self.cooldown = float(c.get("FL_COOLDOWN_S", "0"))
        self.schedule = float(c.get("FL_SCHEDULE_S", "0"))
        self.max_samples = int(c.get("FL_MAX_SAMPLES", "0"))
        self.max_ratio = float(c.get("FL_MAX_UPDATE_RATIO", "10"))
        self.strategy: Strategy | None = None
        self.state = "idle"
        self.round = 0
        self.parameters: Parameters | None = None
        self.history: list[dict[str, Any]] = []
        self.error: str | None = None
        self.run_id = 0
        self.started_at: float | None = None
        self.ended_at: float | None = None
        self.started_by: str | None = None
        self.pending: dict[str, dict[str, Any]] = {}
        self.audit: list[dict[str, Any]] = []
        self._task: asyncio.Task | None = None
        self._sched: asyncio.Task | None = None

    async def start(self) -> None:
        spec = self.config.get("FL_STRATEGY")
        if not spec:
            raise ValueError("FL_STRATEGY is required (module:Class of a Strategy)")
        self.strategy = await asyncio.to_thread(load_plugin, spec, self.config)
        if _bool(self.config.get("FL_AUTOSTART"), True):
            self.start_run(who="autostart")
        if self.schedule > 0:
            self._sched = asyncio.create_task(self._schedule_loop())

    async def stop(self) -> None:
        for t in (self._task, self._sched):
            if t:
                t.cancel()

    # ------------------------------------------------------------------ control
    def _log(self, who: str, action: str, outcome: str, **extra: Any) -> None:
        rec = {"t": round(time.time(), 3), "who": who, "action": action, "outcome": outcome,
               **{k: v for k, v in extra.items() if v not in (None, "")}}
        self.audit.append(rec)
        del self.audit[:-200]
        log.info("audit: %s", rec)
        try:
            self.model_dir.mkdir(parents=True, exist_ok=True)
            with (self.model_dir / f"{self.job}-audit.jsonl").open("a") as f:
                f.write(json.dumps(rec) + "\n")
        except OSError as exc:
            log.warning("audit log not written: %s", exc)

    def _check_start(self, rounds: Any) -> tuple[int | None, str | None]:
        """Validate a start: returns (rounds, None) or (None, reason it is refused)."""
        if self.state in ACTIVE:
            return None, f"a run is already {self.state}"
        if self.cooldown and self.ended_at and time.time() - self.ended_at < self.cooldown:
            return None, f"cooldown: wait {self.cooldown - (time.time() - self.ended_at):.0f}s more"
        try:
            n = int(rounds) if rounds else self.rounds
        except (TypeError, ValueError):
            return None, "rounds must be a number"
        if n < 1:
            return None, "rounds must be at least 1"
        return min(n, self.max_rounds), None

    def _do(self, action: str, who: str, p: dict[str, Any], via: str) -> dict[str, Any]:
        reason = str(p.get("reason") or "")[:300]
        if action == "start":
            n, why = self._check_start(p.get("rounds"))
            if why:
                self._log(who, "start", "refused", reason=reason, why=why, via=via)
                return {**self.status(), "error": why}
            capped = f"capped at FL_MAX_ROUNDS={self.max_rounds}" if p.get("rounds") and int(p["rounds"]) > n else None
            self.start_run(rounds=n, resume=bool(p.get("resume")), who=who)
            self._log(who, "start", "started", rounds=n, reason=reason, note=capped, via=via)
            return {**self.status(), "ok": True, **({"note": capped} if capped else {})}
        if action == "stop":
            if self._task and not self._task.done():
                self._task.cancel()
                self.state, self.ended_at = "stopped", time.time()
                self._log(who, "stop", "stopped", reason=reason, via=via)
            else:
                self._log(who, "stop", "nothing to stop", reason=reason, via=via)
            return {**self.status(), "ok": True}
        return {"error": f"unknown action {action!r}"}

    async def handle(self, msg: Incoming) -> Any:
        t, p = msg.type, (msg.payload if isinstance(msg.payload, dict) else {})
        who = f"{msg.from_agent}@{msg.from_zone}"
        if t == "fl.status":
            return self.status()
        if t not in ("fl.start", "fl.stop", "fl.request", "fl.approve", "fl.reject"):
            return self.unknown(t)
        if self.control_zones and msg.from_zone not in self.control_zones:
            self._log(who, t, "refused", why="zone not in FL_CONTROL_ZONES")
            return {"error": f"zone {msg.from_zone} may not control job {self.job}"}
        operator = msg.from_agent in self.operators

        if t in ("fl.start", "fl.stop"):
            if not operator:
                self._log(who, t, "refused", why="not an operator; send fl.request")
                return {"error": f"{msg.from_agent} is not an operator of job {self.job}: send fl.request "
                                 "and an operator will approve it"}
            return self._do(t.split(".")[1], who, p, via="direct")

        if t == "fl.request":
            action = str(p.get("action", "start"))
            if action not in ("start", "stop"):
                return {"error": "action must be start or stop"}
            if action == "start":
                _, why = self._check_start(p.get("rounds"))
                if why:
                    self._log(who, "request start", "refused", why=why, reason=p.get("reason"))
                    return {**self.status(), "error": why}
            if self.approval == "auto":
                return self._do(action, who, p, via="request (auto-approved)")
            rid = uuid.uuid4().hex[:8]
            self.pending[rid] = {"id": rid, "t": round(time.time(), 3), "who": who, "action": action,
                                 "rounds": p.get("rounds"), "reason": str(p.get("reason") or "")[:300]}
            self._log(who, f"request {action}", "pending approval", id=rid, rounds=p.get("rounds"),
                      reason=p.get("reason"))
            return {**self.status(), "ok": True, "pending": rid, "message": "waiting for an operator to approve"}

        # fl.approve / fl.reject
        if not operator:
            self._log(who, t, "refused", why="not an operator")
            return {"error": f"only operators can {t.split('.')[1]} requests"}
        req = self.pending.pop(str(p.get("id")), None)
        if req is None:
            return {**self.status(), "error": f"no pending request {p.get('id')!r}"}
        if t == "fl.reject":
            self._log(who, f"reject {req['action']}", "rejected", id=req["id"], requested_by=req["who"],
                      reason=p.get("reason"))
            return {**self.status(), "ok": True}
        self._log(who, f"approve {req['action']}", "approved", id=req["id"], requested_by=req["who"])
        return self._do(req["action"], req["who"], req, via=f"request approved by {who}")

    def start_run(self, rounds: int | None = None, resume: bool = False, who: str = "") -> None:
        if rounds:
            self.rounds = int(rounds)
        self.run_id += 1
        self.error = None
        self.state = "waiting_for_clients"
        self.started_at, self.ended_at, self.started_by = time.time(), None, who
        if not resume or self.parameters is None:
            assert self.strategy
            self.parameters = self.strategy.initial_parameters()
            self.round, self.history = 0, []
        self._task = asyncio.create_task(self._run(self.run_id))

    async def _schedule_loop(self) -> None:
        while True:
            await asyncio.sleep(self.schedule)
            n, why = self._check_start(None)
            if why is None:
                self.start_run(rounds=n, who="schedule")
                self._log("schedule", "start", "started", rounds=n, reason=f"every {self.schedule:g}s")

    def status(self) -> dict[str, Any]:
        return {"job": self.job, "state": self.state, "run": self.run_id, "round": self.round,
                "rounds": self.rounds, "min_clients": self.min_clients, "min_fit_clients": self.min_fit,
                "error": self.error, "started_at": self.started_at, "ended_at": self.ended_at,
                "started_by": self.started_by, "history": self.history[-100:],
                "pending": list(self.pending.values()), "audit": self.audit[-30:],
                "limits": {"operators": sorted(self.operators), "approval": self.approval,
                           "max_rounds": self.max_rounds, "cooldown_s": self.cooldown,
                           "schedule_s": self.schedule, "control_zones": sorted(self.control_zones),
                           "max_samples": self.max_samples, "max_update_ratio": self.max_ratio,
                           "round_timeout_s": self.round_timeout}}

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

    async def _fit_one(self, c: dict[str, Any], rnd: int, blob: str, cfg: dict[str, Any]) -> FitResult | dict:
        who = f"{c['agent']}@{c['zone']}"
        t0 = time.time()
        try:
            r = await self.gw.send(c["zone"], c["agent"], {"type": "fl.fit", "job": self.job, "round": rnd,
                                                          "parameters": blob, "config": cfg},
                                   timeout=self.round_timeout)
            if not isinstance(r, dict) or r.get("status") != "ok":
                why = (r.get("error") or r.get("status")) if isinstance(r, dict) else str(r)
                return {"client": who, "status": "failed", "why": str(why)[:200]}
            params = decode(r["parameters"])
            assert self.parameters is not None
            check_shapes(self.parameters, params)
            res = FitResult(c["zone"], c["agent"], params, int(r["num_samples"]), dict(r.get("metrics") or {}))
            res.metrics["_seconds"] = round(time.time() - t0, 2)
            return res
        except (GatewayError, KeyError, ValueError) as exc:
            return {"client": who, "status": "failed", "why": str(exc)[:200]}

    def _screen(self, results: list[FitResult]) -> tuple[list[FitResult], list[dict[str, Any]]]:
        """Cap claimed sample counts and drop outlier updates. Returns (used, report per client)."""
        assert self.parameters is not None
        norms = {id(r): update_norm(r.parameters, self.parameters) for r in results}
        med = float(np.median(list(norms.values()))) if norms else 0.0
        used, report = [], []
        for r in results:
            claimed = r.num_samples
            if self.max_samples and r.num_samples > self.max_samples:
                r.num_samples = self.max_samples
            ratio = norms[id(r)] / med if med > 0 else 1.0
            rejected = self.max_ratio > 0 and len(results) >= 3 and ratio > self.max_ratio
            entry = {"client": f"{r.agent}@{r.zone}", "status": "rejected" if rejected else "used",
                     "samples": r.num_samples, "claimed": claimed, "update_norm": round(norms[id(r)], 4),
                     "seconds": r.metrics.pop("_seconds", None),
                     "metrics": {k: round(float(v), 4) for k, v in r.metrics.items()}}
            if rejected:
                entry["why"] = f"update {ratio:.1f}x the median size (limit {self.max_ratio:g}x)"
            else:
                used.append(r)
            report.append(entry)
        total = sum(r.num_samples for r in used) or 1
        for e in report:
            e["share"] = round(e["samples"] / total, 4) if e["status"] == "used" else 0.0
        return used, report

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
                failed = [o for o in outcomes if isinstance(o, dict)]
                used, report = self._screen(results)
                report += failed
                for f in report:
                    if f["status"] != "used":
                        log.warning("round %d: %s %s: %s", rnd, f["client"], f["status"], f.get("why"))
                if len(used) < self.min_fit:
                    retries += 1
                    log.warning("round %d: only %d/%d usable updates (attempt %d/%d)",
                                rnd, len(used), self.min_fit, retries, self.max_retries)
                    if retries >= self.max_retries:
                        raise RuntimeError(f"round {rnd}: only {len(used)} of {self.min_fit} required clients "
                                           f"sent usable updates, {retries} times")
                    await asyncio.sleep(self.poll_s)
                    continue
                retries = 0
                failures = [f["client"] for f in report if f["status"] != "used"]
                new, fit_metrics = await asyncio.to_thread(
                    self.strategy.aggregate_fit, rnd, used, failures, self.parameters)
                eval_metrics = await asyncio.to_thread(self.strategy.evaluate, rnd, new)
                self.parameters, self.round = new, rnd
                rec = {"round": rnd, "clients": [f"{r.agent}@{r.zone}" for r in used], "failed": len(failures),
                       "seconds": round(time.time() - t0, 2), "bytes_per_update": len(blob) * 3 // 4,
                       **{k: round(float(v), 4) for k, v in {**fit_metrics, **eval_metrics}.items()},
                       "contrib": sorted(report, key=lambda e: e["client"])}
                self.history.append(rec)
                log.info("job %s round %d/%d: %s", self.job, rnd, self.rounds,
                         {k: v for k, v in rec.items() if k != "contrib"})
                await asyncio.to_thread(self._save)
            self.state, self.ended_at = "done", time.time()
            log.info("job %s done after %d rounds", self.job, self.round)
            await self._broadcast(clients)
        except asyncio.CancelledError:
            if run_id == self.run_id:
                self.state, self.ended_at = "stopped", time.time()
            raise
        except Exception as exc:
            log.exception("job %s failed", self.job)
            self.state, self.error, self.ended_at = "failed", f"{type(exc).__name__}: {exc}", time.time()

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
