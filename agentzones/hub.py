"""Hub state: one mailbox per spoke zone, routes for replies, and the agent directory.

Delivery is at-least-once. ``poll`` hands out messages and keeps them *in
flight*; a message that is not acknowledged within ``visibility_s`` goes back
to the mailbox and is delivered again. Messages carry an ``expires`` time (the
sender's timeout) and are dropped after it, so a long offline spoke does not
come back to a pile of stale work.

Routing rules (the hub never trusts what a spoke *claims*):

* a request's ``from_zone`` is always set to the authenticated sender;
* a reply is accepted only from the zone the request was sent to, and it is
  delivered only to the zone that sent the request.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

log = logging.getLogger(__name__)

Message = dict[str, Any]


@dataclass
class Route:
    from_zone: str
    to_zone: str
    expires: float


@dataclass
class Mailbox:
    queue: deque = field(default_factory=deque)
    inflight: dict[str, tuple[Message, float]] = field(default_factory=dict)
    event: asyncio.Event = field(default_factory=asyncio.Event)
    last_poll: float = 0.0
    waiting: int = 0  # long polls currently open


class RoutingError(Exception):
    pass


class Hub:
    def __init__(self, zone: str, spokes: list[str], deliver_local: Callable[[Message], Awaitable[None]],
                 visibility_s: float = 30.0, max_queue: int = 1000, directory_ttl_s: float = 60.0,
                 offline_after_s: float = 30.0):
        self.zone = zone
        self.mailboxes = {z: Mailbox() for z in spokes if z != zone}
        self.deliver_local = deliver_local
        self.visibility_s, self.max_queue, self.directory_ttl_s = visibility_s, max_queue, directory_ttl_s
        self.offline_after_s = offline_after_s
        self._seen: dict[str, float] = {}  # message ids already routed (a retried send is not routed twice)
        self.rate: deque = deque(maxlen=300)  # [second, fl, pd, other, bytes] for the last 5 minutes
        self.routes: dict[str, Route] = {}
        self.directory: dict[str, dict[str, Any]] = {}  # zone -> {"agents": [...], "seen": t, "info": {...}}
        # what crossed the hub, for the dashboard: metadata only (payload "type" and size), never contents
        self.traffic: deque = deque(maxlen=300)
        self.totals: dict[str, dict[str, int]] = {}  # "b>cn" -> {"messages": n, "bytes": n}

    # ------------------------------------------------------------------ directory
    def register(self, zone: str, agents: list[dict[str, Any]], info: dict[str, Any] | None = None) -> None:
        clean = [{"agent": str(a.get("agent") or a.get("name")), "capabilities": list(a.get("capabilities") or []),
                  "description": str(a.get("description") or "")[:300]} for a in agents[:200]]
        info = {"via_proxy": bool((info or {}).get("via_proxy")), "version": str((info or {}).get("version", ""))[:20]}
        self.directory[zone] = {"agents": clean, "seen": time.time(), "info": info}

    def online(self, zone: str) -> bool:
        """A spoke is online while it has a long poll open or polled recently."""
        box = self.mailboxes.get(zone)
        return bool(box) and (box.waiting > 0 or time.time() - box.last_poll < self.offline_after_s)

    def remote_entries(self) -> list[dict[str, Any]]:
        now = time.time()  # agents of a spoke that stopped polling are hidden at once, not after the TTL
        return [{"zone": z, **a} for z, d in sorted(self.directory.items())
                if now - d["seen"] <= self.directory_ttl_s and self.online(z) for a in d["agents"]]

    def zones(self) -> dict[str, dict[str, Any]]:
        now = time.time()
        return {z: {"queued": len(m.queue), "inflight": len(m.inflight), "online": self.online(z),
                    "last_poll_s": round(now - m.last_poll, 1) if m.last_poll else None,
                    "agents": len(self.directory.get(z, {}).get("agents", [])),
                    **self.directory.get(z, {}).get("info", {})}
                for z, m in self.mailboxes.items()}

    # ------------------------------------------------------------------ routing
    async def accept(self, msg: Message, sender: str) -> None:
        """Take a message from ``sender`` (a spoke, or this hub's own zone) and route it."""
        now = time.time()
        self._expire_routes(now)
        key = f"{sender}:{msg.get('id')}"
        if key in self._seen:  # the sender retried after a lost response: already routed
            return
        if msg.get("kind") == "reply":
            route = self.routes.get(str(msg.get("reply_to")))
            if route is None or route.to_zone != sender:
                raise RoutingError("reply does not match a request sent to your zone")
            del self.routes[str(msg["reply_to"])]
            msg = {**msg, "from_zone": sender, "to_zone": route.from_zone}
        else:
            msg = {**msg, "kind": "request", "from_zone": sender}
            target = msg.get("to_zone")
            if target != self.zone and target not in self.mailboxes:
                raise RoutingError(f"unknown zone {target!r}")
            if target != self.zone and not self.online(target):
                raise RoutingError(f"zone {target} is offline (not polling the hub)")
            self.routes[msg["id"]] = Route(sender, target, float(msg.get("expires") or now + 300))
        self._seen[key] = now
        self._record(msg, now)
        if msg["to_zone"] == self.zone:
            await self.deliver_local(msg)
        else:
            self._enqueue(msg["to_zone"], msg)

    def _record(self, msg: Message, now: float) -> None:
        payload = msg.get("payload")
        size = len(json.dumps(payload, separators=(",", ":"), default=str))
        ptype = payload.get("type") if isinstance(payload, dict) else None
        self.traffic.append({"t": round(now, 3), "kind": msg["kind"], "from_zone": msg["from_zone"],
                             "from_agent": msg.get("from_agent"), "to_zone": msg["to_zone"],
                             "to_agent": msg.get("to_agent"), "type": ptype if isinstance(ptype, str) else None,
                             "bytes": size, "status": payload.get("status") if isinstance(payload, dict) else None})
        tot = self.totals.setdefault(f"{msg['from_zone']}>{msg['to_zone']}", {"messages": 0, "bytes": 0})
        tot["messages"] += 1
        tot["bytes"] += size
        sec = int(now)
        if not self.rate or self.rate[-1][0] != sec:
            self.rate.append([sec, 0, 0, 0, 0])
        fam = ptype.split(".")[0] if isinstance(ptype, str) else ""
        self.rate[-1][{"fl": 1, "pd": 2}.get(fam, 3)] += 1
        self.rate[-1][4] += size

    def _enqueue(self, zone: str, msg: Message) -> None:
        box = self.mailboxes[zone]
        if len(box.queue) >= self.max_queue:
            dropped = box.queue.popleft()
            log.warning("mailbox %s full, dropped message %s", zone, dropped.get("id"))
        box.queue.append(msg)
        box.event.set()

    def _expire_routes(self, now: float) -> None:
        if len(self.routes) > 100 and any(r.expires < now for r in self.routes.values()):
            self.routes = {k: r for k, r in self.routes.items() if r.expires >= now}
        if len(self._seen) > 5000:
            self._seen = {k: t for k, t in self._seen.items() if now - t < 600}

    # ------------------------------------------------------------------ spoke side
    def _take(self, box: Mailbox, limit: int) -> list[Message]:
        now = time.time()
        for mid, (msg, deadline) in list(box.inflight.items()):  # unacknowledged -> redeliver
            if deadline < now:
                del box.inflight[mid]
                box.queue.appendleft(msg)
        out: list[Message] = []
        while box.queue and len(out) < limit:
            msg = box.queue.popleft()
            if float(msg.get("expires") or now + 1) < now:
                continue
            box.inflight[msg["id"]] = (msg, now + self.visibility_s)
            out.append(msg)
        if not box.queue:
            box.event.clear()
        return out

    async def poll(self, zone: str, wait: float, limit: int = 50) -> list[Message]:
        box = self.mailboxes[zone]
        box.last_poll = time.time()
        msgs = self._take(box, limit)
        if msgs or wait <= 0:
            return msgs
        # wake up early enough to hand out messages whose ack timed out
        deadline = min([d for _, d in box.inflight.values()] + [time.time() + wait])
        box.waiting += 1
        try:
            await asyncio.wait_for(box.event.wait(), timeout=max(deadline - time.time(), 0.05))
        except asyncio.TimeoutError:
            pass
        finally:
            box.waiting -= 1
            box.last_poll = time.time()
        return self._take(box, limit)

    def ack(self, zone: str, ids: list[str]) -> int:
        box = self.mailboxes[zone]
        return sum(box.inflight.pop(str(i), None) is not None for i in ids)
