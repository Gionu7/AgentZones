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


def clean_accepts(v: Any) -> dict[str, str]:
    """A directory entry's message types, size-limited (they travel through the hub to every zone)."""
    if not isinstance(v, dict):
        return {}
    return {str(k)[:64]: str(d)[:300] for k, d in list(v.items())[:50]}


class RoutingError(Exception):
    pass


class Hub:
    def __init__(self, zone: str, spokes: list[str], deliver_local: Callable[[Message], Awaitable[None]],
                 visibility_s: float = 30.0, max_queue: int = 1000, directory_ttl_s: float = 60.0):
        self.zone = zone
        self.mailboxes = {z: Mailbox() for z in spokes if z != zone}
        self.deliver_local = deliver_local
        self.visibility_s, self.max_queue, self.directory_ttl_s = visibility_s, max_queue, directory_ttl_s
        self.routes: dict[str, Route] = {}
        self.directory: dict[str, dict[str, Any]] = {}  # zone -> {"agents": [...], "seen": t}

    # ------------------------------------------------------------------ directory
    def register(self, zone: str, agents: list[dict[str, Any]]) -> None:
        clean = [{"agent": str(a.get("agent") or a.get("name")), "capabilities": list(a.get("capabilities") or []),
                  "description": str(a.get("description") or "")[:300], "version": str(a.get("version") or "")[:40],
                  "accepts": clean_accepts(a.get("accepts"))} for a in agents[:200]]
        self.directory[zone] = {"agents": clean, "seen": time.time()}

    def remote_entries(self) -> list[dict[str, Any]]:
        now = time.time()
        return [{"zone": z, **a} for z, d in sorted(self.directory.items())
                if now - d["seen"] <= self.directory_ttl_s for a in d["agents"]]

    def zones(self) -> dict[str, dict[str, Any]]:
        now = time.time()
        return {z: {"queued": len(m.queue), "inflight": len(m.inflight),
                    "last_poll_s": round(now - m.last_poll, 1) if m.last_poll else None,
                    "agents": len(self.directory.get(z, {}).get("agents", []))}
                for z, m in self.mailboxes.items()}

    # ------------------------------------------------------------------ routing
    async def accept(self, msg: Message, sender: str) -> None:
        """Take a message from ``sender`` (a spoke, or this hub's own zone) and route it."""
        now = time.time()
        self._expire_routes(now)
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
            self.routes[msg["id"]] = Route(sender, target, float(msg.get("expires") or now + 300))
        if msg["to_zone"] == self.zone:
            await self.deliver_local(msg)
        else:
            self._enqueue(msg["to_zone"], msg)

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
        try:
            await asyncio.wait_for(box.event.wait(), timeout=max(deadline - time.time(), 0.05))
        except asyncio.TimeoutError:
            pass
        box.last_poll = time.time()
        return self._take(box, limit)

    def ack(self, zone: str, ids: list[str]) -> int:
        box = self.mailboxes[zone]
        return sum(box.inflight.pop(str(i), None) is not None for i in ids)
