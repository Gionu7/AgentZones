"""Registry-free peer discovery by gossip.

Each zone starts with a few *seed* URLs (any zone already in the network).
Every ``gossip_interval_s`` it POSTs its own card to every known peer at
``/p2p/announce``; the reply contains the peer's card plus the peers *it*
knows. Unknown URLs become candidates and are contacted next round, so the
membership spreads epidemically and survives any single zone going down.
Peers not heard from within ``peer_ttl_s`` are dropped (seeds are retried).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import httpx

from .security import sign_headers

log = logging.getLogger(__name__)


@dataclass
class Peer:
    url: str
    name: str | None = None
    card: dict[str, Any] = field(default_factory=dict)
    last_seen: float = 0.0
    failures: int = 0
    via: str | None = None  # who told us about it

    def public(self) -> dict[str, Any]:
        return {"name": self.name, "url": self.url, "last_seen": self.last_seen}


class PeerClient:
    """Signed HTTP calls to other zones."""

    def __init__(self, zone_name: str, secret: str, timeout: float = 10.0):
        self.zone_name, self.secret = zone_name, secret
        self.http = httpx.AsyncClient(timeout=timeout)

    async def request(self, base_url: str, method: str, path: str, payload: Any = None,
                      timeout: float | None = None) -> Any:
        body = b"" if payload is None else json.dumps(payload).encode()
        headers = sign_headers(self.secret, self.zone_name, method, path, body)
        if payload is not None:
            headers["Content-Type"] = "application/json"
        r = await self.http.request(method, base_url.rstrip("/") + path, content=body or None,
                                    headers=headers, timeout=timeout or self.http.timeout)
        r.raise_for_status()
        return r.json() if r.content else None

    async def close(self) -> None:
        await self.http.aclose()


class PeerTable:
    def __init__(self, self_name: str, self_url: str, seeds: list[str], ttl_s: float, max_peers: int):
        self.self_name, self.self_url = self_name, self_url.rstrip("/")
        # each entry may hold several comma-separated URLs (handy for ZONE_SEED=a,b)
        self.seeds = [u.strip().rstrip("/") for s in seeds for u in (s or "").split(",") if u.strip()]
        self.ttl_s, self.max_peers = ttl_s, max_peers
        self.by_url: dict[str, Peer] = {}
        # hook: return False to refuse a card (e.g. public-key pin mismatch)
        self.accept: Callable[[dict[str, Any]], bool] = lambda card: True
        for s in self.seeds:
            self.add_candidate(s, via="seed")

    def _is_self(self, url: str | None, name: str | None) -> bool:
        return (url or "").rstrip("/") == self.self_url or (name is not None and name == self.self_name)

    def add_candidate(self, url: str, via: str | None = None, name: str | None = None) -> None:
        url = url.rstrip("/")
        if not url or self._is_self(url, name) or url in self.by_url:
            return
        if len(self.by_url) >= self.max_peers:
            return
        self.by_url[url] = Peer(url=url, name=name, via=via)

    def observe(self, card: dict[str, Any], via: str | None = None) -> None:
        """Record a fresh card received from (or about) a live peer."""
        url = (card.get("url") or "").rstrip("/")
        name = card.get("name")
        if not url or self._is_self(url, name):
            return
        if not self.accept(card):
            return
        # a zone may change URL (e.g. pod restart); collapse duplicates by name
        for other_url, p in list(self.by_url.items()):
            if p.name == name and other_url != url and other_url not in self.seeds:
                del self.by_url[other_url]
        peer = self.by_url.get(url)
        if peer is None:
            if len(self.by_url) >= self.max_peers:
                return
            peer = self.by_url[url] = Peer(url=url, via=via)
        peer.name, peer.card, peer.last_seen, peer.failures = name, card, time.time(), 0

    def mark_failure(self, url: str) -> None:
        p = self.by_url.get(url.rstrip("/"))
        if p:
            p.failures += 1

    def expire(self) -> None:
        now = time.time()
        for url, p in list(self.by_url.items()):
            stale = p.last_seen and now - p.last_seen > self.ttl_s
            dead_candidate = not p.last_seen and p.failures >= 5
            if (stale or dead_candidate) and url not in self.seeds:
                del self.by_url[url]
            elif stale:  # seed: forget the card, keep trying
                p.name, p.card, p.last_seen = None, {}, 0.0

    def alive(self) -> list[Peer]:
        now = time.time()
        return [p for p in self.by_url.values() if p.last_seen and now - p.last_seen <= self.ttl_s]

    def get(self, name: str) -> Peer | None:
        for p in self.alive():
            if p.name == name:
                return p
        return None

    def listing(self) -> list[dict[str, Any]]:
        return [p.public() for p in self.alive()]


async def gossip_once(table: PeerTable, client: PeerClient, my_card: dict[str, Any]) -> None:
    async def one(peer: Peer) -> None:
        try:
            resp = await client.request(peer.url, "POST", "/p2p/announce", my_card, timeout=5)
        except Exception as exc:
            log.debug("gossip to %s failed: %s", peer.url, exc)
            table.mark_failure(peer.url)
            return
        card = resp.get("self") or {}
        card.setdefault("url", peer.url)
        table.observe(card, via=peer.url)
        for other in resp.get("peers", []):
            table.add_candidate(other.get("url", ""), via=peer.url, name=other.get("name"))

    await asyncio.gather(*(one(p) for p in list(table.by_url.values())))
    table.expire()
