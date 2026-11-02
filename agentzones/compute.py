"""Sharing idle compute: zones lend their local model to peers, and route LLM calls.

Each zone advertises in its gossip card what it can run::

    "compute": {"share": true, "model": "qwen2.5:7b", "size_b": 7.0, "gpu": "NVIDIA T4",
                "active": 0, "max_concurrent": 2, "cpus": 8, "load1": 0.4, "served": 12}

and, if ``compute.share`` is on, serves ``POST /p2p/llm/chat`` (signed, like
every peer call) with its *own* model only - no further routing, so no loops.
Tools are never executed by the lender: a borrowed model only returns text or
tool calls, and the borrowing zone runs its own tools. So a laptop with a 1.5B
model can borrow a 7B on an idle GPU server for the hard cases.

Routes (``AgentConfig.model_route``, the ``model`` field of the zone's
OpenAI-compatible ``/v1/chat/completions``):

    local      this zone's model (default)
    largest    the biggest model in the network with free capacity (local wins ties)
    any        the least-loaded model in the network
    <zone>     a specific peer zone's model (also "<zone>/<model>")

If the chosen model fails or is busy, the router falls back to the next
candidate and finally to the local model.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from typing import TYPE_CHECKING, Any

from .config import ComputeConfig
from .llm import LLMReply, LLMUnavailable, ToolCall

if TYPE_CHECKING:
    from .peers import Peer
    from .zone import Zone

_SIZE = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)\s*[bB](?![a-zA-Z])")


def guess_size_b(model: str | None) -> float | None:
    """'qwen2.5:7b' -> 7.0, 'Qwen2.5-1.5B-Instruct' -> 1.5, 'llama3.3:70b-q4' -> 70.0"""
    if not model:
        return None
    m = _SIZE.findall(model.split("/")[-1].replace(":", " "))
    return float(m[-1]) if m else None


def detect_gpu() -> str:
    if not shutil.which("nvidia-smi"):
        return ""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=5).stdout.strip()
        return ", ".join(sorted(set(out.splitlines())))
    except Exception:
        return ""


class ComputeShare:
    """Lending side: advertise capacity, serve peers' chat requests within limits."""

    def __init__(self, zone: "Zone", cfg: ComputeConfig):
        self.zone, self.cfg = zone, cfg
        self.size_b = cfg.size_b if cfg.size_b is not None else guess_size_b(zone.cfg.model.model)
        self.gpu = cfg.gpu if cfg.gpu is not None else detect_gpu()
        self.active = 0
        self.served = 0
        self.borrowed = 0

    def card(self) -> dict[str, Any]:
        enabled = self.zone.llm.enabled
        try:
            load1 = round(os.getloadavg()[0], 2)
        except OSError:
            load1 = None
        return {"share": bool(self.cfg.share and enabled), "model": self.zone.cfg.model.model if enabled else None,
                "size_b": self.size_b if enabled else None, "gpu": self.gpu or None, "active": self.active,
                "max_concurrent": self.cfg.max_concurrent, "cpus": os.cpu_count(), "load1": load1,
                "served": self.served, "borrowed": self.borrowed}

    async def serve(self, payload: dict[str, Any], caller: str) -> tuple[int, dict[str, Any]]:
        if not (self.cfg.share and self.zone.llm.enabled):
            return 403, {"error": "this zone does not share its model"}
        if self.cfg.accept_from and caller not in self.cfg.accept_from:
            return 403, {"error": f"zone {caller} may not use this model"}
        if self.active >= self.cfg.max_concurrent:
            return 503, {"error": "busy"}
        self.active += 1
        t0 = time.time()
        try:
            reply = await self.zone.llm.chat(payload.get("messages") or [], payload.get("tools") or None,
                                             payload.get("temperature"), bool(payload.get("json_mode")))
        except LLMUnavailable as exc:
            return 502, {"error": str(exc)}
        finally:
            self.active -= 1
        self.served += 1
        ms = int((time.time() - t0) * 1000)
        self.zone.event("llm", f"served a chat for {caller} on {self.zone.cfg.model.model} ({ms} ms)",
                        peer=caller, direction="in")
        return 200, {"content": reply.content, "model": reply.model, "zone": self.zone.name, "ms": ms,
                     "tool_calls": [{"id": c.id, "name": c.name, "arguments": c.arguments} for c in reply.tool_calls]}


class ModelRouter:
    """Borrowing side: send an LLM call to the right model in the network."""

    def __init__(self, zone: "Zone"):
        self.zone = zone

    def _free(self, card: dict[str, Any]) -> bool:
        c = card.get("compute") or {}
        return bool(c.get("share") and c.get("model")) and c.get("active", 0) < c.get("max_concurrent", 1)

    def network_models(self) -> list[dict[str, Any]]:
        out = []
        if self.zone.llm.enabled:
            c = self.zone.compute.card()
            out.append({"zone": self.zone.name, "local": True, **c})
        for p in self.zone.peers.alive():
            c = p.card.get("compute") or {}
            if c.get("share") and c.get("model"):
                out.append({"zone": p.name, "local": False, **c, "free": self._free(p.card)})
        return out

    def candidates(self, route: str) -> list["Peer | None"]:
        """Ordered list; None means the local model."""
        local_ok = self.zone.llm.enabled
        peers = [p for p in self.zone.peers.alive() if self._free(p.card)]
        size = lambda p: (p.card.get("compute") or {}).get("size_b") or 0.0  # noqa: E731
        load = lambda p: (p.card.get("compute") or {}).get("active", 0) / max(  # noqa: E731
            (p.card.get("compute") or {}).get("max_concurrent", 1), 1)
        route = (route or "local").strip()
        if route == "local":
            return [None]
        if route == "largest":
            my = (self.zone.compute.size_b or 0.0) if local_ok else -1.0
            ranked = sorted(peers, key=lambda p: (-size(p), load(p)))
            bigger = [p for p in ranked if size(p) > my]
            return bigger + ([None] if local_ok else []) + [p for p in ranked if p not in bigger]
        if route == "any":
            ranked = sorted(peers, key=load)
            return ([None] if local_ok and self.zone.compute.active == 0 else []) + ranked + \
                   ([None] if local_ok else [])
        zone_name = route.split("/", 1)[0]
        if zone_name == self.zone.name:
            return [None]
        p = self.zone.peers.get(zone_name)
        if p is None:
            raise LLMUnavailable(f"unknown or offline zone {zone_name!r}")
        return [p] + ([None] if local_ok else [])

    async def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
                   route: str = "local", temperature: float | None = None, json_mode: bool = False,
                   timeout: float = 180) -> LLMReply:
        errors = []
        seen: set[str] = set()
        for cand in self.candidates(route):
            key = "local" if cand is None else cand.url
            if key in seen:
                continue
            seen.add(key)
            if cand is None:
                if not self.zone.llm.enabled:
                    continue
                try:
                    reply = await self.zone.llm.chat(messages, tools, temperature, json_mode)
                    reply.served_by = self.zone.name
                    return reply
                except LLMUnavailable as exc:
                    errors.append(f"local: {exc}")
                    continue
            try:
                res = await self.zone.peer_client.request(cand.url, "POST", "/p2p/llm/chat", {
                    "messages": messages, "tools": tools, "temperature": temperature, "json_mode": json_mode},
                    timeout=timeout)
            except Exception as exc:
                errors.append(f"{cand.name}: {exc}")
                continue
            self.zone.compute.borrowed += 1
            self.zone.event("llm", f"borrowed {res.get('model')} on {cand.name} ({res.get('ms')} ms)",
                            peer=cand.name, direction="out")
            return LLMReply(content=res.get("content") or "", served_by=res.get("zone", cand.name or ""),
                            model=res.get("model", ""),
                            tool_calls=[ToolCall(c.get("id", f"call_{i}"), c.get("name", ""), c.get("arguments") or {})
                                        for i, c in enumerate(res.get("tool_calls") or [])])
        raise LLMUnavailable("no model available for route %r (%s)" % (route, "; ".join(errors) or "none"))
