"""The Zone runtime: wires model, MCP tools, agents, peers, fixes and FL together."""

from __future__ import annotations

import asyncio
import contextvars
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from . import __version__
from .agent import Agent, AskContext, build_agent
from .config import ZoneConfig
from .federated import FederatedNode, load_task, task_kwargs
from .identity import Identity, KeyPins, verify_sig
from .knowledge import FixRejected, FixStore
from .llm import LLM
from .peers import PeerClient, PeerTable, gossip_once
from .tools import ToolRegistry

log = logging.getLogger(__name__)

MAX_HOPS = 2


class Zone:
    def __init__(self, cfg: ZoneConfig):
        self.cfg = cfg
        self.name = cfg.name
        self.data_dir = Path(cfg.data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.llm = LLM(cfg.model)
        self.tools = ToolRegistry()
        self.peer_client = PeerClient(cfg.name, cfg.security.network_secret)
        self.peers = PeerTable(cfg.name, cfg.url, cfg.discovery.peers, cfg.discovery.peer_ttl_s,
                               cfg.discovery.max_peers)
        self.identity = Identity.load_or_create(self.data_dir / "zone_key.pem")
        self.pins = KeyPins(self.data_dir / "known_zones.json",
                            {**cfg.security.trusted_keys, cfg.name: self.identity.public_b64})
        self.peers.accept = self._accept_card
        self.fixes = FixStore(self.data_dir / "fixes.json", self._verify_attestation)
        # domain validators for fixes received from peers: fn(fix) -> None if OK, else reason
        self.fix_validators: list[Callable[[dict[str, Any]], str | None]] = []
        self.fl: FederatedNode | None = None
        self.agents: dict[str, Agent] = {}
        self.events: list[dict[str, Any]] = []  # recent activity, for the dashboard
        self.current_ctx: contextvars.ContextVar[AskContext] = contextvars.ContextVar("az_ctx", default=AskContext())
        self._tasks: list[asyncio.Task] = []
        self._seen_requests: dict[str, float] = {}
        self.started = time.time()

    # ------------------------------------------------------------------ lifecycle
    async def start(self) -> None:
        self._register_builtins()
        await self.tools.start_mcp(self.cfg.mcp_servers)
        for acfg in self.cfg.agents:
            agent = build_agent(self, acfg)
            await agent.start()
            self.agents[acfg.name] = agent
        if self.cfg.federated.enabled:
            f = self.cfg.federated
            task = load_task(f.task, **task_kwargs(f.data, self.data_dir, self.name))
            self.fl = FederatedNode(task, f.local_epochs, f.learning_rate, f.dp_noise_std)
        self._tasks.append(asyncio.create_task(self._gossip_loop()))
        if self.fl and self.cfg.federated.auto_rounds > 0:
            self._tasks.append(asyncio.create_task(self._fl_loop()))
        self.event("zone", f"zone {self.name} started at {self.cfg.url}")

    async def stop(self) -> None:
        for t in self._tasks:
            t.cancel()
        await self.tools.close()
        await self.peer_client.close()
        await self.llm.close()

    def event(self, kind: str, msg: str, **data: Any) -> None:
        self.events.append({"t": time.time(), "kind": kind, "msg": msg, **data})
        del self.events[:-200]
        log.info("[%s] %s", kind, msg)

    def _register_builtins(self) -> None:
        t = self.tools

        @t.builtin("list_peers", "List the peer zones currently reachable, with their public agents and exposed tools.")
        async def _list_peers(args):
            return [{"zone": p.name, "url": p.url,
                     "agents": [a["name"] + ": " + a.get("description", "") for a in p.card.get("agents", [])],
                     "tools": [x["name"] for x in p.card.get("tools", [])]} for p in self.peers.alive()]

        @t.builtin("ask_peers", "Ask agents in peer zones for help. Use when you are unsure. "
                   "Returns one answer per zone.", {
                       "type": "object", "required": ["question"], "properties": {
                           "question": {"type": "string"},
                           "agent": {"type": "string", "description": "agent name to ask (default: same kind)"},
                           "zone": {"type": "string", "description": "only ask this zone"}}})
        async def _ask_peers(args):
            return await self.ask_peers(args["question"], args.get("context"), args.get("agent"),
                                        [args["zone"]] if args.get("zone") else None)

        @t.builtin("call_peer_tool", "Call a tool exposed by a peer zone (see list_peers).", {
            "type": "object", "required": ["zone", "tool"], "properties": {
                "zone": {"type": "string"}, "tool": {"type": "string"},
                "arguments": {"type": "object"}}})
        async def _call_peer_tool(args):
            return await self.call_peer_tool(args["zone"], args["tool"], args.get("arguments") or {})

        @t.builtin("search_fixes", "Search this zone's store of verified fixes.", {
            "type": "object", "properties": {"problem": {"type": "string"}, "text": {"type": "string"}}},
            exposed=True)
        async def _search_fixes(args):
            return self.fixes.search(args.get("problem"), args.get("text", ""))

    # ------------------------------------------------------------------ identity
    def card(self) -> dict[str, Any]:
        return {
            "name": self.name, "url": self.cfg.url, "version": __version__, "tags": self.cfg.tags,
            "agents": [a.card() for a in self.agents.values() if a.cfg.public],
            "tools": [t.spec() for t in self.tools.exposed()],
            "fl_task": self.fl.task.name if self.fl else None,
            "model": self.cfg.model.model if self.llm.enabled else None,
            "pubkey": self.identity.public_b64,
            "ts": time.time(),
        }

    # ------------------------------------------------------------------ identity / trust
    def _accept_card(self, card: dict[str, Any]) -> bool:
        name, key = card.get("name"), card.get("pubkey")
        if not self.cfg.security.tofu and name not in self.cfg.security.trusted_keys:
            return False
        if self.pins.check(name, key):
            return True
        self.event("security", f"refused card for '{name}': public key does not match the pinned key")
        return False

    def _verify_attestation(self, zone: str, data: bytes, sig: str) -> bool:
        key = self.pins.key_for(zone)
        return bool(key) and verify_sig(key, data, sig)

    # ------------------------------------------------------------------ gossip
    async def _gossip_loop(self) -> None:
        while True:
            try:
                before = {p.name for p in self.peers.alive()}
                await gossip_once(self.peers, self.peer_client, self.card())
                after = {p.name for p in self.peers.alive()}
                for n in sorted(after - before):
                    self.event("peer", f"discovered peer zone {n}")
                for n in sorted(before - after):
                    self.event("peer", f"lost peer zone {n}")
            except Exception:
                log.exception("gossip failed")
            await asyncio.sleep(self.cfg.discovery.gossip_interval_s)

    def on_announce(self, card: dict[str, Any], remote_ip: str | None) -> dict[str, Any]:
        is_new = self.peers.get(card.get("name", "")) is None
        self.peers.observe(card, via="announce")
        if self.peers.get(card.get("name", "")) is None:
            return {"error": "card refused"}
        if is_new and card.get("name") and card.get("name") != self.name:
            self.event("peer", f"peer zone {card.get('name')} announced itself")
        return {"self": self.card(), "peers": self.peers.listing()}

    # ------------------------------------------------------------------ asking
    def _dedupe(self, request_id: str) -> bool:
        now = time.time()
        for k, t in list(self._seen_requests.items()):
            if now - t > 300:
                del self._seen_requests[k]
        if request_id in self._seen_requests:
            return False
        self._seen_requests[request_id] = now
        return True

    async def ask(self, agent_name: str, question: str, context: dict[str, Any] | None = None,
                  ctx: AskContext | None = None) -> dict[str, Any]:
        ctx = ctx or AskContext(origin="local", request_id=uuid.uuid4().hex)
        agent = self.agents.get(agent_name)
        if agent is None:
            return {"zone": self.name, "agent": agent_name, "error": "no such agent"}
        if ctx.origin != "local" and not agent.cfg.public:
            return {"zone": self.name, "agent": agent_name, "error": "agent not public"}
        token = self.current_ctx.set(ctx)
        try:
            res = await agent.ask(question, context, ctx)
        finally:
            self.current_ctx.reset(token)
        return {"zone": self.name, "agent": agent_name, **res}

    async def on_peer_ask(self, payload: dict[str, Any]) -> dict[str, Any]:
        c = payload.get("ctx") or {}
        ctx = AskContext(origin=c.get("origin", "peer"), hops=int(c.get("hops", 1)),
                         trace=list(c.get("trace", [])), request_id=c.get("request_id") or uuid.uuid4().hex)
        if self.name in ctx.trace or not self._dedupe(ctx.request_id + ":" + payload.get("agent", "")):
            return {"zone": self.name, "error": "loop detected"}
        ctx.trace.append(self.name)
        self.event("ask", f"question from {ctx.origin} for agent {payload.get('agent')}",
                   question=payload.get("question", "")[:200])
        return await self.ask(payload.get("agent", ""), payload.get("question", ""), payload.get("context"), ctx)

    async def ask_peers(self, question: str, context: dict[str, Any] | None = None, agent: str | None = None,
                        zones: list[str] | None = None, timeout: float = 120) -> list[dict[str, Any]]:
        parent = self.current_ctx.get()
        if parent.hops >= MAX_HOPS:
            return [{"error": f"hop limit ({MAX_HOPS}) reached; not forwarding"}]
        trace = list(parent.trace) or [self.name]
        if self.name not in trace:
            trace.append(self.name)
        ctx = {"origin": parent.origin if parent.origin != "local" else self.name, "hops": parent.hops + 1,
               "trace": trace, "request_id": parent.request_id or uuid.uuid4().hex}
        targets = []
        for p in self.peers.alive():
            if zones and p.name not in zones:
                continue
            if p.name in trace:
                continue
            names = [a["name"] for a in p.card.get("agents", [])]
            pick = agent if agent in names else (None if agent else (names[0] if names else None))
            if pick:
                targets.append((p, pick))
        if not targets:
            return []
        self.event("ask", f"asking {len(targets)} peer zone(s): {', '.join(p.name or '?' for p, _ in targets)}")

        async def one(p, agent_name):
            try:
                return await self.peer_client.request(
                    p.url, "POST", "/p2p/ask",
                    {"agent": agent_name, "question": question, "context": context, "ctx": ctx}, timeout=timeout)
            except Exception as exc:
                return {"zone": p.name, "agent": agent_name, "error": str(exc)}

        return list(await asyncio.gather(*(one(p, a) for p, a in targets)))

    async def call_peer_tool(self, zone: str, tool: str, arguments: dict[str, Any] | None = None) -> Any:
        p = self.peers.get(zone)
        if not p:
            raise KeyError(f"unknown or offline zone {zone}")
        self.event("tool", f"calling {tool} on zone {zone}")
        res = await self.peer_client.request(p.url, "POST", "/p2p/tools/call",
                                             {"tool": tool, "arguments": arguments or {}}, timeout=120)
        if res.get("error"):
            raise RuntimeError(res["error"])
        return res.get("result")

    async def on_peer_tool_call(self, payload: dict[str, Any], caller: str) -> dict[str, Any]:
        name = payload.get("tool", "")
        tool = self.tools.tools.get(name)
        if not tool or not tool.exposed:
            return {"error": f"tool {name} is not exposed by zone {self.name}"}
        self.event("tool", f"zone {caller} called {name}")
        try:
            return {"result": await tool.handler(payload.get("arguments") or {})}
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}

    # ------------------------------------------------------------------ fixes
    async def share_fix(self, fix: dict[str, Any]) -> list[str]:
        async def one(p):
            try:
                r = await self.peer_client.request(p.url, "POST", "/p2p/fixes", fix, timeout=10)
                return p.name if r and r.get("ok") else None
            except Exception:
                return None
        sent = [n for n in await asyncio.gather(*(one(p) for p in self.peers.alive())) if n]
        if sent:
            self.event("fix", f"shared fix '{fix['summary']}' with {', '.join(sent)}")
        return sent

    def add_verified_fix(self, **kw: Any) -> dict[str, Any]:
        """Store a fix this zone verified itself, with a signed attestation."""
        return self.fixes.add_verified(zone=self.name, sign=self.identity.sign, **kw)

    def on_peer_fix(self, fix: dict[str, Any], caller: str) -> dict[str, Any]:
        for check in self.fix_validators:
            reason = check(fix)
            if reason:
                self.event("security", f"rejected fix from {caller}: {reason}")
                return {"ok": False, "reason": reason}
        try:
            merged, n_new, n_bad = self.fixes.merge(fix)
        except FixRejected as exc:
            self.event("security", f"rejected fix from {caller}: {exc}")
            return {"ok": False, "reason": str(exc)}
        if n_bad:
            self.event("security", f"ignored {n_bad} invalid attestation(s) in fix from {caller}")
        if n_new:
            self.event("fix", f"received fix from {caller}: {merged['summary']} "
                              f"(verified by {', '.join(merged['verified_by'])})")
        res = {"ok": True, "id": merged["id"], "verified_by": merged["verified_by"]}
        if n_bad:
            res["ignored_attestations"] = n_bad
        return res

    # ------------------------------------------------------------------ federated
    async def fl_train(self) -> dict[str, Any]:
        if not self.fl:
            raise RuntimeError("federated learning not enabled")
        rec = await asyncio.to_thread(self.fl.train)
        self.event("fl", f"local training done (round {rec['round']}), local acc={rec.get('local_acc', 0):.3f}")
        return rec

    async def fl_aggregate(self) -> dict[str, Any]:
        if not self.fl:
            raise RuntimeError("federated learning not enabled")
        peers = [p for p in self.peers.alive() if p.card.get("fl_task") == self.fl.task.name]

        async def one(p):
            try:
                r = await self.peer_client.request(p.url, "GET", "/p2p/fl/weights", timeout=30)
                r["zone"] = p.name
                return r
            except Exception as exc:
                log.warning("fl weights from %s failed: %s", p.name, exc)
                return None
        payloads = [r for r in await asyncio.gather(*(one(p) for p in peers)) if r]
        rec = self.fl.aggregate(payloads)
        self.event("fl", f"aggregated with {len(payloads)} peer(s) (round {rec['round']}), "
                         f"global acc={rec.get('global_acc', 0):.3f}")
        return rec

    async def fl_round(self) -> dict[str, Any]:
        train = await self.fl_train()
        agg = await self.fl_aggregate()
        return {"train": train, "aggregate": agg}

    async def _fl_loop(self) -> None:
        for _ in range(self.cfg.federated.auto_rounds):
            await asyncio.sleep(self.cfg.federated.round_interval_s)
            try:
                await self.fl_round()
            except Exception:
                log.exception("fl round failed")
