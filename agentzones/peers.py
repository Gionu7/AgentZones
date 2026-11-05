"""Agents that learn from the agents of the same kind in other zones.

Several zones can run an agent that does the same job on its own machine, for example a ``proxy-doctor``
in every zone. :class:`PeerAgent` lets such agents pool what they learn, and nothing else:

``<prefix>.consult`` {question, observation}
    An agent that is unsure of its diagnosis, or knows no verified fix for it, sends its observation (only
    the keys in ``share_keys``) to its peers in the other zones. Each one classifies it with its own model
    and returns its diagnosis and the fixes verified for it. Unsure: the peers vote. Sure: only their fixes
    are used.
``<prefix>.fix`` {fix}
    A fix an agent applied and then verified on its own machine goes to its peers in the other zones. The
    receiver stores it as verified by the sending zone only (a zone can vouch only for itself) and never
    applies it: its own agent must verify it before its zone is added to ``verified_by``.
``fixes`` {problem?, text?}
    The verified fixes this agent knows.

Peers are the agents that advertise ``peer_capability``. They are found in the directory, and every message
goes through the gateway like any other. A subclass supplies the domain::

    class DnsDoctor(PeerAgent):
        capabilities = ["dns-doctor", "chat"]
        peer_capability = "dns-doctor"          # who the peers are
        peer_prefix = "dns"                     # dns.consult, dns.fix; settings DNS_THRESHOLD, ...
        share_keys = ("resolver", "rcode")      # all that peers ever see of an observation
        problems = {"healthy": "...", "stale_cache": "..."}   # the labels a diagnosis can have

        async def classify(self, observation):  # -> {"problem": label, "confidence": 0..1, ...}
            ...

        async def handle(self, msg):
            if (reply := await self.handle_peer(msg)) is not None:
                return reply
            ...                                 # its own messages: look at the machine, then
            d = await self.ask_peers(question, obs, await self.classify(obs))
            fix = self.best_fix(d.problem, d.fixes)   # ours > a peer's > None (use your own catalog)
            ...                                 # apply it, check again, and only if that passes:
            await self.learned(d.problem, summary, steps)

A subclass that overrides :meth:`start` or :meth:`status` calls ``super()`` there.

Settings (``<P>`` is ``peer_prefix`` in upper case): ``<P>_THRESHOLD`` (0.75), below which the peers vote;
``<P>_SHARE_FIXES`` (1), send verified fixes to the peers; ``<P>_PEER_TIMEOUT_S`` (30); and ``DATA_DIR``, where
the fixes are kept (``fixes_file``).
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .agent import Agent, Incoming
from .config import env_bool
from .knowledge import FixStore, fix_id

log = logging.getLogger(__name__)

FIX_VOTE = 0.1        # vote a peer's fix adds to its problem, per zone that verified it
SHARE_TIMEOUT_S = 20
MAX_ZONES = 16        # verified_by entries kept from another zone's fix


@dataclass
class PeerDiagnosis:
    """The local diagnosis combined with the peers' (:meth:`PeerAgent.ask_peers`)."""

    problem: str
    confidence: float
    local: dict[str, Any]
    peers: list[dict[str, Any]] = field(default_factory=list)  # {zone, problem, confidence, verified_fixes}
    fixes: list[dict[str, Any]] = field(default_factory=list)  # fixes the peers verified
    trace: list[dict[str, Any]] = field(default_factory=list)  # the consult step, if the peers were asked

    def summary(self) -> dict[str, Any]:
        return {"problem": self.problem, "confidence": round(self.confidence, 3), "local": self.local,
                "peers": self.peers}


class PeerAgent(Agent):
    """An agent that consults its peers in other zones and shares the fixes it verified. See the module doc."""

    peer_capability: str = ""
    peer_prefix: str = ""
    share_keys: Sequence[str] = ()
    problems: Mapping[str, str] = {}   # label -> one-line description
    healthy: str | None = "healthy"    # the label that needs no fix
    fixes_file: str = ""               # in DATA_DIR; default <peer_prefix>-fixes.json

    def __init__(self, *a: Any, **kw: Any):
        super().__init__(*a, **kw)
        if not (self.peer_capability and self.peer_prefix and self.problems):
            raise TypeError(f"{type(self).__name__} must set peer_capability, peer_prefix and problems")
        p = self.peer_prefix
        peer = {
            f"{p}.consult": "{question, observation}: classify another zone's observation "
                            "-> {zone, answer, diagnosis, fixes}",
            f"{p}.fix": "{fix}: a fix verified by the sending zone, stored but never applied -> {ok, id}",
            "fixes": "{problem?, text?} -> verified fixes known here",
        }
        self.accepts = {**self.accepts, **{k: v for k, v in peer.items() if k not in self.accepts}}

    async def start(self) -> None:
        c, env = self.config, re.sub(r"\W", "_", self.peer_prefix).upper()
        self.threshold = float(c.get(f"{env}_THRESHOLD", "0.75"))
        self.share = env_bool(c.get(f"{env}_SHARE_FIXES"), True)
        self.peer_timeout = float(c.get(f"{env}_PEER_TIMEOUT_S", "30"))
        self.fixes = FixStore(Path(c.get("DATA_DIR", "/data")) / (self.fixes_file or f"{self.peer_prefix}-fixes.json"))

    def status(self) -> dict[str, Any]:
        return {"fixes": len(self.fixes.fixes)}

    # ------------------------------------------------------------------ the domain (subclass)
    async def classify(self, observation: dict[str, Any]) -> dict[str, Any]:
        """Diagnose an observation: ``{"problem": a key of problems, "confidence": 0..1, ...}``.

        Used for the peers' observations, which hold only ``share_keys`` and ``question``."""
        raise NotImplementedError

    def peer_view(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        """All that peers see of an observation: the ``share_keys``."""
        return {k: observation[k] for k in self.share_keys if k in observation}

    # ------------------------------------------------------------------ messages from peers
    async def handle_peer(self, msg: Incoming) -> Any:
        """The reply to a peer message, or None if ``msg`` is not one. Call it first in :meth:`handle`."""
        p = msg.payload if isinstance(msg.payload, dict) else {}
        t = p.get("type")
        if t == f"{self.peer_prefix}.consult":
            return await self.answer_consult(str(p.get("question", "")), p.get("observation"))
        if t == f"{self.peer_prefix}.fix":
            return self.receive_fix(p.get("fix"), msg.from_zone)
        if t == "fixes":
            return self.fixes.search(p.get("problem"), str(p.get("text") or ""))
        return None

    async def answer_consult(self, question: str, observation: Any) -> dict[str, Any]:
        """Classify another zone's observation with our model and add the fixes verified for that problem.
        Nothing runs on this machine: it is not the one with the problem."""
        if not isinstance(observation, dict) or not observation:
            return {"answer": "Send the diagnostic observation.", "diagnosis": None}
        d = await self.classify({**self.peer_view(observation), "question": question})
        return {"zone": self.zone,
                "answer": f"{d['problem']} ({d['confidence']:.0%}): {self.problems.get(d['problem'], '')}",
                "diagnosis": d, "fixes": self.verified_fixes(d["problem"])[:2]}

    def receive_fix(self, fix: Any, zone: str) -> dict[str, Any]:
        """A fix from a peer: kept as verified by ``zone`` only, never applied."""
        clean = self._clean_fix(fix, zone, verified_by=[zone])
        if clean is None:
            return {"ok": False, "error": "not a fix"}
        known = clean["id"] in self.fixes.fixes
        merged = self.fixes.merge(clean)
        if not known:
            log.info("received verified fix from %s: %s", zone, merged["summary"])
        return {"ok": True, "id": merged["id"]}

    # ------------------------------------------------------------------ asking the peers
    async def consult_peers(self, question: str, observation: Mapping[str, Any]) -> list[tuple[str, Any]]:
        """``<prefix>.consult`` to every peer in another zone at once: (zone, reply) pairs."""
        try:
            replies = await self.gw.send_all(self.peer_capability, {
                "type": f"{self.peer_prefix}.consult", "question": question,
                "observation": self.peer_view(observation)}, timeout=self.peer_timeout, other_zones_only=True)
        except Exception as exc:
            log.warning("directory unavailable: %s", exc)
            return []
        return [(e["zone"], r) for e, r in replies]

    async def ask_peers(self, question: str, observation: Mapping[str, Any],
                        local: dict[str, Any]) -> PeerDiagnosis:
        """Combine ``local`` (this agent's diagnosis of ``observation``) with the peers'.

        * sure (confidence >= threshold) and a verified fix known here, or healthy: the peers are not asked;
        * sure, but no verified fix: the peers are asked for their verified fixes only;
        * unsure: every peer votes with its confidence, plus ``FIX_VOTE`` per zone that verified a fix it
          returns. The winner's confidence is its mean support over all voters, this agent included.
        """
        out = PeerDiagnosis(local["problem"], float(local["confidence"]), local)
        unsure = out.confidence < self.threshold
        if not unsure and (out.problem == self.healthy or self.verified_fixes(out.problem)):
            return out
        votes = {out.problem: out.confidence}
        for zone, reply in await self.consult_peers(question, observation):
            d = self._peer_diagnosis(reply)
            if d is None:
                continue
            raw = reply.get("fixes")
            fixes = [f for f in (self._clean_fix(f, zone) for f in raw) if f] if isinstance(raw, list) else []
            out.peers.append({"zone": zone, "problem": d[0], "confidence": d[1], "verified_fixes": len(fixes)})
            out.fixes += fixes
            if unsure:
                votes[d[0]] = votes.get(d[0], 0.0) + d[1]
                for f in fixes:
                    votes[f["problem"]] = votes.get(f["problem"], 0.0) + FIX_VOTE * len(f["verified_by"])
        out.trace.append({"step": "consult_zones", "mode": "vote" if unsure else "fixes_only", "answers": out.peers})
        if unsure:
            out.problem = max(votes, key=votes.__getitem__)
            out.confidence = min(votes[out.problem] / (1 + len(out.peers)), 1.0)
        return out

    # ------------------------------------------------------------------ fixes
    def verified_fixes(self, problem: str) -> list[dict[str, Any]]:
        return [f for f in self.fixes.search(problem) if f["verified_by"]]

    def best_fix(self, problem: str, peer_fixes: Sequence[dict[str, Any]] = ()) -> dict[str, Any] | None:
        """The fix to try: one verified here, else one a peer verified. None: fall back to your own catalog."""
        ours = self.verified_fixes(problem)
        return ours[0] if ours else next((f for f in peer_fixes if f["problem"] == problem), None)

    @staticmethod
    def fix_plan(fix: Mapping[str, Any]) -> dict[str, Any]:
        """A verified fix as a plan: problem, summary, steps and who verified it."""
        return {"problem": fix["problem"], "summary": fix["summary"], "steps": list(fix["steps"]),
                "source": f"verified by {', '.join(fix['verified_by'])}"}

    async def learned(self, problem: str, summary: str, steps: Sequence[str], *, symptoms: str = "",
                      platform: str = "any", evidence: dict[str, Any] | None = None
                      ) -> tuple[dict[str, Any], list[str] | None]:
        """Keep a fix that this agent applied and then verified on its own machine, and send it to the peers.

        Call it only after the checks passed again: peers take it as verified by this zone. Returns the stored
        fix and the zones that accepted it (None when ``<P>_SHARE_FIXES`` is off)."""
        fix = self.fixes.add(problem=problem, summary=summary, steps=list(steps), symptoms=symptoms[:500],
                             verified_by=[self.zone], origin=self.zone, platform=platform, evidence=evidence)
        log.info("verified fix for %s: %s", problem, summary)
        return fix, (await self.share_fix(fix) if self.share else None)

    async def share_fix(self, fix: dict[str, Any]) -> list[str]:
        """``<prefix>.fix`` to every peer in another zone; the zones that accepted it."""
        try:
            replies = await self.gw.send_all(self.peer_capability, {"type": f"{self.peer_prefix}.fix", "fix": fix},
                                             timeout=SHARE_TIMEOUT_S, other_zones_only=True)
        except Exception:
            return []
        return sorted({e["zone"] for e, r in replies if isinstance(r, dict) and r.get("ok")})

    # ------------------------------------------------------------------ checking what other zones send
    def _peer_diagnosis(self, reply: Any) -> tuple[str, float] | None:
        d = reply.get("diagnosis") if isinstance(reply, dict) else None
        if not isinstance(d, dict) or d.get("problem") not in self.problems:
            return None
        try:
            c = float(d.get("confidence"))
        except (TypeError, ValueError):
            return None
        return (d["problem"], min(max(c, 0.0), 1.0)) if math.isfinite(c) else None

    def _clean_fix(self, fix: Any, zone: str, verified_by: list[str] | None = None) -> dict[str, Any] | None:
        """A fix from another zone with its fields checked and cut to size; None if it is not one.
        ``verified_by`` replaces the list the fix claims."""
        if not (isinstance(fix, dict) and fix.get("problem") in self.problems and isinstance(fix.get("steps"), list)):
            return None
        if verified_by is None:
            claimed = fix.get("verified_by")
            claimed = claimed if isinstance(claimed, list) else []
            verified_by = list(dict.fromkeys(str(z)[:64] for z in claimed))[:MAX_ZONES]
        steps = [str(s)[:500] for s in fix["steps"][:20]]
        return {"id": fix_id(fix["problem"], steps), "problem": fix["problem"],
                "summary": str(fix.get("summary", ""))[:300], "steps": steps,
                "symptoms": str(fix.get("symptoms", ""))[:500], "platform": str(fix.get("platform") or "any")[:64],
                "origin": str(fix.get("origin") or zone)[:64], "verified_by": verified_by, "evidence": {}}
