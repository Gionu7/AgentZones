"""The Proxy Doctor agent.

Local question (from an operator or agent in the *same* zone), payload
``{"type": "ask", "question": ..., "context": {...}}``:

  1. diagnose this machine via the proxy MCP server
  2. classify the observation with the student model (the federated one if PD_MODEL_PATH exists)
  3. unsure (confidence < PD_THRESHOLD) or no verified fix known? -> consult the proxy doctors
     of the other zones through the gateway, sending only the diagnostic observation
  4. combine votes, pick a fix (prefer fixes other zones have verified)
  5. apply it (simulated machine: always; real machine: dry-run unless allowed)
  6. re-diagnose; if healthy the fix is *verified*: store it and send it to the other zones
  7. explain the result in plain language with the local LLM (template fallback)

From another zone only ``pd.consult`` (classify *their* observation with our
model, return our verified fixes) and ``pd.fix`` (a verified fix to store,
never auto-applied) are accepted. We never run diagnostics on our machine for
someone else's problem.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

from agentzones.agent import Agent, GatewayError, Incoming
from agentzones.config import MCPServerConfig, model_from_env
from agentzones.knowledge import FixStore
from agentzones.llm import LLM, LLMUnavailable
from agentzones.tools import ToolRegistry

from .catalog import DESCRIPTIONS, render_fix
from .features import featurize_many
from .student import Student, one_hot
from .synth import make_dataset

log = logging.getLogger(__name__)

CAPABILITY = "proxy-doctor"
SHARE_KEYS = ("os", "target", "env", "proxy_url", "proxy_tcp_ok", "external", "external_noverify", "internal",
              "dns_external_ok", "tls", "ca", "clock_skew_s", "tool_errors", "hint_internal_suffix", "simulated")


def sanitize(obs: dict[str, Any]) -> dict[str, Any]:
    """What we are willing to send to other zones: diagnostics only."""
    return {k: obs[k] for k in SHARE_KEYS if k in obs}


def _bool(v: str | None, default: bool) -> bool:
    return default if v is None or v == "" else v.strip().lower() in ("1", "true", "yes", "on")


class ProxyDoctor(Agent):
    capabilities = [CAPABILITY, "chat"]
    description = "Diagnoses and fixes corporate proxy and certificate problems"

    async def start(self) -> None:
        c = self.config
        self.threshold = float(c.get("PD_THRESHOLD", "0.75"))
        self.auto_apply = _bool(c.get("PD_AUTO_APPLY"), False)
        self.share = _bool(c.get("PD_SHARE_FIXES"), True)
        self.use_llm = _bool(c.get("PD_EXPLAIN_WITH_LLM"), True)
        self.peer_timeout = float(c.get("PD_PEER_TIMEOUT_S", "30"))
        self.fixes = FixStore(Path(c.get("DATA_DIR", "/data")) / "fixes.json")
        self.llm = LLM(model_from_env(c))
        self.tools = ToolRegistry()
        await self.tools.start_mcp([MCPServerConfig(name="proxy", command=c.get("PD_MCP_COMMAND", sys.executable),
                                                    args=["-m", "proxy_doctor.mcp_server"],
                                                    env={k: v for k, v in c.items() if k.startswith("PROXY_DOCTOR_")})])
        if "proxy.diagnose" not in self.tools.tools:
            raise RuntimeError("proxy MCP server did not start")
        self.model_path = c.get("PD_MODEL_PATH", "")
        self._model_mtime = 0.0
        self._model_source = "bootstrap"
        cases, labels = make_dataset(int(c.get("PD_BOOTSTRAP_SAMPLES", "300")), c.get("PD_PROFILE", "uniform"),
                                     seed=int(c.get("PD_SEED", "7")))
        self._model = Student()
        self._model.fit(featurize_many(cases), one_hot(labels), epochs=300, lr=0.5)
        self.events: list[str] = []

    async def stop(self) -> None:
        await self.tools.close()
        await self.llm.close()

    @property
    def model(self) -> Student:
        """The bootstrap model, replaced by the federated model whenever PD_MODEL_PATH changes."""
        p = Path(self.model_path) if self.model_path else None
        if p and p.exists() and p.stat().st_mtime != self._model_mtime:
            try:
                with np.load(p, allow_pickle=False) as z:
                    W = z["W"].astype(np.float64)
                if W.shape != self._model.W.shape:
                    raise ValueError(f"shape {W.shape} != {self._model.W.shape}")
                self._model = Student(W)
                self._model_source = str(p)
                log.info("proxy doctor now uses the federated model %s", p)
            except Exception as exc:
                log.warning("cannot load %s: %s", p, exc)
            self._model_mtime = p.stat().st_mtime
        return self._model

    async def _tool(self, name: str, **args: Any) -> Any:
        return await self.tools.call(f"proxy.{name}", args)

    def status(self) -> dict[str, Any]:
        self.model  # picks up a new federated model
        return {"model": self._model_source, "fixes": len(self.fixes.fixes)}

    # ------------------------------------------------------------------ messages
    async def handle(self, msg: Incoming) -> Any:
        p = msg.payload if isinstance(msg.payload, dict) else {"type": "ask", "question": str(msg.payload)}
        t = p.get("type")
        if t == "pd.consult":
            return await self._consult(p.get("question", ""), p.get("observation"))
        if t == "pd.fix":
            return self._receive_fix(p.get("fix") or {}, msg.from_zone)
        if t == "fixes":
            return self.fixes.search(p.get("problem"), p.get("text", ""))
        if t == "status":
            return self.status()
        if t in ("ask", None):
            if not msg.local:  # another zone: advice from the text only, never checks on this machine
                return await self.advise(str(p.get("question", "")), msg.from_zone)
            return await self.ask(str(p.get("question", "")), p.get("context") or {})
        return {"error": f"unknown message type {t!r}", "accepts": ["ask", "pd.consult", "pd.fix", "fixes", "status"]}

    # ------------------------------------------------------------------ other zones
    async def _consult(self, question: str, obs: dict[str, Any] | None) -> dict[str, Any]:
        if not obs:
            return {"answer": "Send the diagnostic observation.", "diagnosis": None}
        d = self.model.classify({**obs, "question": question})
        fixes = [f for f in self.fixes.search(d["problem"]) if f["verified_by"]]
        return {"zone": self.zone, "answer": f"{d['problem']} ({d['confidence']:.0%}): {DESCRIPTIONS[d['problem']]}",
                "diagnosis": d, "fixes": fixes[:2]}

    def _receive_fix(self, fix: dict[str, Any], zone: str) -> dict[str, Any]:
        try:
            if fix.get("problem") not in DESCRIPTIONS or not isinstance(fix.get("steps"), list):
                raise ValueError("not a fix")
            # a zone can only vouch for itself
            fix = {**fix, "verified_by": [zone], "origin": str(fix.get("origin") or zone)[:64],
                   "summary": str(fix.get("summary", ""))[:300], "symptoms": str(fix.get("symptoms", ""))[:500],
                   "steps": [str(s)[:500] for s in fix["steps"][:20]], "evidence": {}}
            known = fix.get("id") in self.fixes.fixes
            merged = self.fixes.merge(fix)
            if not known:
                log.info("received verified fix from %s: %s", zone, merged["summary"])
            return {"ok": True, "id": merged["id"]}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    async def _consult_zones(self, question: str, obs: dict[str, Any]) -> list[dict[str, Any]]:
        try:
            peers = [e for e in await self.gw.directory(CAPABILITY) if e["zone"] != self.zone]
        except Exception as exc:
            log.warning("directory unavailable: %s", exc)
            return []

        async def one(e: dict[str, Any]) -> dict[str, Any]:
            try:
                r = await self.gw.send(e["zone"], e["agent"], {"type": "pd.consult", "question": question,
                                                              "observation": sanitize(obs)}, timeout=self.peer_timeout)
                return {**r, "zone": e["zone"]} if isinstance(r, dict) else {"zone": e["zone"], "error": str(r)}
            except GatewayError as exc:
                return {"zone": e["zone"], "error": str(exc)}
        return list(await asyncio.gather(*(one(e) for e in peers)))

    async def _share_fix(self, fix: dict[str, Any]) -> list[str]:
        try:
            peers = [e for e in await self.gw.directory(CAPABILITY) if e["zone"] != self.zone]
        except Exception:
            return []

        async def one(e: dict[str, Any]) -> str | None:
            try:
                r = await self.gw.send(e["zone"], e["agent"], {"type": "pd.fix", "fix": fix}, timeout=20)
                return e["zone"] if isinstance(r, dict) and r.get("ok") else None
            except GatewayError:
                return None
        return sorted({z for z in await asyncio.gather(*(one(e) for e in peers)) if z})

    async def advise(self, question: str, zone: str) -> dict[str, Any]:
        """A question from another zone: classify the error text in it and share what we know.
        Runs no diagnostics here: this machine is not the one with the problem."""
        d = self.model.classify({"tool_errors": [question], "question": question})
        problem = d["problem"]
        verified = [f for f in self.fixes.search(problem) if f["verified_by"]]
        plan = ({"problem": problem, "summary": verified[0]["summary"], "steps": verified[0]["steps"],
                 "source": f"verified by {', '.join(verified[0]['verified_by'])}"} if verified
                else {**render_fix(problem, {}), "source": "catalog"})
        result: dict[str, Any] = {"mode": "advice", "diagnosis": {**d, "local": d, "peers": []},
                                  "description": DESCRIPTIONS[problem], "fix": plan, "verified": False,
                                  "trace": [{"step": "classify the question text", "by": self.zone, **d}]}
        result["answer"] = await self._explain(question, result) + (
            f"\n\n(Based on the text of your question only: I ran no checks, and this is not your machine. "
            f"For a real diagnosis, ask the proxy-doctor of zone {zone}.)")
        return result

    # ------------------------------------------------------------------ own zone
    async def ask(self, question: str, context: dict[str, Any]) -> dict[str, Any]:
        trace: list[dict[str, Any]] = []
        dargs = {k: context[k] for k in ("target", "internal_url", "simulate") if context.get(k)}
        dargs["error_text"] = context.get("error_text") or question

        obs = await self._tool("diagnose", **dargs, fresh=True)
        obs["question"] = question
        trace.append({"step": "diagnose", "simulated": obs.get("simulated", False),
                      "external_ok": obs["external"]["ok"], "error": obs["external"].get("error")})

        local = self.model.classify(obs)
        trace.append({"step": "classify", "by": self.zone, "model": self._model_source, **local})
        votes: dict[str, float] = {local["problem"]: local["confidence"]}
        known = [f for f in self.fixes.search(local["problem"]) if f["verified_by"]]

        peer_fixes: list[dict[str, Any]] = []
        consulted: list[dict[str, Any]] = []
        unsure = local["confidence"] < self.threshold
        if unsure or (local["problem"] != "healthy" and not known):
            # unsure -> peers vote on the diagnosis; confident -> peers are only asked for verified fixes
            answers = await self._consult_zones(question, obs)
            for a in answers:
                d = a.get("diagnosis")
                if not d:
                    continue
                consulted.append({"zone": a.get("zone"), "problem": d["problem"], "confidence": d["confidence"],
                                  "verified_fixes": len(a.get("fixes") or [])})
                if unsure:
                    votes[d["problem"]] = votes.get(d["problem"], 0.0) + d["confidence"]
                for f in a.get("fixes") or []:
                    if unsure:
                        votes[f["problem"]] = votes.get(f["problem"], 0.0) + 0.1 * len(f.get("verified_by", []))
                    peer_fixes.append(f)
            trace.append({"step": "consult_zones", "mode": "vote" if unsure else "fixes_only", "answers": consulted})

        problem = max(votes, key=votes.get)
        voters = 1 + (len(consulted) if unsure else 0)
        confidence = min(votes[problem] / voters, 1.0)  # mean support across voters
        result: dict[str, Any] = {"diagnosis": {"problem": problem, "confidence": round(confidence, 3),
                                                "local": local, "peers": consulted},
                                  "description": DESCRIPTIONS[problem], "trace": trace}
        if problem == "healthy":
            result["answer"] = await self._explain(question, result)
            return result

        # pick a plan: our verified fix > peer-verified fix > catalog
        candidates = [f for f in self.fixes.search(problem) if f["verified_by"]] or \
                     [f for f in peer_fixes if f["problem"] == problem]
        plan = ({"problem": problem, "summary": candidates[0]["summary"], "steps": candidates[0]["steps"],
                 "source": f"verified by {', '.join(candidates[0]['verified_by'])}"} if candidates
                else {**render_fix(problem, obs), "source": "catalog"})
        result["fix"] = plan

        applied = await self._tool("apply_fix", problem=problem, dry_run=not self.auto_apply,
                                   observation=sanitize(obs))
        trace.append({"step": "apply_fix", "applied": applied.get("applied"),
                      "simulated": applied.get("simulated", False)})
        verified = False
        if applied.get("applied"):
            re_args = {**dargs, "fresh": False}
            re_args.pop("error_text", None)
            obs2 = await self._tool("diagnose", **re_args)
            after = self.model.classify(obs2)
            verified = bool(obs2["external"]["ok"]) and (obs2.get("internal", {}).get("ok") is not False)
            trace.append({"step": "verify", "external_ok": obs2["external"]["ok"], "after": after["problem"],
                          "verified": verified})
        result["verified"] = verified

        if verified:
            fix = self.fixes.add(problem=problem, summary=plan["summary"], steps=plan["steps"],
                                      symptoms=" ".join(obs.get("tool_errors", []))[:500],
                                      verified_by=[self.zone], origin=self.zone,
                                      platform=obs.get("os", "any"),
                                      evidence={"simulated": obs.get("simulated", False)})
            log.info("verified fix for %s: %s", problem, plan["summary"])
            if self.share:
                result["shared_with"] = await self._share_fix(fix)
        result["answer"] = await self._explain(question, result)
        return result

    async def _explain(self, question: str, result: dict[str, Any]) -> str:
        d = result["diagnosis"]
        fallback = f"Diagnosis: {d['problem']} ({d['confidence']:.0%}). {result['description']}"
        if result.get("fix"):
            fallback += f"\nFix ({result['fix']['source']}): {result['fix']['summary']}\n  - " + \
                        "\n  - ".join(result["fix"]["steps"])
            fallback += "\nVerified: fix applied and checks pass." if result.get("verified") else \
                        "\nNot yet verified on this machine (dry run)."
        if d.get("peers"):
            fallback += "\nConsulted zones: " + ", ".join(f"{p['zone']}->{p['problem']}" for p in d["peers"])
        if not (self.use_llm and self.llm.enabled):
            return fallback
        brief = {k: result.get(k) for k in ("diagnosis", "description", "fix", "verified")}
        try:
            reply = await asyncio.wait_for(self.llm.chat([
                {"role": "system", "content": "You are Proxy Doctor. Explain the diagnosis to the user in at "
                 "most 5 short sentences, then list the fix steps exactly as given. Do not invent steps."},
                {"role": "user", "content": f"User question: {question}\nFindings: {json.dumps(brief, default=str)}"},
            ]), timeout=float(self.config.get("PD_LLM_TIMEOUT_S", "60")))
            return reply.content.strip() or fallback
        except (LLMUnavailable, asyncio.TimeoutError, Exception) as exc:  # never fail the diagnosis on the LLM
            log.info("LLM explanation unavailable: %s", exc)
            return fallback
