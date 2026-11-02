"""The Proxy Doctor agent.

Local question ("pip install fails with SSL error"):
  1. diagnose this machine via the proxy MCP server
  2. classify the observation with the student model (the zone's federated model if enabled)
  3. unsure (confidence < threshold) or no verified fix known? -> ask proxy doctors in peer zones,
     sending only the diagnostic observation (no files, no credentials)
  4. combine votes, pick a fix (prefer fixes other zones have verified)
  5. apply it (simulated machine: always; real machine: dry-run unless allowed)
  6. re-diagnose; if healthy the fix is *verified*: store it and share it with peers
  7. explain the result in plain language with the local LLM (template fallback)

Question from a peer zone: classify the peer's observation with our model and
return our verified fixes for it - we never run diagnostics on our machine for
someone else's problem.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from agentzones.agent import Agent, AskContext
from agentzones.llm import LLMUnavailable

from .catalog import DESCRIPTIONS, render_fix
from .student import ProxyStudentTask, Student, one_hot
from .features import featurize_many
from .synth import make_dataset

log = logging.getLogger(__name__)

SHARE_KEYS = ("os", "target", "env", "proxy_url", "proxy_tcp_ok", "external", "external_noverify", "internal",
              "dns_external_ok", "tls", "ca", "clock_skew_s", "tool_errors", "hint_internal_suffix", "simulated")


def sanitize(obs: dict[str, Any]) -> dict[str, Any]:
    """What we are willing to send to peers: diagnostics only."""
    return {k: obs[k] for k in SHARE_KEYS if k in obs}


class ProxyDoctor(Agent):
    async def start(self) -> None:
        o = self.cfg.options
        self.mcp = o.get("mcp_server", "proxy")
        self.threshold = float(o.get("confidence_threshold", 0.75))
        self.auto_apply = bool(o.get("auto_apply", True))
        self.share = bool(o.get("share_fixes", True))
        self.use_llm = bool(o.get("explain_with_llm", True))
        self._own_model: Student | None = None
        if not self._fl_model():
            path = o.get("model_path")
            if path and Path(path).exists():
                self._own_model = Student.load(path)
            else:  # bootstrap from synthetic cases of this site's profile
                cases, labels = make_dataset(int(o.get("bootstrap_samples", 300)),
                                             o.get("profile", "uniform"), seed=int(o.get("seed", 7)))
                self._own_model = Student()
                self._own_model.fit(featurize_many(cases), one_hot(labels), epochs=300, lr=0.5)

    def _fl_model(self) -> Student | None:
        fl = self.zone.fl
        if fl and isinstance(fl.task, ProxyStudentTask):
            return fl.task.model
        return None

    @property
    def model(self) -> Student:
        return self._fl_model() or self._own_model  # type: ignore[return-value]

    async def _tool(self, name: str, **args: Any) -> Any:
        return await self.zone.tools.call(f"{self.mcp}.{name}", args)

    # ------------------------------------------------------------------ peer side
    async def _answer_peer(self, question: str, context: dict[str, Any] | None) -> dict[str, Any]:
        obs = (context or {}).get("observation")
        if not obs:
            return {"answer": "Send the diagnostic observation in context.observation.", "diagnosis": None}
        obs = {**obs, "question": question}
        d = self.model.classify(obs)
        fixes = [f for f in self.zone.fixes.search(d["problem"]) if f["verified_by"]]
        return {"answer": f"{d['problem']} ({d['confidence']:.0%}): {DESCRIPTIONS[d['problem']]}",
                "diagnosis": d, "fixes": fixes[:2]}

    # ------------------------------------------------------------------ local side
    async def ask(self, question: str, context: dict[str, Any] | None, ctx: AskContext) -> dict[str, Any]:
        if ctx.origin != "local":
            return await self._answer_peer(question, context)
        context = context or {}
        trace: list[dict[str, Any]] = []
        dargs = {k: context[k] for k in ("target", "internal_url", "simulate") if context.get(k)}
        dargs["error_text"] = context.get("error_text") or question

        obs = await self._tool("diagnose", **dargs, fresh=True)
        obs["question"] = question
        trace.append({"step": "diagnose", "simulated": obs.get("simulated", False),
                      "external_ok": obs["external"]["ok"], "error": obs["external"].get("error")})

        local = self.model.classify(obs)
        trace.append({"step": "classify", "by": self.zone.name, **local})
        votes: dict[str, float] = {local["problem"]: local["confidence"]}
        known = [f for f in self.zone.fixes.search(local["problem"]) if f["verified_by"]]

        peer_fixes: list[dict[str, Any]] = []
        consulted: list[dict[str, Any]] = []
        unsure = local["confidence"] < self.threshold
        if unsure or (local["problem"] != "healthy" and not known):
            # unsure -> peers vote on the diagnosis; confident -> peers are only asked for verified fixes
            answers = await self.zone.ask_peers(question, {"observation": sanitize(obs)}, agent=self.name)
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
            trace.append({"step": "ask_peers", "mode": "vote" if unsure else "fixes_only", "answers": consulted})

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
        candidates = [f for f in self.zone.fixes.search(problem) if f["verified_by"]] or \
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
            fix = self.zone.fixes.add(problem=problem, summary=plan["summary"], steps=plan["steps"],
                                      symptoms=" ".join(obs.get("tool_errors", []))[:500],
                                      verified_by=[self.zone.name], origin=self.zone.name,
                                      platform=obs.get("os", "any"),
                                      evidence={"simulated": obs.get("simulated", False)})
            self.zone.event("fix", f"verified fix for {problem}: {plan['summary']}")
            if self.share:
                result["shared_with"] = await self.zone.share_fix(fix)
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
        if not (self.use_llm and self.zone.llm.enabled):
            return fallback
        brief = {k: result.get(k) for k in ("diagnosis", "description", "fix", "verified")}
        try:
            reply = await asyncio.wait_for(self.zone.llm.chat([
                {"role": "system", "content": "You are Proxy Doctor. Explain the diagnosis to the user in at "
                 "most 5 short sentences, then list the fix steps exactly as given. Do not invent steps."},
                {"role": "user", "content": f"User question: {question}\nFindings: {json.dumps(brief, default=str)}"},
            ]), timeout=float(self.cfg.options.get("llm_timeout_s", 60)))
            return reply.content.strip() or fallback
        except (LLMUnavailable, asyncio.TimeoutError, Exception) as exc:  # never fail the diagnosis on the LLM
            log.info("LLM explanation unavailable: %s", exc)
            return fallback
