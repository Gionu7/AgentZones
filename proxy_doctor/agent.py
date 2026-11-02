"""The Proxy Doctor agent.

Local question ("pip install fails with SSL error"):
  1. diagnose this machine via the proxy MCP server
  2. classify the observation with the student model (the zone's federated model if enabled)
  3. unsure (confidence < threshold) or no verified fix known? -> ask proxy doctors in peer zones,
     sending only the diagnostic observation (no files, no credentials)
  4. still unsure? -> the local LLM *investigates*: it reasons over the observation and, on a
     real machine, runs extra read-only checks with the MCP tools, then votes like a peer.
     If it concludes "unknown", nothing is applied - the user gets its hypothesis and next checks.
  5. pick the fix from the catalog; peers' signed attestations say where it already worked
  6. apply it (simulated machine: always; real machine: dry-run unless allowed)
  7. re-diagnose; if healthy the fix is *verified*: sign an attestation, store, share with peers
  8. explain the result in plain language with the local LLM (template fallback)

Question from a peer zone: classify the peer's observation with our model and
return our verified fixes for it - we never run diagnostics on our machine for
someone else's problem.

Fixes travel between zones as catalog *templates* with signed attestations;
every zone renders the steps with its own observation (see catalog.validate_fix).
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from agentzones.agent import Agent, AskContext, run_tool_loop
from agentzones.llm import LLMUnavailable

from .catalog import CLASSES, DESCRIPTIONS, FIXES, render_fix, validate_fix
from .features import featurize_many
from .student import ProxyStudentTask, Student, one_hot
from .synth import make_dataset

log = logging.getLogger(__name__)

SHARE_KEYS = ("os", "target", "env", "proxy_url", "proxy_tcp_ok", "external", "external_noverify", "internal",
              "dns_external_ok", "tls", "ca", "clock_skew_s", "tool_errors", "hint_internal_suffix", "simulated")
# read-only tools the investigator may run on a real machine
INVESTIGATION_TOOLS = ("check_proxy_env", "check_https", "inspect_tls", "dns_lookup", "list_problems")

INVESTIGATOR_PROMPT = (
    "You are Proxy Doctor's investigator, running locally. A small classifier and peer zones could not "
    "confidently diagnose a corporate proxy / TLS problem. Reason over the diagnostic observation{tools}.\n"
    "Known problem classes:\n{classes}\n"
    "Key evidence: external vs external_noverify (TLS trust), tls.intercepted, ca.system_has_issuer vs "
    "ca.python_has_issuer, HTTP status 407, proxy_tcp_ok, dns_external_ok, internal checks, clock_skew_s.\n"
    "Finish with ONLY a JSON object: {{\"problem\": <one class name or \"unknown\">, \"confidence\": <0..1>, "
    "\"reasoning\": <2-4 sentences citing evidence>, \"next_checks\": [<what a human should check next>]}}"
)


def sanitize(obs: dict[str, Any]) -> dict[str, Any]:
    """What we are willing to send to peers: diagnostics only."""
    return {k: obs[k] for k in SHARE_KEYS if k in obs}


def parse_json_object(text: str | None) -> dict[str, Any] | None:
    if not text:
        return None
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


class ProxyDoctor(Agent):
    async def start(self) -> None:
        o = self.cfg.options
        self.mcp = o.get("mcp_server", "proxy")
        self.threshold = float(o.get("confidence_threshold", 0.75))
        self.investigate_below = float(o.get("investigate_below", 0.5))
        self.investigate = bool(o.get("investigate_with_llm", True))
        self.auto_apply = bool(o.get("auto_apply", True))
        self.share = bool(o.get("share_fixes", True))
        self.use_llm = bool(o.get("explain_with_llm", True))
        self.zone.fix_validators.append(validate_fix)
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

    # ------------------------------------------------------------------ LLM investigation
    async def _investigate(self, question: str, obs: dict[str, Any], hints: dict[str, Any]) -> dict[str, Any] | None:
        if not (self.investigate and self.zone.llm.enabled):
            return None
        live = not obs.get("simulated")  # on a simulated machine, live checks would contradict the simulation
        tools = [t for t in (self.zone.tools.tools.get(f"{self.mcp}.{n}") for n in INVESTIGATION_TOOLS)
                 if t is not None] if live else []
        system = INVESTIGATOR_PROMPT.format(
            tools=" and run extra read-only checks with your tools if they would help" if tools else "",
            classes="\n".join(f"- {c}: {d}" for c, d in DESCRIPTIONS.items()))
        user = (f"User question: {question}\nObservation: {json.dumps(sanitize(obs), default=str)}\n"
                f"Classifier and peer opinions: {json.dumps(hints, default=str)}")
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        try:
            text, steps, err = await asyncio.wait_for(
                run_tool_loop(self.zone, messages, tools, max_steps=int(self.cfg.options.get("investigate_steps", 6)),
                              json_mode=True),
                timeout=float(self.cfg.options.get("llm_timeout_s", 90)))
        except asyncio.TimeoutError:
            return {"error": "investigation timed out", "checks": []}
        if err:
            return {"error": err, "checks": steps}
        ans = parse_json_object(text)
        if not ans:
            return {"error": "investigator did not return JSON", "raw": (text or "")[:500], "checks": steps}
        problem = ans.get("problem") if ans.get("problem") in CLASSES else "unknown"
        try:
            conf = min(max(float(ans.get("confidence", 0.5)), 0.0), 1.0)
        except (TypeError, ValueError):
            conf = 0.5
        nxt = ans.get("next_checks") if isinstance(ans.get("next_checks"), list) else []
        return {"problem": problem, "confidence": conf, "reasoning": str(ans.get("reasoning", ""))[:1000],
                "next_checks": [str(x)[:200] for x in nxt][:5], "checks": steps}

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
        voters = 1
        known = [f for f in self.zone.fixes.search(local["problem"]) if f["verified_by"]]

        consulted: list[dict[str, Any]] = []
        unsure = local["confidence"] < self.threshold
        if unsure or (local["problem"] != "healthy" and not known):
            # unsure -> peers vote on the diagnosis; confident -> peers are only asked for verified fixes
            answers = await self.zone.ask_peers(question, {"observation": sanitize(obs)}, agent=self.name)
            for a in answers:
                d = a.get("diagnosis")
                if not d or d.get("problem") not in CLASSES:
                    continue
                # peer fixes go through the same gate as pushed fixes: catalog check + signatures
                accepted = 0
                for f in a.get("fixes") or []:
                    if self.zone.on_peer_fix(f, a.get("zone", "?")).get("ok"):
                        accepted += 1
                consulted.append({"zone": a.get("zone"), "problem": d["problem"],
                                  "confidence": float(d.get("confidence", 0)), "verified_fixes": accepted})
                if unsure:
                    votes[d["problem"]] = votes.get(d["problem"], 0.0) + float(d.get("confidence", 0))
                    voters += 1
            trace.append({"step": "ask_peers", "mode": "vote" if unsure else "fixes_only", "answers": consulted})
            if unsure:  # attested fixes add a little weight to their problem
                for p in list(votes):
                    n = max((len(f["verified_by"]) for f in self.zone.fixes.search(p)), default=0)
                    votes[p] += 0.1 * n

        problem = max(votes, key=votes.get)
        confidence = min(votes[problem] / voters, 1.0)  # mean support across voters

        investigation = None
        if confidence < self.investigate_below or context.get("investigate"):
            investigation = await self._investigate(question, obs, {"local": local, "peers": consulted})
            if investigation:
                trace.append({"step": "llm_investigate", **{k: v for k, v in investigation.items() if k != "checks"},
                              "checks": [c["tool"] for c in investigation.get("checks", [])]})
                if investigation.get("problem") in CLASSES:
                    votes[investigation["problem"]] = votes.get(investigation["problem"], 0.0) + investigation["confidence"]
                    voters += 1
                    problem = max(votes, key=votes.get)
                    confidence = min(votes[problem] / voters, 1.0)

        result: dict[str, Any] = {"diagnosis": {"problem": problem, "confidence": round(confidence, 3),
                                                "local": local, "peers": consulted},
                                  "description": DESCRIPTIONS[problem], "trace": trace}
        if investigation:
            result["investigation"] = investigation
            if investigation.get("problem") == "unknown":
                result["diagnosis"]["problem"] = "unknown"
                result["description"] = "Not a known problem class - needs a human. Nothing was changed."
                result["answer"] = await self._explain(question, result)
                return result
        if problem == "healthy":
            result["answer"] = await self._explain(question, result)
            return result

        # the plan always comes from our own catalog, rendered with our own observation;
        # peers contribute *evidence* (signed attestations) about where it worked
        attested = self.zone.fixes.search(problem)
        plan = {**render_fix(problem, obs),
                "source": f"verified by {', '.join(attested[0]['verified_by'])}" if attested else "catalog"}
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
            fix = self.zone.add_verified_fix(problem=problem, summary=FIXES[problem]["summary"],
                                             steps=FIXES[problem]["steps"],
                                             symptoms=" ".join(obs.get("tool_errors", []))[:500],
                                             platform=obs.get("os", "any"),
                                             evidence={"simulated": obs.get("simulated", False)})
            self.zone.event("fix", f"verified fix for {problem}: {plan['summary']} (signed)")
            if self.share:
                result["shared_with"] = await self.zone.share_fix(fix)
        result["answer"] = await self._explain(question, result)
        return result

    async def _explain(self, question: str, result: dict[str, Any]) -> str:
        d = result["diagnosis"]
        fallback = f"Diagnosis: {d['problem']} ({d['confidence']:.0%}). {result['description']}"
        inv = result.get("investigation") or {}
        if inv.get("reasoning"):
            fallback += f"\nLLM investigation: {inv['reasoning']}"
            if inv.get("next_checks"):
                fallback += "\nNext checks:\n  - " + "\n  - ".join(inv["next_checks"])
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
        brief["investigation"] = {k: inv.get(k) for k in ("problem", "reasoning", "next_checks")} if inv else None
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
