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

Steps 3, 4 and 6 and the ``pd.*`` messages are the generic
:class:`agentzones.peers.PeerAgent`; this module is the proxy-specific part.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

from agentzones.agent import Incoming
from agentzones.config import MCPServerConfig, env_bool, model_from_env
from agentzones.llm import LLM, LLMUnavailable
from agentzones.peers import PeerAgent
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


class ProxyDoctor(PeerAgent):
    capabilities = [CAPABILITY, "chat"]
    description = "Diagnoses and fixes corporate proxy and certificate problems"
    accepts = {
        "ask": "{question, context?: {target, internal_url, error_text, simulate}}: own zone only -> diagnosis, fix",
        "status": "{} -> model, fixes",
    }
    # peers: the proxy doctors of the other zones (pd.consult, pd.fix, PD_THRESHOLD, PD_SHARE_FIXES, ...)
    peer_capability = CAPABILITY
    peer_prefix = "pd"
    share_keys = SHARE_KEYS
    problems = DESCRIPTIONS
    healthy = "healthy"
    fixes_file = "fixes.json"

    async def start(self) -> None:
        await super().start()
        c = self.config
        self.auto_apply = env_bool(c.get("PD_AUTO_APPLY"), False)
        self.use_llm = env_bool(c.get("PD_EXPLAIN_WITH_LLM"), True)
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

    async def classify(self, observation: dict[str, Any]) -> dict[str, Any]:
        return self.model.classify(observation)

    async def _tool(self, name: str, **args: Any) -> Any:
        return await self.tools.call(f"proxy.{name}", args)

    def status(self) -> dict[str, Any]:
        self.model  # picks up a new federated model
        return {"model": self._model_source, **super().status()}

    # ------------------------------------------------------------------ messages
    async def handle(self, msg: Incoming) -> Any:
        if (reply := await self.handle_peer(msg)) is not None:  # pd.consult, pd.fix, fixes
            return reply
        p = msg.payload if isinstance(msg.payload, dict) else {"type": "ask", "question": str(msg.payload)}
        t = p.get("type")
        if t == "status":
            return self.status()
        if t in ("ask", None):
            if not msg.local:  # another zone: advice from the text only, never checks on this machine
                return await self.advise(str(p.get("question", "")), msg.from_zone)
            return await self.ask(str(p.get("question", "")), p.get("context") or {})
        return self.unknown(t)

    # ------------------------------------------------------------------ other zones
    async def advise(self, question: str, zone: str) -> dict[str, Any]:
        """A question from another zone: classify the error text in it and share what we know.
        Runs no diagnostics here: this machine is not the one with the problem."""
        d = self.model.classify({"tool_errors": [question], "question": question})
        problem = d["problem"]
        fix = self.best_fix(problem)
        plan = self.fix_plan(fix) if fix else {**render_fix(problem, {}), "source": "catalog"}
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
        # unsure -> the other zones' doctors vote; sure but no verified fix -> they are asked for theirs
        d = await self.ask_peers(question, obs, local)
        trace += d.trace
        problem = d.problem
        result: dict[str, Any] = {"diagnosis": d.summary(), "description": DESCRIPTIONS[problem], "trace": trace}
        if problem == "healthy":
            result["answer"] = await self._explain(question, result)
            return result

        # pick a plan: our verified fix > peer-verified fix > catalog
        fix = self.best_fix(problem, d.fixes)
        plan = self.fix_plan(fix) if fix else {**render_fix(problem, obs), "source": "catalog"}
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

        if verified:  # store it and send it to the other zones' doctors
            _, shared = await self.learned(problem, plan["summary"], plan["steps"],
                                           symptoms=" ".join(obs.get("tool_errors", [])),
                                           platform=obs.get("os", "any"),
                                           evidence={"simulated": obs.get("simulated", False)})
            if shared is not None:
                result["shared_with"] = shared
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
