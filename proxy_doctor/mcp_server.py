"""Proxy Doctor MCP server (stdio).

Run standalone:   python -m proxy_doctor.mcp_server
Use from a zone:  mcp_servers: [{name: proxy, command: python, args: [-m, proxy_doctor.mcp_server]}]
Use from Claude Desktop / VS Code: same command.

Simulation (demos, tests, laptops without problems): pass ``simulate=<class>``
to ``diagnose`` or set PROXY_DOCTOR_SIMULATE=<class>. ``apply_fix`` then
changes the simulated machine, so re-running ``diagnose`` verifies the fix.

Real fixes are only *applied* when PROXY_DOCTOR_ALLOW_APPLY=1, and only
user-level, reversible changes (an env file, pip/git config). Everything
else is returned as steps for a human.
"""

from __future__ import annotations

import os
import random
import shlex
import subprocess
from pathlib import Path
from typing import Any

from agentzones.mcp_lite import MiniMCPServer

from . import diagnostics
from .catalog import CLASSES, DESCRIPTIONS, render_fix
from .synth import make_case

srv = MiniMCPServer("proxy-doctor", instructions="Diagnose and fix corporate proxy / TLS interception problems.")

# simulated machine state: {"scenario": class, "fixed": bool, "seed": int}
SIM: dict[str, Any] = {"scenario": None, "fixed": False, "seed": 0}
ENV_FILE = Path(os.environ.get("PROXY_DOCTOR_ENV_FILE", Path.home() / ".config/agentzones/proxy.env"))


def _sim_obs() -> dict[str, Any]:
    rng = random.Random(SIM["seed"])
    obs = make_case("healthy" if SIM["fixed"] else SIM["scenario"], rng)
    obs.update({"simulated": True, "scenario": SIM["scenario"], "fixed": SIM["fixed"]})
    return obs


@srv.tool()
def diagnose(target: str = "https://pypi.org", internal_url: str = "", error_text: str = "",
             simulate: str = "", fresh: bool = True) -> dict:
    """Run all proxy/TLS checks against target (and optionally an internal URL).
    error_text: the error message the user saw. simulate: a failure class to simulate
    (demo mode). fresh=false re-checks the current simulated machine (after a fix)."""
    scenario = simulate or os.environ.get("PROXY_DOCTOR_SIMULATE", "")
    if scenario:
        if scenario not in CLASSES:
            raise ValueError(f"unknown scenario {scenario}; choose from {CLASSES}")
        if fresh or SIM["scenario"] != scenario:
            SIM.update(scenario=scenario, fixed=False, seed=random.randrange(1 << 30))
        obs = _sim_obs()
        if error_text and not SIM["fixed"]:
            obs["tool_errors"].append(error_text)
        return obs
    return diagnostics.diagnose(target, internal_url, error_text)


@srv.tool()
def check_proxy_env() -> dict:
    """Show proxy-related environment variables (credentials redacted) and CA bundle overrides."""
    return diagnostics.proxy_env()


@srv.tool()
def check_https(url: str, verify: bool = True) -> dict:
    """HEAD request to url through the configured proxy. verify=false skips TLS verification (diagnosis only)."""
    import ssl
    return diagnostics.https_get(url, ssl.create_default_context() if verify else diagnostics._noverify_ctx())


@srv.tool()
def inspect_tls(host: str) -> dict:
    """Return the issuer of the certificate presented for host and whether TLS looks intercepted."""
    issuer = diagnostics.peer_issuer(host)
    return {"host": host, "issuer": issuer, "intercepted": diagnostics.intercepted(issuer)}


@srv.tool()
def dns_lookup(host: str) -> dict:
    """Check whether host resolves locally."""
    return {"host": host, "resolves": diagnostics.dns_check(host)}


@srv.tool()
def list_problems() -> dict:
    """List the failure classes the doctor knows, with descriptions."""
    return DESCRIPTIONS


@srv.tool()
def propose_fix(problem: str, observation: dict | None = None) -> dict:
    """Return the fix plan (ordered steps) for a failure class."""
    if problem not in CLASSES:
        raise ValueError(f"unknown problem {problem}")
    return render_fix(problem, observation or {})


def _apply_real(problem: str, plan: dict) -> dict:
    safe_cmds, env_lines, manual = [], [], []
    for step in plan["steps"]:
        if step.startswith("export "):
            env_lines.append(step)
        elif step.startswith(("pip config set", "git config --global")):
            safe_cmds.append(step)
        else:
            manual.append(step)
    done = []
    if env_lines:
        ENV_FILE.parent.mkdir(parents=True, exist_ok=True)
        existing = ENV_FILE.read_text().splitlines() if ENV_FILE.exists() else []
        ENV_FILE.write_text("\n".join(dict.fromkeys(existing + env_lines)) + "\n")
        done.append(f"wrote {ENV_FILE} (source it from your shell profile)")
    for cmd in safe_cmds:
        try:
            subprocess.run(shlex.split(cmd), check=True, capture_output=True, timeout=30)
            done.append(cmd)
        except Exception as exc:
            manual.append(f"{cmd}   # failed automatically: {exc}")
    return {"applied": bool(done), "done": done, "manual_steps": manual}


@srv.tool()
def apply_fix(problem: str, dry_run: bool = True, observation: dict | None = None) -> dict:
    """Apply the fix for a failure class. dry_run=true only returns what would be done.
    On a simulated machine the fix changes the simulation, so diagnose(fresh=false) can verify it."""
    plan = render_fix(problem, observation or {})
    if SIM["scenario"] and (observation or {}).get("simulated", True):
        if not dry_run and problem == SIM["scenario"]:
            SIM["fixed"] = True
        return {"simulated": True, "applied": not dry_run, "plan": plan}
    if dry_run or os.environ.get("PROXY_DOCTOR_ALLOW_APPLY") != "1":
        return {"applied": False, "dry_run": True, "plan": plan,
                "note": "set PROXY_DOCTOR_ALLOW_APPLY=1 to let the doctor apply user-level changes"}
    return {"plan": plan, **_apply_real(problem, plan)}


if __name__ == "__main__":
    srv.run()
