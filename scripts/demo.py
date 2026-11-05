#!/usr/bin/env python3
"""Three agent zones on one machine, no Docker needed:

    python scripts/demo.py           run the story and exit
    python scripts/demo.py --keep    leave the zones running afterwards

Topology (the same as zones/*/zone.yml):
    az  HUB      the only zone that accepts connections
    cn  spoke    outbound only
    b   spoke    outbound only, and only through an HTTP proxy

Story
  1. Hub and spoke   b and cn register with the hub; b talks to cn through the hub, via its proxy.
  2. Local fix       b hits "407 Proxy Authentication Required", fixes it, verifies, sends the fix out.
  3. Consult         az hits the same problem, is unsure, consults the other zones and reuses b's fix.
  4. Federated       cn starts a run of the aggregator in b; clients in all three zones train the
                     Proxy Doctor model on their own data; no data leaves a zone.
  5. LLM request     an assistant asks for another run through the MCP tools; it waits until an
                     operator approves it, and everything lands in the audit log.
  6. Security        a spoke with the wrong token is turned away.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import sys
import tempfile
from pathlib import Path

import httpx
import uvicorn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agentzones.agent import AgentHost  # noqa: E402
from agentzones.devproxy import ForwardProxy, free_port  # noqa: E402
from agentzones.gateway import Gateway, GatewaySettings, HubClient, hub_app, local_app  # noqa: E402

B, DIM, G, R, Y, X = "\033[1m", "\033[2m", "\033[32m", "\033[31m", "\033[33m", "\033[0m"
TOKENS = {"b": "demo-token-b-4f1c2a", "cn": "demo-token-cn-9e7d3b"}
PROFILES = {"az": "dev-laptops", "b": "build-servers", "cn": "k8s-cluster"}


def h(title: str) -> None:
    print(f"\n{B}=== {title} ==={X}")


class Server:
    def __init__(self, app, port: int):
        self.port = port
        self.server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))

    async def __aenter__(self):
        self.task = asyncio.create_task(self.server.serve())
        while not self.server.started:
            await asyncio.sleep(0.02)
        return self

    async def __aexit__(self, *exc):
        self.server.should_exit = True
        await self.task


def port_or_free(port: int) -> int:
    import socket
    with socket.socket() as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # like uvicorn: TIME_WAIT is fine
        try:
            s.bind(("127.0.0.1", port))
            return port
        except OSError:
            return free_port()


async def gateway(stack, port: int = 0, hub_port: int = 0, **kw) -> tuple[Gateway, str, str | None]:
    s = GatewaySettings(host="127.0.0.1", port=port_or_free(port) if port else free_port(),
                        hub_port=port_or_free(hub_port) if hub_port else free_port(), poll_wait_s=10,
                        register_interval_s=2, **kw)
    gw = Gateway(s)
    await stack.enter_async_context(Server(local_app(gw), s.port))
    hub = None
    if gw.hub:
        await stack.enter_async_context(Server(hub_app(gw), s.hub_port))
        hub = f"http://127.0.0.1:{s.hub_port}"
    return gw, f"http://127.0.0.1:{s.port}", hub


async def agent(stack, gw_url: str, name: str, impl: str, **env) -> None:
    port = free_port()
    host = AgentHost({"AGENT_NAME": name, "AGENT_IMPL": impl, "AGENT_PORT": str(port),
                      "AGENT_URL": f"http://127.0.0.1:{port}", "GATEWAY_URL": gw_url, "REGISTER_INTERVAL_S": "2",
                      **env})
    await stack.enter_async_context(Server(host.app(), port))


async def main(keep: bool, rounds: int, llm: bool = False) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    work = Path(tempfile.mkdtemp(prefix="agentzones-demo-"))
    ok = True
    async with contextlib.AsyncExitStack() as stack, httpx.AsyncClient(timeout=120) as http:
        async def send(gw_url, zone, name, payload, timeout=60):
            r = await http.post(gw_url + "/send", json={"target_zone": zone, "target_agent": name,
                                                       "payload": payload, "timeout": timeout})
            return r.json()

        proxy = await stack.enter_async_context(ForwardProxy())
        gws, urls = {}, {}
        fixed = {"az": 8080, "b": 8081, "cn": 8082, "hub": 9090} if keep else {}  # same ports as the Docker zones
        gws["az"], urls["az"], hub_url = await gateway(stack, fixed.get("az", 0), fixed.get("hub", 0), zone="az",
                                                       role="hub", spoke_tokens=TOKENS)
        gws["cn"], urls["cn"], _ = await gateway(stack, fixed.get("cn", 0), zone="cn", role="spoke", hub_url=hub_url,
                                                 hub_token=TOKENS["cn"])
        gws["b"], urls["b"], _ = await gateway(stack, fixed.get("b", 0), zone="b", role="spoke", hub_url=hub_url,
                                               hub_token=TOKENS["b"], hub_proxy=proxy.url)
        for z in ("az", "cn", "b"):
            model = work / z / "proxy-student.npz"
            await agent(stack, urls[z], "echo", "agentzones.agent:EchoAgent")
            await agent(stack, urls[z], "proxy-doctor", "proxy_doctor.agent:ProxyDoctor",
                        DATA_DIR=str(work / z), PD_AUTO_APPLY="1", PD_EXPLAIN_WITH_LLM="0", MODEL_PROVIDER="none",
                        PD_PROFILE=PROFILES[z], PD_MODEL_PATH=str(model))
            await agent(stack, urls[z], "pd-fl-client", "agentzones.fl.client:FLClient", ZONE=z,
                        FL_JOB="proxy-student", FL_TASK="tasks.proxy_student:ProxyStudentClient",
                        PD_PROFILE=PROFILES[z], FL_MODEL_OUT=str(model))
        if llm:  # LLM agents in az, on MODEL_URL (default: Ollama on this machine)
            import os
            model = {"MODEL_URL": os.environ.get("MODEL_URL", "http://127.0.0.1:11434/v1"),
                     "MODEL_NAME": os.environ.get("MODEL_NAME", "llama3.2:3b")}
            zone_mcp = {"name": "zone_mcp", "command": sys.executable, "args": ["-m", "agentzones.cli", "mcp", "--stdio"],
                        "env": {"GATEWAY_URL": urls["az"], "AGENT_NAME": "assistant"}}
            await agent(stack, urls["az"], "assistant", "agentzones.llm_agent:LLMAgent", **model,
                        MCP_SERVERS=json.dumps([zone_mcp]), TOOLS="zone.*,zone_mcp.fl_*")
            code_mcp = {"name": "code", "command": sys.executable, "args": ["-m", "agentzones.code_mcp"], "env": {"CODE_ROOT": str(ROOT)}}
            await agent(stack, urls["az"], "code-assistant", "agentzones.llm_agent:LLMAgent", **model, TOOLS="code.*",
                        MCP_SERVERS=json.dumps([code_mcp]), AGENT_CAPABILITIES="code", MAX_STEPS="10",
                        AGENT_DESCRIPTION="Answers questions about the Agent Zones code (read-only)",
                        SYSTEM_PROMPT="You answer questions about the code in the code folder. Use search and read_file, "
                                      "and cite file paths and line numbers.")
        await agent(stack, urls["b"], "pd-aggregator", "agentzones.fl.aggregator:FLAggregator",
                    FL_JOB="proxy-student", FL_STRATEGY="tasks.proxy_student:ProxyStudentStrategy",
                    FL_ROUNDS=str(rounds), FL_MIN_CLIENTS="3", FL_AUTOSTART="0", FL_POLL_S="0.5", FL_MAX_ROUNDS="20",
                    FL_MODEL_DIR=str(work / "b" / "models"))

        h("1. Hub and spoke")
        for _ in range(100):
            d = (await http.get(urls["b"] + "/directory")).json()
            if len({e["zone"] for e in d}) == 3 and len(d) >= 10:
                break
            await asyncio.sleep(0.2)
        for z in ("az", "cn", "b"):
            names = sorted(e["agent"] for e in d if e["zone"] == z)
            role = {"az": "hub", "cn": "spoke", "b": "spoke via proxy"}[z]
            print(f"  {B}{z:<3}{X} {role:<16} {', '.join(names)}  {DIM}(local API {urls[z]}){X}")
        r = await send(urls["b"], "cn", "echo", {"hello": "from b"})
        print(f"  b -> hub -> cn -> hub -> b: {G}{r['echo']}{X} answered by zone {r['zone']}")
        print(f"  requests through b's proxy so far: {proxy.requests}; connections into b or cn: {G}none{X}")

        h("2. b fixes a 407 locally and sends the verified fix to the other zones")
        r = await send(urls["b"], "b", "proxy-doctor", {"type": "ask", "question": "407 Proxy Authentication Required",
                                                         "context": {"simulate": "proxy_auth_required"}})
        d = r["diagnosis"]
        print(f"  diagnosis {d['problem']} ({d['confidence']:.0%}), verified={r['verified']}, "
              f"sent to {', '.join(r.get('shared_with') or []) or '-'}")
        ok &= bool(r["verified"])

        h("3. az is unsure about the same problem and consults the other zones")
        r = await send(urls["az"], "az", "proxy-doctor", {"type": "ask", "question": "pip: 407 Proxy Authentication Required",
                                                           "context": {"simulate": "proxy_auth_required"}})
        d = r["diagnosis"]
        print(f"  own model: {d['local']['problem']} ({d['local']['confidence']:.0%})")
        for p in d["peers"]:
            print(f"  zone {p['zone']:<3} says {p['problem']} ({p['confidence']:.0%}), verified fixes: {p['verified_fixes']}")
        print(f"  result {B}{d['problem']}{X}, fix {r['fix']['source']}, verified on az: {r['verified']}")
        ok &= d["problem"] == "proxy_auth_required" and bool(r["verified"])

        h(f"4. Federated learning: cn starts the aggregator in b, {rounds} rounds, clients in az, b, cn")
        r = await send(urls["cn"], "b", "pd-aggregator", {"type": "fl.start"})
        print(f"  fl.start -> {r['state']}")
        seen = 0
        while True:
            st = await send(urls["az"], "b", "pd-aggregator", {"type": "fl.status"})
            for rec in st["history"][seen:]:
                bar = "#" * int(rec["global_acc"] * 40)
                print(f"  round {rec['round']}: benchmark accuracy {rec['global_acc']:.1%} {G}{bar}{X} "
                      f"{DIM}({', '.join(rec['clients'])}){X}")
            seen = len(st["history"])
            if st["state"] in ("done", "failed", "stopped"):
                break
            await asyncio.sleep(0.3)
        print(f"  state: {st['state']}; model saved to {work / 'b' / 'models' / 'proxy-student-latest.npz'}")
        ok &= st["state"] == "done"
        await asyncio.sleep(0.5)
        for z in ("az", "cn", "b"):
            s = await send(urls[z], z, "proxy-doctor", {"type": "status"})
            print(f"  proxy-doctor@{z} now uses {DIM}{s['model']}{X}")

        h("5. An LLM agent asks for more training (MCP tools); an operator decides")
        from agentzones.mcp_gateway import build_server
        mcp = build_server(urls["az"], name="assistant")  # what an LLM agent in az sees as MCP tools

        def tool(name, **args):
            res = mcp.call(name, args)
            return json.loads(res["content"][0]["text"]) if not res["isError"] else {"error": res["content"][0]["text"]}
        req = await asyncio.to_thread(tool, "fl_request_start", job="proxy-student", rounds=500,
                                      reason="12 new verified 407 incidents since the last run")
        print(f"  assistant@az: fl_request_start(rounds=500) -> pending {req['pending']}, state still {req['state']}")
        r = await http.post(urls["az"] + "/send", json={"from_agent": "assistant", "target_zone": "b",
                                                       "target_agent": "pd-aggregator", "payload": {"type": "fl.start"}})
        print(f"  assistant@az sends fl.start directly -> {Y}{r.json()['error'][:60]}…{X}")
        r = await send(urls["az"], "b", "pd-aggregator", {"type": "fl.approve", "id": req["pending"]})
        print(f"  operator approves -> {G}run {r['run']} started{X}, rounds capped at {r['rounds']} ({r.get('note', '')})")
        while (await send(urls["az"], "b", "pd-aggregator", {"type": "fl.status"}))["state"] != "done":
            await asyncio.sleep(0.3)
        st = await send(urls["az"], "b", "pd-aggregator", {"type": "fl.status"})
        for a in st["audit"][-5:]:
            print(f"  {DIM}audit{X} {a['who']:<22} {a['action']:<16} {a['outcome']}")
        if keep:  # leave one waiting, so it can be approved on the dashboard
            await asyncio.to_thread(tool, "fl_request_start", job="proxy-student", rounds=10,
                                    reason="nightly retrain (approve me on the dashboard)")

        h("6. Security")
        bad = HubClient(GatewaySettings(zone="b", role="spoke", hub_url=hub_url, hub_token="stolen-or-wrong"))
        try:
            await bad.call("GET", "/hub/directory")
            print(f"  {R}wrong token accepted!{X}")
            ok = False
        except PermissionError as exc:
            print(f"  spoke with a wrong token: {G}rejected{X} {DIM}({exc}){X}")
        await bad.close()
        print("  tokens never travel on the wire (HMAC per request); the hub signs every answer")

        print(f"\n{G if ok else R}{B}demo {'passed' if ok else 'FAILED'}{X}")
        if keep:
            print(f"\n{B}Hub dashboard: {urls['az']}/{X}   (spokes: {urls['b']}/ and {urls['cn']}/)")
            print(f"\nZones keep running. Try:\n  curl -s {urls['cn']}/send -H 'content-type: application/json' "
                  "-d '{\"target_zone\":\"b\",\"target_agent\":\"pd-aggregator\",\"payload\":{\"type\":\"fl.status\"}}'"
                  f"\n  python -m agentzones.cli --gateway {urls['b']} directory\nCtrl+C to stop.")
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.Event().wait()
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--keep", action="store_true", help="leave the zones running")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--llm", action="store_true", help="add LLM agents in az (assistant, code-assistant) on MODEL_URL")
    a = ap.parse_args()
    try:
        sys.exit(asyncio.run(main(a.keep, a.rounds, a.llm)))
    except KeyboardInterrupt:
        pass
