"""``agentzone`` command line.

Run the parts of a zone (each reads its settings from the environment)::

  agentzone gateway                 ROLE=hub|spoke|standalone, ZONE, HUB_URL, HUB_TOKEN, SPOKE_TOKENS, ...
  agentzone agent                   AGENT_NAME, AGENT_IMPL, GATEWAY_URL, ...
  agentzone mcp [--stdio]           MCP server with list_agents / send_message

Talk to a zone (through its gateway, default http://127.0.0.1:8080 or $GATEWAY_URL)::

  agentzone status | directory [-c CAPABILITY]
  agentzone send ZONE AGENT '{"type": "fl.status"}'
  agentzone ask proxy-doctor "pip fails with 407" [--simulate proxy_auth_required]
  agentzone fl status|start|stop [--job mnist] [--rounds 5] [--reason "..."]
  agentzone fl approve|reject --id ID [--job mnist]        decide on a pending fl.request
  agentzone token                   print a random token for SPOKE_TOKENS / HUB_TOKEN
"""

from __future__ import annotations

import os

# Several agents often share one host; unlimited BLAS threads made FL rounds ~250x slower.
_threads = os.environ.get("AGENT_BLAS_THREADS", "2")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, _threads)

import argparse  # noqa: E402
import asyncio  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import secrets  # noqa: E402
import sys  # noqa: E402

import httpx  # noqa: E402


def _gw(args) -> httpx.Client:
    token = os.environ.get("ZONE_API_TOKEN", "")
    return httpx.Client(base_url=args.gateway.rstrip("/"), timeout=args.timeout + 10, trust_env=False,
                        headers={"authorization": f"Bearer {token}"} if token else {})


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, default=str))


def _send(args, zone: str, agent: str, payload) -> int:
    r = _gw(args).post("/send", json={"from_agent": "cli", "target_zone": zone, "target_agent": agent,
                                      "payload": payload, "timeout": args.timeout})
    try:
        _print(r.json())
    except ValueError:
        print(r.text)
    return 0 if r.status_code < 400 else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="agentzone", description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--gateway", default=os.environ.get("GATEWAY_URL", "http://127.0.0.1:8080"))
    p.add_argument("--timeout", type=float, default=120)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("gateway", help="run the zone gateway")
    sub.add_parser("agent", help="run one agent")
    m = sub.add_parser("mcp", help="run the zone MCP server")
    m.add_argument("--stdio", action="store_true")
    m.add_argument("--host", default=os.environ.get("MCP_HOST", "0.0.0.0"))
    m.add_argument("--port", type=int, default=int(os.environ.get("MCP_PORT", "8000")))

    sub.add_parser("status")
    d = sub.add_parser("directory")
    d.add_argument("-c", "--capability")
    s = sub.add_parser("send")
    s.add_argument("zone")
    s.add_argument("agent")
    s.add_argument("payload", help="JSON")
    a = sub.add_parser("ask")
    a.add_argument("agent")
    a.add_argument("question")
    a.add_argument("--zone", default="")
    a.add_argument("--simulate", help="Proxy Doctor: simulate a scenario instead of checking this machine")
    a.add_argument("--target")
    f = sub.add_parser("fl")
    f.add_argument("action", choices=["status", "start", "stop", "approve", "reject"])
    f.add_argument("--id", help="pending request id (approve / reject)")
    f.add_argument("--reason", default="")
    f.add_argument("--zone", default="")
    f.add_argument("--agent", default="")
    f.add_argument("--job", default="", help="find the aggregator of this job in the directory")
    f.add_argument("--rounds", type=int)
    sub.add_parser("token")

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    if args.cmd == "gateway":
        from .gateway import GatewaySettings, serve
        asyncio.run(serve(GatewaySettings.from_env()))
        return 0
    if args.cmd == "agent":
        from .agent import AgentHost
        asyncio.run(AgentHost().serve())
        return 0
    if args.cmd == "mcp":
        from .mcp_gateway import http_app, main_from_env
        srv = main_from_env()
        if args.stdio:
            srv.run()
        else:
            import uvicorn
            uvicorn.run(http_app(srv), host=args.host, port=args.port, log_level="warning")
        return 0
    if args.cmd == "token":
        print(secrets.token_urlsafe(32))
        return 0
    if args.cmd == "status":
        _print(_gw(args).get("/status").json())
        return 0
    if args.cmd == "directory":
        _print(_gw(args).get("/directory", params={"capability": args.capability} if args.capability else None).json())
        return 0
    if args.cmd == "send":
        return _send(args, args.zone, args.agent, json.loads(args.payload))
    if args.cmd == "ask":
        ctx = {k: v for k, v in (("simulate", args.simulate), ("target", args.target)) if v}
        return _send(args, args.zone, args.agent, {"type": "ask", "question": args.question, "context": ctx})
    if args.cmd == "fl":
        zone, agent = args.zone, args.agent
        if not agent:
            cap = f"fl-aggregator:{args.job}" if args.job else "fl-aggregator:*"
            found = _gw(args).get("/directory", params={"capability": cap}).json()
            if not found:
                print(f"no aggregator with capability {cap} in the directory", file=sys.stderr)
                return 1
            zone, agent = found[0]["zone"], found[0]["agent"]
        payload = {"type": f"fl.{args.action}", **({"rounds": args.rounds} if args.rounds else {}),
                   **({"id": args.id} if args.id else {}), **({"reason": args.reason} if args.reason else {})}
        return _send(args, zone, agent, payload)
    return 2


if __name__ == "__main__":
    sys.exit(main())
