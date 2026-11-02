"""``agentzone`` command line.

  agentzone init                       write a starter zone.yaml
  agentzone up -c zone.yaml            run a zone
  agentzone info|peers|tools|fixes     inspect a running zone
  agentzone ask AGENT "question"       ask an agent (in this or, via --zone, another zone)
  agentzone call TOOL '{"arg": 1}'     call a tool directly
  agentzone fl status|train|aggregate|round
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import httpx

STARTER = """\
# Agent Zone config - one file per zone. ${VAR:-default} reads the environment.
name: ${ZONE_NAME:-my-zone}
port: ${ZONE_PORT:-7700}
advertise_url: ${ZONE_URL:-}          # how peers reach this zone, e.g. http://10.1.2.3:7700
tags: [laptop]

model:                                 # any OpenAI-compatible endpoint (Ollama, llama.cpp, vLLM)
  base_url: ${MODEL_URL:-http://localhost:11434/v1}
  model: ${MODEL_NAME:-qwen2.5:1.5b}

discovery:
  peers: []                            # seed URLs of any zones you know, e.g. [http://10.1.2.4:7700]

security:
  network_secret: ${ZONE_SECRET:-}     # same value on every zone of the network
  api_token: ${ZONE_API_TOKEN:-}

mcp_servers:
  - name: proxy
    command: python
    args: [-m, proxy_doctor.mcp_server]
    expose: [list_problems, propose_fix] # tools peers may call

agents:
  - name: proxy-doctor
    kind: python
    impl: proxy_doctor.agent:ProxyDoctor
    description: Diagnoses and fixes corporate proxy and certificate problems
  - name: assistant
    kind: llm
    description: General assistant with this zone's tools
    tools: ["proxy.*"]

federated:
  enabled: true
  task: proxy_doctor.student:ProxyStudentTask
  data: {profile: uniform, n_samples: 400}
"""


def _base(args) -> str:
    return (args.zone or os.environ.get("AGENTZONE_URL") or "http://127.0.0.1:7700").rstrip("/")


def _headers() -> dict[str, str]:
    tok = os.environ.get("ZONE_API_TOKEN")
    return {"Authorization": f"Bearer {tok}"} if tok else {}


def _get(args, path):
    r = httpx.get(_base(args) + path, headers=_headers(), timeout=30)
    r.raise_for_status()
    return r.json()


def _post(args, path, body=None, timeout=600):
    r = httpx.post(_base(args) + path, json=body or {}, headers=_headers(), timeout=timeout)
    r.raise_for_status()
    return r.json()


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, default=str))


def cmd_up(args) -> None:
    import uvicorn

    from .config import load_config
    from .server import create_app

    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    overrides = {"name": args.name, "port": args.port, "advertise_url": args.advertise_url}
    cfg = load_config(args.config, overrides)
    if args.peer:
        cfg.discovery.peers = list(dict.fromkeys(cfg.discovery.peers + args.peer))
    if not cfg.advertise_url and cfg.host in ("127.0.0.1", "localhost"):
        cfg.advertise_url = f"http://127.0.0.1:{cfg.port}"
    if args.host:
        cfg.host = args.host
    app = create_app(cfg)
    uvicorn.run(app, host=cfg.host, port=cfg.port, log_level="warning")


def cmd_init(args) -> None:
    p = Path(args.output)
    if p.exists() and not args.force:
        sys.exit(f"{p} exists (use --force)")
    p.write_text(STARTER)
    print(f"wrote {p}. Start with: agentzone up -c {p}")


def cmd_ask(args) -> None:
    ctx = json.loads(args.context) if args.context else {}
    if args.simulate:
        ctx["simulate"] = args.simulate
    res = _post(args, f"/api/agents/{args.agent}/ask", {"question": args.question, "context": ctx or None})
    if args.json:
        _print(res)
    else:
        print(res.get("answer") or res.get("error"))
        if res.get("trace"):
            print("\n-- trace --")
            for step in res["trace"]:
                print(" ", json.dumps(step, default=str)[:300])


def cmd_call(args) -> None:
    _print(_post(args, "/api/tools/call", {"tool": args.tool, "arguments": json.loads(args.arguments or "{}")}))


def cmd_fl(args) -> None:
    _print(_get(args, "/api/fl") if args.phase == "status" else _post(args, f"/api/fl/{args.phase}"))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="agentzone", description="Peer-to-peer self-hosted AI agent zones")
    ap.add_argument("--zone", help="zone URL for client commands (default $AGENTZONE_URL or http://127.0.0.1:7700)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    up = sub.add_parser("up", help="run a zone")
    up.add_argument("-c", "--config", default=os.environ.get("ZONE_CONFIG", "zone.yaml"))
    up.add_argument("--name")
    up.add_argument("--host")
    up.add_argument("--port", type=int)
    up.add_argument("--advertise-url")
    up.add_argument("--peer", action="append", help="seed peer URL (repeatable)")
    up.set_defaults(fn=cmd_up)

    ini = sub.add_parser("init", help="write a starter zone.yaml")
    ini.add_argument("-o", "--output", default="zone.yaml")
    ini.add_argument("--force", action="store_true")
    ini.set_defaults(fn=cmd_init)

    for name, path in (("info", "/api/info"), ("peers", "/api/peers"), ("tools", "/api/tools"),
                       ("fixes", "/api/fixes"), ("agents", "/api/agents"), ("events", "/api/events")):
        p = sub.add_parser(name, help=f"GET {path}")
        p.set_defaults(fn=lambda a, path=path: _print(_get(a, path)))

    ask = sub.add_parser("ask", help="ask an agent")
    ask.add_argument("agent")
    ask.add_argument("question")
    ask.add_argument("--context", help="JSON object")
    ask.add_argument("--simulate", help="proxy-doctor: simulate a failure class")
    ask.add_argument("--json", action="store_true")
    ask.set_defaults(fn=cmd_ask)

    call = sub.add_parser("call", help="call a tool of the zone")
    call.add_argument("tool")
    call.add_argument("arguments", nargs="?")
    call.set_defaults(fn=cmd_call)

    fl = sub.add_parser("fl", help="federated learning")
    fl.add_argument("phase", choices=["status", "train", "aggregate", "round"])
    fl.set_defaults(fn=cmd_fl)

    args = ap.parse_args(argv)
    try:
        args.fn(args)
    except httpx.HTTPError as exc:
        sys.exit(f"error: {exc}")


if __name__ == "__main__":
    main()
