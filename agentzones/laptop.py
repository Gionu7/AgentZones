"""``agentzone laptop``: a zone on a personal or work laptop, in one process, without Docker.

Why not a container here: Proxy Doctor must diagnose *this* machine (its proxy settings, CA store, tools),
not a container; Ollama installed on the laptop uses the GPU (Apple Silicon, NVIDIA), Ollama in Docker on a
Mac cannot; and many work laptops have no Docker at all.

The laptop is a spoke: it only makes outbound calls to the hub, through a proxy if needed. Without a hub it
runs on its own (standalone). It runs:

* the gateway (local API and dashboard on 127.0.0.1:8080)
* ``proxy-doctor`` - diagnoses this machine; proposes fixes, applies nothing unless PROXY_DOCTOR_ALLOW_APPLY=1
* ``assistant`` - LLM agent on the local model, with Proxy Doctor's tools and the FL request tools
* ``code-assistant`` (``--code DIR``) - answers questions about the code in DIR, read-only; other zones may
  ask it but only get its file tools with ``--share-code``
* ``pd-fl-client`` (``--fl``) - joins federated training of Proxy Doctor's classifier
* the zone MCP server on 127.0.0.1:8000 for Claude Desktop, an IDE or any MCP client

Everything listens on 127.0.0.1 only.
"""

from __future__ import annotations

import asyncio
import contextlib
import getpass
import json
import logging
import os
import re
import socket
import sys
from pathlib import Path
from typing import Any

import uvicorn

from .agent import AgentHost
from .gateway import Gateway, GatewaySettings, local_app

log = logging.getLogger(__name__)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _Server:
    def __init__(self, app, port: int):
        self.server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))

    async def __aenter__(self):
        self.task = asyncio.create_task(self.server.serve())
        while not self.server.started:
            if self.task.done():
                self.task.result()  # raises, e.g. port in use
                raise RuntimeError("server stopped")
            await asyncio.sleep(0.05)
        return self

    async def __aexit__(self, *exc):
        self.server.should_exit = True
        await self.task


def default_zone() -> str:
    try:
        user = getpass.getuser()
    except Exception:
        user = "me"
    return "laptop-" + (re.sub(r"[^a-z0-9-]", "", user.lower()) or "me")


async def run(zone: str, hub_url: str = "", hub_token: str = "", hub_proxy: str = "", port: int = 8080,
              mcp_port: int = 8000, data_dir: str = "~/.agentzones", model_url: str = "http://127.0.0.1:11434/v1",
              model_name: str = "llama3.2:3b", model_source: str = "", llm: bool = True, fl: bool = False,
              setup_model: bool = True, code_dir: str = "", share_code: bool = False) -> None:
    data = Path(os.path.expanduser(data_dir)) / zone
    data.mkdir(parents=True, exist_ok=True)
    gw_url = f"http://127.0.0.1:{port}"
    student = str(data / "proxy-student.npz")
    model_env = {"MODEL_PROVIDER": "openai" if llm else "none", "MODEL_URL": model_url, "MODEL_NAME": model_name}

    settings = GatewaySettings(zone=zone, role="spoke" if hub_url else "standalone", host="127.0.0.1", port=port,
                               hub_url=hub_url.rstrip("/"), hub_token=hub_token, hub_proxy=hub_proxy)
    async with contextlib.AsyncExitStack() as stack:
        await stack.enter_async_context(_Server(local_app(Gateway(settings)), port))

        async def agent(name: str, impl: str, **env: str) -> None:
            p = _free_port()
            host = AgentHost({"AGENT_NAME": name, "AGENT_IMPL": impl, "AGENT_PORT": str(p), "ZONE": zone,
                              "AGENT_URL": f"http://127.0.0.1:{p}", "GATEWAY_URL": gw_url, **env})
            await stack.enter_async_context(_Server(host.app(), p))

        await agent("proxy-doctor", "proxy_doctor.agent:ProxyDoctor", **model_env, PD_PROFILE="dev-laptops",
                    DATA_DIR=str(data / "proxy-doctor"), PD_MODEL_PATH=student)
        if llm:
            zone_mcp = {"name": "zone_mcp", "command": sys.executable, "args": ["-m", "agentzones.cli", "mcp", "--stdio"],
                        "env": {"GATEWAY_URL": gw_url, "AGENT_NAME": "assistant"}}
            proxy_mcp = {"name": "proxy", "command": sys.executable, "args": ["-m", "proxy_doctor.mcp_server"]}
            await agent("assistant", "agentzones.llm_agent:LLMAgent", **model_env, TOOLS="proxy.*,zone_mcp.fl_*",
                        MCP_SERVERS=json.dumps([proxy_mcp, zone_mcp]))
        if llm and code_dir:
            code_mcp = {"name": "code", "command": sys.executable, "args": ["-m", "agentzones.code_mcp"],
                        "env": {"CODE_ROOT": str(Path(os.path.expanduser(code_dir)).resolve())}}
            await agent("code-assistant", "agentzones.llm_agent:LLMAgent", **model_env, TOOLS="code.*", MAX_STEPS="10",
                        AGENT_CAPABILITIES="code", MCP_SERVERS=json.dumps([code_mcp]),
                        AGENT_DESCRIPTION=f"Answers questions about {Path(code_dir).name}/ on {zone} (read-only)",
                        LLM_REMOTE_TOOLS="code.*" if share_code else "",
                        SYSTEM_PROMPT="You answer questions about the code in the code folder. Use list_files and "
                                      "search to find the relevant files, then read_file to read them, and cite file "
                                      "paths and line numbers. Say so when you did not find something. Never invent code.")
        if fl:
            await agent("pd-fl-client", "agentzones.fl.client:FLClient", FL_JOB="proxy-student",
                        FL_TASK="tasks.proxy_student:ProxyStudentClient", PD_PROFILE="dev-laptops",
                        FL_MODEL_OUT=student, DATA_DIR=str(data))
        if mcp_port:
            from .mcp_gateway import build_server, http_app
            await stack.enter_async_context(_Server(http_app(build_server(gw_url, name="mcp")), mcp_port))

        print(f"\nZone {zone} ({settings.role}{' via ' + hub_proxy if hub_proxy else ''}) is running.")
        print(f"  dashboard   {gw_url}/")
        if mcp_port:
            print(f"  MCP         http://127.0.0.1:{mcp_port}/mcp")
        print(f"  model       {model_name if llm else 'none'}{' at ' + model_url if llm else ''}")
        print(f"  data        {data}")
        print("  ask         python -m agentzones.cli ask proxy-doctor \"pip fails with SSL errors\"")
        print("Ctrl+C to stop.\n")
        if llm and setup_model:
            asyncio.create_task(_setup_model(model_name, model_source, model_url, str(data / "models")))
        await asyncio.Event().wait()


async def _setup_model(name: str, source: str, url: str, cache: str) -> None:
    from .models import setup
    try:
        print("  model setup:", await asyncio.to_thread(setup, name, source, url, cache, wait_s=15))
    except Exception as exc:
        print(f"  model setup: {exc}\n  The agents still work; Proxy Doctor explains without the LLM. Start Ollama "
              "(https://ollama.com) or point --model-url at your model server.")


def main(args: Any) -> None:
    env = os.environ
    try:
        asyncio.run(run(zone=args.zone or env.get("ZONE") or default_zone(), hub_url=args.hub or env.get("HUB_URL", ""),
                        hub_token=args.token or env.get("HUB_TOKEN", ""), hub_proxy=args.proxy or env.get("HUB_PROXY", ""),
                        port=args.port, mcp_port=args.mcp_port, data_dir=args.data,
                        model_url=args.model_url or env.get("MODEL_URL", "http://127.0.0.1:11434/v1"),
                        model_name=args.model or env.get("MODEL_NAME", "llama3.2:3b"),
                        model_source=args.model_source or env.get("MODEL_SOURCE", ""),
                        llm=not args.no_llm, fl=args.fl, code_dir=args.code, share_code=args.share_code))
    except KeyboardInterrupt:
        pass
