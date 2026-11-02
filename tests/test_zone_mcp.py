"""The zone as an MCP server: HTTP transport, stdio bridge, routing to peer zones."""

import asyncio
import json
import sys

import httpx

from agentzones.config import ZoneConfig
from agentzones.mcp_lite import HttpMCPClient, StdioMCPClient
from agentzones.server import create_app
from conftest import ROOT, free_port


def cfg(name, port, seeds, tmp_path):
    return ZoneConfig.model_validate({
        "name": name, "host": "127.0.0.1", "port": port, "advertise_url": f"http://127.0.0.1:{port}",
        "data_dir": str(tmp_path / name), "model": {"provider": "none"},
        "discovery": {"peers": seeds, "gossip_interval_s": 0.3},
        "security": {"network_secret": "s"},
        "mcp_servers": [{"name": "proxy", "command": sys.executable, "args": ["-m", "proxy_doctor.mcp_server"],
                         "expose": ["list_problems"]}],
        "agents": [{"name": "proxy-doctor", "kind": "python", "impl": "proxy_doctor.agent:ProxyDoctor",
                    "options": {"explain_with_llm": False}}],
    })


async def test_zone_mcp_http_and_stdio(tmp_path, serve):
    pa, pb = free_port(), free_port()
    ua = f"http://127.0.0.1:{pa}"
    async with serve(create_app(cfg("zone-a", pa, [], tmp_path)), pa), \
               serve(create_app(cfg("zone-b", pb, [ua], tmp_path)), pb):
        http = httpx.AsyncClient(timeout=30)
        for _ in range(50):
            if len((await http.get(ua + "/api/peers")).json()) == 1:
                break
            await asyncio.sleep(0.2)

        c = HttpMCPClient("az", ua + "/mcp")
        await c.start()
        try:
            names = {t.name for t in c.tools}
            assert {"list_zones", "ask_agent", "ask_all_zones", "call_tool", "list_tools", "search_fixes",
                    "diagnose_proxy"} <= names

            zones = json.loads((await c.call_tool("list_zones"))[0])
            assert [z["zone"] for z in zones] == ["zone-a", "zone-b"]

            # local agent, through the convenience tool
            text, err = await c.call_tool("diagnose_proxy", {"problem_description": "407 from proxy",
                                                             "simulate": "proxy_auth_required"})
            res = json.loads(text)
            assert not err and res["verified"] and "trace" not in res

            # route a question to the peer zone's agent
            text, err = await c.call_tool("ask_agent", {"agent": "proxy-doctor", "zone_name": "zone-b",
                                                        "question": "what is this?", "context": {}})
            assert not err and json.loads(text)["zone"] == "zone-b"

            # peer tools: exposed ok, hidden refused
            text, err = await c.call_tool("call_tool", {"tool": "proxy.list_problems", "zone_name": "zone-b"})
            assert not err and "clock_skew" in text
            text, err = await c.call_tool("call_tool", {"tool": "proxy.apply_fix", "zone_name": "zone-b",
                                                        "arguments": {"problem": "clock_skew"}})
            assert err and "not exposed" in text

            # verified fix from the diagnose above is searchable
            fixes = json.loads((await c.call_tool("search_fixes", {"problem": "proxy_auth_required"}))[0])
            assert fixes and fixes[0]["verified_by"] == ["zone-a"]
        finally:
            await c.close()

        # browsers from other origins are refused (DNS rebinding)
        r = await http.post(ua + "/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "ping"},
                       headers={"Origin": "http://evil.example"})
        assert r.status_code == 403

        # stdio bridge, as Claude Desktop would launch it
        s = StdioMCPClient("bridge", sys.executable, ["-m", "agentzones.cli", "--zone", ua, "mcp"],
                           env={"PYTHONPATH": str(ROOT)})
        await s.start()
        try:
            assert "ask_agent" in {t.name for t in s.tools}
            zones = json.loads((await s.call_tool("list_zones"))[0])
            assert {z["zone"] for z in zones} == {"zone-a", "zone-b"}
        finally:
            await s.close()
            await http.aclose()
