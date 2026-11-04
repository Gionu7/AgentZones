"""LLMAgent tool-calling loop against a fake OpenAI-compatible endpoint, plus the zone MCP server."""

import contextlib
import json
import sys

import httpx
from fastapi import FastAPI

from agentzones.mcp_gateway import build_server, http_app
from agentzones.mcp_lite import HttpMCPClient
from conftest import RunningServer, free_port, run_agent, run_gateway, wait_for


def fake_llm():
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def chat(body: dict):
        msgs = body["messages"]
        names = [t["function"]["name"] for t in body.get("tools", [])]
        assert {"proxy__list_problems", "zone__send_message", "zone__list_agents"} <= set(names)
        tool_msgs = [m for m in msgs if m["role"] == "tool"]
        if not tool_msgs:
            call = {"name": "proxy__list_problems", "arguments": "{}"}
        elif len(tool_msgs) == 1:
            call = {"name": "zone__send_message",
                    "arguments": json.dumps({"zone": "solo", "agent": "echo", "payload": {"n": 1}})}
        else:
            problems, echo = json.loads(tool_msgs[0]["content"]), json.loads(tool_msgs[1]["content"])
            return {"choices": [{"message": {"role": "assistant",
                                             "content": f"{len(problems)} problems, echo hops={echo['echo']['hops']}"}}]}
        return {"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [
            {"id": f"c{len(tool_msgs)}", "type": "function", "function": call}]}}]}
    return app


async def test_llm_agent_uses_mcp_and_zone_tools():
    lp = free_port()
    async with contextlib.AsyncExitStack() as stack, httpx.AsyncClient(timeout=30) as http:
        await stack.enter_async_context(RunningServer(fake_llm(), lp))
        gw, url, _ = await stack.enter_async_context(run_gateway(stack, zone="solo"))
        await run_agent(stack, url, "echo", "agentzones.agent:EchoAgent")
        await run_agent(stack, url, "assistant", "agentzones.llm_agent:LLMAgent",
                        MODEL_URL=f"http://127.0.0.1:{lp}/v1", MODEL_NAME="fake", TOOLS="proxy.*",
                        MCP_SERVERS=json.dumps([{"name": "proxy", "command": sys.executable,
                                                 "args": ["-m", "proxy_doctor.mcp_server"]}]))
        await wait_for(lambda: len(gw.live_agents()) == 2)
        r = (await http.post(url + "/send", json={"target_agent": "assistant",
                                                 "payload": {"type": "ask", "question": "what can you diagnose?"}})).json()
        assert r["answer"] == "9 problems, echo hops=1"
        assert [s["tool"] for s in r["steps"]] == ["proxy.list_problems", "zone.send_message"]

        # zone MCP server: any MCP client can use the zone through the gateway
        mp = free_port()
        await stack.enter_async_context(RunningServer(http_app(build_server(url)), mp))
        c = HttpMCPClient("zone", f"http://127.0.0.1:{mp}/mcp")
        await c.start()
        assert {t.name for t in c.tools} == {"list_agents", "send_message"}
        text, err = await c.call_tool("list_agents", {"capability": "echo"})
        assert not err and json.loads(text)[0]["agent"] == "echo"
        text, err = await c.call_tool("send_message", {"zone": "solo", "agent": "echo", "payload": {"x": 1}})
        assert not err and json.loads(text)["echo"] == {"x": 1}
        await c.close()

