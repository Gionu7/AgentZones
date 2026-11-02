"""LLMAgent tool-calling loop against a fake OpenAI-compatible endpoint."""

import json
import sys

import httpx
from fastapi import FastAPI

from agentzones.config import ZoneConfig
from agentzones.server import create_app
from conftest import free_port


def fake_llm():
    app = FastAPI()
    app.state.calls = []

    @app.get("/v1/models")
    async def models():
        return {"data": [{"id": "fake"}]}

    @app.post("/v1/chat/completions")
    async def chat(body: dict):
        app.state.calls.append(body)
        msgs = body["messages"]
        if msgs[-1]["role"] == "tool":
            data = json.loads(msgs[-1]["content"])
            return {"choices": [{"message": {"role": "assistant",
                                             "content": f"Known problems: {len(data)}"}}]}
        names = [t["function"]["name"] for t in body.get("tools", [])]
        assert "proxy__list_problems" in names and "zone__ask_peers" in names
        return {"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "proxy__list_problems", "arguments": "{}"}}]}}]}
    return app


async def test_llm_agent_uses_mcp_tool(tmp_path, serve):
    lp, zp = free_port(), free_port()
    llm_app = fake_llm()
    cfg = ZoneConfig.model_validate({
        "name": "solo", "port": zp, "data_dir": str(tmp_path),
        "model": {"base_url": f"http://127.0.0.1:{lp}/v1", "model": "fake"},
        "mcp_servers": [{"name": "proxy", "command": sys.executable, "args": ["-m", "proxy_doctor.mcp_server"]}],
        "agents": [{"name": "assistant", "kind": "llm", "tools": ["proxy.*"]}],
    })
    async with serve(llm_app, lp), serve(create_app(cfg), zp):
        async with httpx.AsyncClient(timeout=30) as http:
            r = (await http.post(f"http://127.0.0.1:{zp}/api/agents/assistant/ask",
                                 json={"question": "what can you diagnose?"})).json()
            assert r["answer"] == "Known problems: 9"
            assert r["steps"][0]["tool"] == "proxy.list_problems"
            info = (await http.get(f"http://127.0.0.1:{zp}/api/info")).json()
            assert info["llm"]["ok"]
