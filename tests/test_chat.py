"""Asking agents from other zones: remote tool limits, chat history, Proxy Doctor advice, read-only code server."""

import contextlib
import json
import os
import sys

import httpx
from fastapi import FastAPI

from agentzones.code_mcp import make_server
from conftest import RunningServer, free_port, run_agent, run_gateway, wait_for


def scripted_llm():
    """Calls code.search once if it may, then answers with what it saw (and how many turns it got)."""
    app = FastAPI()
    app.state.seen = []

    @app.post("/v1/chat/completions")
    async def chat(body: dict):
        msgs, tools = body["messages"], [t["function"]["name"] for t in body.get("tools") or []]
        app.state.seen.append({"tools": tools, "n": len(msgs)})
        if "code__search" in tools and not any(m["role"] == "tool" for m in msgs):
            return {"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "type": "function",
                    "function": {"name": "code__search", "arguments": json.dumps({"pattern": "def online"})}}]}}]}
        found = next((m["content"] for m in msgs if m["role"] == "tool"), "")
        return {"choices": [{"message": {"role": "assistant",
                "content": f"turns={sum(m['role'] == 'user' for m in msgs)} tools={len(tools)} found={'hub.py' in found}"}}]}
    return app


async def test_remote_questions_use_the_model_but_not_the_tools(tmp_path):
    from conftest import ForwardProxy
    lp = free_port()
    llm = scripted_llm()
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    async with contextlib.AsyncExitStack() as stack, httpx.AsyncClient(timeout=60) as http:
        await stack.enter_async_context(RunningServer(llm, lp))
        proxy = await stack.enter_async_context(ForwardProxy())
        hub, az, hub_url = await stack.enter_async_context(run_gateway(stack, zone="az", role="hub", spoke_tokens={"b": "tb"}))
        _, b, _ = await stack.enter_async_context(run_gateway(stack, zone="b", role="spoke", hub_url=hub_url, hub_token="tb",
                                                              hub_proxy=proxy.url))
        code = json.dumps([{"name": "code", "command": sys.executable, "args": ["-m", "agentzones.code_mcp"], "env": {"CODE_ROOT": repo}}])
        common = dict(MODEL_URL=f"http://127.0.0.1:{lp}/v1", MODEL_NAME="scripted", MCP_SERVERS=code, TOOLS="code.*", AGENT_CAPABILITIES="code")
        await run_agent(stack, az, "private-code", "agentzones.llm_agent:LLMAgent", **common)
        await run_agent(stack, az, "shared-code", "agentzones.llm_agent:LLMAgent", **common, LLM_REMOTE_TOOLS="code.*")
        await run_agent(stack, az, "proxy-doctor", "proxy_doctor.agent:ProxyDoctor", DATA_DIR=str(tmp_path / "pd"),
                        MODEL_PROVIDER="none")
        await wait_for(lambda: len(hub.hub.remote_entries()) == 0 and len(hub.live_agents()) == 3)

        async def ask(gw, agent, q, history=None):
            r = await http.post(gw + "/send", json={"from_agent": "dashboard-chat", "target_zone": "az", "target_agent": agent,
                                                   "payload": {"type": "ask", "question": q, "history": history or []}})
            return r.json()

        # in its own zone: tools on, and the answer is grounded in the code
        r = await ask(az, "private-code", "where is online decided?")
        assert r["answer"] == "turns=1 tools=5 found=True" and r["steps"][0]["tool"] == "code.search" and not r["remote"]
        # from zone b: same model, no tools, unless the owner shares them
        r = await ask(b, "private-code", "where is online decided?")
        assert r["answer"] == "turns=1 tools=0 found=False" and r["tools"] == [] and r["remote"]
        r = await ask(b, "shared-code", "where is online decided?",
                      history=[{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}, {"role": "system", "content": "x"}])
        assert r["answer"] == "turns=2 tools=3 found=True" and sorted(r["tools"]) == ["code.list_files", "code.read_file", "code.search"]
        assert proxy.requests > 0

        # Proxy Doctor from another zone: advice from the question's text, no checks on its machine
        r = await ask(b, "proxy-doctor", "pip: 407 Proxy Authentication Required")
        assert r["mode"] == "advice" and r["diagnosis"]["problem"] == "proxy_auth_required" and r["fix"]["steps"]
        assert "ran no checks" in r["answer"] and not any(t["step"] == "diagnose" for t in r["trace"])
        d = (await http.get(az + "/directory", params={"capability": "chat"})).json()
        assert {e["agent"] for e in d} == {"private-code", "shared-code", "proxy-doctor"}


def test_code_server_is_read_only_and_confined(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("def main():\n    return 42\n")
    (tmp_path / ".env").write_text("TOKEN=secret\n")
    (tmp_path / "id_rsa").write_text("KEY\n")
    (tmp_path / "blob.bin").write_bytes(b"\0\1\2")
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("secret outside\n")
    (tmp_path / "link.txt").symlink_to(outside)
    srv = make_server(str(tmp_path))

    def call(name, **args):
        r = srv.call(name, args)
        return r["isError"], r["content"][0]["text"]
    err, text = call("list_files")
    assert not err and "src/app.py" in text and ".env" not in text
    err, text = call("read_file", path="src/app.py")
    assert not err and "    2      return 42" in text
    for bad in (".env", "id_rsa", "blob.bin", "link.txt", "../outside.txt", "/etc/passwd"):
        assert call("read_file", path=bad)[0], bad
    err, text = call("search", pattern="secret|return")
    assert not err and "src/app.py:2" in text and "secret" not in text
    assert {t for t in srv._tools} == {"list_files", "search", "read_file"}
