"""Proxy Doctor's LLM investigation step, against a fake OpenAI-compatible model."""

import json
import sys

import httpx
from fastapi import FastAPI

from agentzones.config import ZoneConfig
from agentzones.server import create_app
from conftest import free_port


def fake_llm(verdict):
    app = FastAPI()
    app.state.seen = []

    @app.post("/v1/chat/completions")
    async def chat(body: dict):
        system = body["messages"][0]["content"]
        app.state.seen.append(body)
        assert "investigator" in system
        assert "tools" not in body  # simulated machine: no live checks offered
        user = body["messages"][1]["content"]
        assert "Observation:" in user and "Classifier and peer opinions" in user
        return {"choices": [{"message": {"role": "assistant", "content": "Here you go:\n" + json.dumps(verdict)}}]}
    return app


def zone_cfg(port, llm_port, tmp_path):
    return ZoneConfig.model_validate({
        "name": "solo", "port": port, "data_dir": str(tmp_path),
        "model": {"base_url": f"http://127.0.0.1:{llm_port}/v1", "model": "fake"},
        "mcp_servers": [{"name": "proxy", "command": sys.executable, "args": ["-m", "proxy_doctor.mcp_server"]}],
        "agents": [{"name": "proxy-doctor", "kind": "python", "impl": "proxy_doctor.agent:ProxyDoctor",
                    "options": {"explain_with_llm": False}}],
    })


async def ask(port, scenario):
    async with httpx.AsyncClient(timeout=60) as http:
        return (await http.post(f"http://127.0.0.1:{port}/api/agents/proxy-doctor/ask", json={
            "question": "my tool fails", "context": {"simulate": scenario, "investigate": True}})).json()


async def test_investigator_vote_feeds_decision(tmp_path, serve):
    lp, zp = free_port(), free_port()
    llm = fake_llm({"problem": "tool_ca_bundle", "confidence": 0.9, "reasoning": "system trusts CA, certifi not",
                    "next_checks": []})
    async with serve(llm, lp), serve(create_app(zone_cfg(zp, lp, tmp_path)), zp):
        r = await ask(zp, "tool_ca_bundle")
    assert llm.state.seen, "investigator was not called"
    steps = [t["step"] for t in r["trace"]]
    assert "llm_investigate" in steps
    assert r["investigation"]["problem"] == "tool_ca_bundle"
    assert r["diagnosis"]["problem"] == "tool_ca_bundle" and r["verified"]


async def test_investigator_unknown_stops_before_any_change(tmp_path, serve):
    lp, zp = free_port(), free_port()
    llm = fake_llm({"problem": "unknown", "confidence": 0.4, "reasoning": "MTU issue on VPN?",
                    "next_checks": ["ping -M do -s 1400 proxy"]})
    async with serve(llm, lp), serve(create_app(zone_cfg(zp, lp, tmp_path)), zp):
        r = await ask(zp, "untrusted_corp_ca")
    assert r["diagnosis"]["problem"] == "unknown"
    assert "fix" not in r and not any(t["step"] == "apply_fix" for t in r["trace"])
    assert "MTU" in r["answer"] and "ping -M do" in r["answer"]


async def test_investigator_garbage_is_ignored(tmp_path, serve):
    lp, zp = free_port(), free_port()
    llm = fake_llm({"problem": "rm -rf /", "confidence": "very"})
    async with serve(llm, lp), serve(create_app(zone_cfg(zp, lp, tmp_path)), zp):
        r = await ask(zp, "proxy_auth_required")
    # an out-of-catalog answer becomes "unknown": safe default, nothing applied
    assert r["investigation"]["problem"] == "unknown" and "fix" not in r
