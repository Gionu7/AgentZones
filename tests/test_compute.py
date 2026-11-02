"""Compute sharing: model routing across zones, fallback, the OpenAI-compatible gateway, escalation."""

import asyncio
import json
import sys

import httpx
from fastapi import FastAPI

from agentzones.compute import guess_size_b
from agentzones.config import ZoneConfig
from agentzones.server import create_app
from conftest import free_port


def test_guess_size():
    assert guess_size_b("qwen2.5:7b") == 7.0
    assert guess_size_b("Qwen/Qwen2.5-1.5B-Instruct") == 1.5
    assert guess_size_b("llama3.3:70b-instruct-q4_K_M") == 70.0
    assert guess_size_b("phi4") is None


def fake_llm(name, verdict):
    app = FastAPI()
    app.state.calls = 0

    @app.get("/v1/models")
    async def models():
        return {"data": [{"id": name}]}

    @app.post("/v1/chat/completions")
    async def chat(body: dict):
        app.state.calls += 1
        if "investigator" in body["messages"][0]["content"]:
            content = json.dumps(verdict)
        else:
            content = f"answer from {name}"
        return {"choices": [{"message": {"role": "assistant", "content": content}}]}
    return app


def cfg(name, port, llm_port, model, seeds, tmp_path, share=True, **compute):
    return ZoneConfig.model_validate({
        "name": name, "host": "127.0.0.1", "port": port, "advertise_url": f"http://127.0.0.1:{port}",
        "data_dir": str(tmp_path / name), "model": {"base_url": f"http://127.0.0.1:{llm_port}/v1", "model": model},
        "compute": {"share": share, **compute},
        "discovery": {"peers": seeds, "gossip_interval_s": 0.2}, "security": {"network_secret": "s"},
        "mcp_servers": [{"name": "proxy", "command": sys.executable, "args": ["-m", "proxy_doctor.mcp_server"]}],
        "agents": [{"name": "proxy-doctor", "kind": "python", "impl": "proxy_doctor.agent:ProxyDoctor",
                    "options": {"explain_with_llm": False}},
                   {"name": "big-thinker", "kind": "llm", "model_route": "largest", "tools": ["zone.search_fixes"],
                    "can_ask_peers": False}],
    })


async def connected(http, urls):
    for _ in range(100):
        counts = [len((await http.get(u + "/api/peers")).json()) for u in urls]
        if all(c == len(urls) - 1 for c in counts):
            # wait one more gossip round so compute cards are fresh
            await asyncio.sleep(0.5)
            return
        await asyncio.sleep(0.1)
    raise AssertionError("zones did not connect")


async def test_route_to_largest_and_gateway(tmp_path, serve):
    la, lb, pa, pb = free_port(), free_port(), free_port(), free_port()
    small = fake_llm("small", {"problem": "unknown", "confidence": 0.3, "reasoning": "no idea"})
    big = fake_llm("big", {"problem": "tool_ca_bundle", "confidence": 0.9, "reasoning": "certifi lacks CA"})
    ua, ub = f"http://127.0.0.1:{pa}", f"http://127.0.0.1:{pb}"
    async with serve(small, la), serve(big, lb), \
            serve(create_app(cfg("laptop", pa, la, "qwen2.5:1.5b", [], tmp_path)), pa), \
            serve(create_app(cfg("gpu-box", pb, lb, "qwen2.5:7b", [ua], tmp_path, gpu="NVIDIA T4")), pb):
        async with httpx.AsyncClient(timeout=60) as http:
            await connected(http, [ua, ub])

            models = (await http.get(ua + "/v1/models")).json()["data"]
            ids = [m["id"] for m in models]
            assert "gpu-box/qwen2.5:7b" in ids and "laptop/qwen2.5:1.5b" in ids
            assert next(m for m in models if m["id"] == "gpu-box/qwen2.5:7b")["gpu"] == "NVIDIA T4"

            async def gw(model):
                r = await http.post(ua + "/v1/chat/completions",
                                    json={"model": model, "messages": [{"role": "user", "content": "hi"}]})
                return r.json()
            r = await gw("largest")
            assert r["model"] == "gpu-box/qwen2.5:7b" and r["choices"][0]["message"]["content"] == "answer from big"
            assert (await gw("local"))["model"] == "laptop/qwen2.5:1.5b"
            assert (await gw("gpu-box"))["model"] == "gpu-box/qwen2.5:7b"
            assert (await gw("qwen2.5:1.5b"))["model"] == "laptop/qwen2.5:1.5b"

            # an LLM agent configured with model_route: largest borrows the big model; tools stay local
            r = (await http.post(ua + "/api/agents/big-thinker/ask", json={"question": "hello"})).json()
            assert r["answer"] == "answer from big" and r["model_zones"] == ["gpu-box"]

            # investigator: the local small model says "unknown" -> escalates to the GPU box
            r = (await http.post(ua + "/api/agents/proxy-doctor/ask", json={
                "question": "pip fails", "context": {"simulate": "tool_ca_bundle", "investigate": True}})).json()
            inv = r["investigation"]
            assert inv["model_zone"] == "gpu-box" and inv["problem"] == "tool_ca_bundle"
            assert inv["escalated_from"]["problem"] == "unknown"
            # the borrowed model's verdict is one vote next to the local classifier's
            assert r["diagnosis"]["problem"] in ("tool_ca_bundle", r["diagnosis"]["local"]["problem"])

            info = (await http.get(ub + "/api/info")).json()
            assert info["compute"]["served"] >= 3
            events = (await http.get(ua + "/api/events")).json()
            assert any(e["kind"] == "llm" and e.get("peer") == "gpu-box" for e in events)

            # network map: both zones, one link, borrowed-model traffic laptop -> gpu-box
            net = (await http.get(ua + "/api/network")).json()
            assert {n["name"] for n in net["nodes"]} == {"laptop", "gpu-box"}
            assert net["links"] == [["gpu-box", "laptop"]]
            gpu = next(n for n in net["nodes"] if n["name"] == "gpu-box")
            assert gpu["share"] and gpu["gpu"] == "NVIDIA T4" and gpu["size_b"] == 7.0
            assert any(f["kind"] == "llm" and f["src"] == "laptop" and f["dst"] == "gpu-box" for f in net["flows"])
            later = (await http.get(ua + f"/api/network?since={net['t']}")).json()
            assert later["flows"] == []
            assert (await http.get(ua + "/map")).status_code == 200


async def test_refusal_falls_back_to_local(tmp_path, serve):
    la, lb, pa, pb = free_port(), free_port(), free_port(), free_port()
    ua, ub = f"http://127.0.0.1:{pa}", f"http://127.0.0.1:{pb}"
    async with serve(fake_llm("small", {}), la), serve(fake_llm("big", {}), lb), \
            serve(create_app(cfg("laptop", pa, la, "qwen2.5:1.5b", [], tmp_path)), pa), \
            serve(create_app(cfg("gpu-box", pb, lb, "qwen2.5:7b", [ua], tmp_path, accept_from=["someone-else"])), pb):
        async with httpx.AsyncClient(timeout=60) as http:
            await connected(http, [ua, ub])
            r = (await http.post(ua + "/v1/chat/completions",
                                 json={"model": "largest", "messages": [{"role": "user", "content": "hi"}]})).json()
            assert r["model"] == "laptop/qwen2.5:1.5b"  # gpu-box refused (403) -> local fallback
            # direct peer call is refused
            r = await http.post(ua + "/v1/chat/completions",
                                json={"model": "nowhere", "messages": [{"role": "user", "content": "hi"}]})
            assert r.status_code == 503


async def test_not_shared_is_not_used(tmp_path, serve):
    la, lb, pa, pb = free_port(), free_port(), free_port(), free_port()
    ua, ub = f"http://127.0.0.1:{pa}", f"http://127.0.0.1:{pb}"
    async with serve(fake_llm("small", {}), la), serve(fake_llm("big", {}), lb), \
            serve(create_app(cfg("laptop", pa, la, "qwen2.5:1.5b", [], tmp_path)), pa), \
            serve(create_app(cfg("gpu-box", pb, lb, "qwen2.5:7b", [ua], tmp_path, share=False)), pb):
        async with httpx.AsyncClient(timeout=60) as http:
            await connected(http, [ua, ub])
            ids = [m["id"] for m in (await http.get(ua + "/v1/models")).json()["data"]]
            assert "gpu-box/qwen2.5:7b" not in ids
            r = (await http.post(ua + "/v1/chat/completions",
                                 json={"model": "largest", "messages": [{"role": "user", "content": "x"}]})).json()
            assert r["model"] == "laptop/qwen2.5:1.5b"
