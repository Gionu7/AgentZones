"""agentzone model setup against a fake Ollama API, and agentzone laptop."""

import asyncio
import contextlib
import hashlib

import httpx
import pytest
from fastapi import FastAPI, Request, Response

from agentzones import laptop
from agentzones.models import ollama_root, setup, source_kind
from conftest import RunningServer, free_port, wait_for

GGUF = b"GGUF" + b"\0" * 1000


def fake_ollama():
    app = FastAPI()
    st = app.state
    st.models, st.blobs, st.calls = {"qwen2.5:1.5b"}, {}, []

    @app.get("/api/version")
    async def version():
        return {"version": "0.12.6"}

    @app.get("/api/tags")
    async def tags():
        return {"models": [{"name": m} for m in sorted(st.models)]}

    @app.post("/api/pull")
    async def pull(body: dict):
        st.calls.append(("pull", body["model"]))
        name = body["model"] if ":" in body["model"].rsplit("/", 1)[-1] else body["model"] + ":latest"
        st.models.add(name)
        return {"status": "success"}

    @app.post("/api/copy")
    async def copy(body: dict):
        st.calls.append(("copy", body["source"], body["destination"]))
        st.models.add(body["destination"])
        return {}

    @app.head("/api/blobs/{digest}")
    async def has_blob(digest: str):
        return Response(status_code=200 if digest in st.blobs else 404)

    @app.post("/api/blobs/{digest}")
    async def blob(digest: str, request: Request):
        data = await request.body()
        assert "sha256:" + hashlib.sha256(data).hexdigest() == digest
        st.blobs[digest] = data
        return Response(status_code=201)

    @app.post("/api/create")
    async def create(body: dict):
        assert all(d in st.blobs for d in body["files"].values())
        st.calls.append(("create", body["model"], body.get("system", "")))
        st.models.add(body["model"] + ":latest")
        return {"status": "success"}
    return app


def files_app():
    app = FastAPI()

    @app.get("/models/{name}")
    async def get(name: str):
        return Response(GGUF, media_type="application/octet-stream")
    return app


def test_source_kinds():
    assert source_kind("qwen2.5:3b") == source_kind("hf.co/Qwen/Qwen2.5-3B-Instruct-GGUF:Q4_K_M") == "ollama"
    assert source_kind("https://blob.example/m.gguf?sv=x") == "url"
    assert source_kind("/local-models/pd.gguf") == source_kind("models/pd.gguf") == "file"
    assert ollama_root("http://ollama:11434/v1/") == "http://ollama:11434"


async def test_setup_pull_url_file_and_skip(tmp_path):
    op, fp = free_port(), free_port()
    ol = fake_ollama()
    async with RunningServer(ol, op), RunningServer(files_app(), fp):
        url = f"http://127.0.0.1:{op}/v1"
        run = lambda *a, **k: asyncio.to_thread(setup, *a, model_url=url, cache_dir=str(tmp_path / "cache"), **k)
        assert "already" in await run("qwen2.5:1.5b")
        assert await run("llama3.2:3b") == "pulled llama3.2:3b"
        hf = "hf.co/Qwen/Qwen2.5-3B-Instruct-GGUF:Q4_K_M"
        assert await run("assistant-model", hf) == f"pulled {hf} as assistant-model"
        assert ("copy", hf, "assistant-model") in ol.state.calls

        sha = hashlib.sha256(GGUF).hexdigest()
        out = await run("proxy-doctor", f"http://127.0.0.1:{fp}/models/proxy-doctor-1.5b.gguf?sig=secret", sha256=sha,
                        system="You are Proxy Doctor.")
        assert out == "created proxy-doctor from proxy-doctor-1.5b.gguf"
        assert (tmp_path / "cache" / "proxy-doctor-1.5b.gguf").read_bytes() == GGUF
        assert ("create", "proxy-doctor", "You are Proxy Doctor.") in ol.state.calls
        with pytest.raises(ValueError, match="sha256"):
            await run("other", f"http://127.0.0.1:{fp}/models/x.gguf", sha256="0" * 64)

        local = tmp_path / "mine.gguf"
        local.write_bytes(GGUF)
        assert await run("mine", str(local)) == "created mine from mine.gguf"
        assert "already" in await run("mine", str(local))
        assert (await run("mine", str(local), force=True)).startswith("created")


async def test_setup_with_a_server_that_is_not_ollama():
    app = FastAPI()

    @app.get("/v1/models")
    async def models():
        return {"data": [{"id": "phi-4-mini"}]}
    port = free_port()
    async with RunningServer(app, port):
        url = f"http://127.0.0.1:{port}/v1"
        assert "not Ollama" in await asyncio.to_thread(setup, "phi-4-mini", model_url=url, wait_s=5)
        with pytest.raises(RuntimeError, match="serves"):
            await asyncio.to_thread(setup, "llama3.2:3b", model_url=url, wait_s=5)


async def test_laptop_zone_runs_without_docker(tmp_path):
    port = free_port()
    task = asyncio.create_task(laptop.run("laptop-test", port=port, mcp_port=0, data_dir=str(tmp_path), llm=False, fl=True))
    try:
        async with httpx.AsyncClient(timeout=60) as http:
            async def ready():
                try:
                    d = (await http.get(f"http://127.0.0.1:{port}/directory")).json()
                except httpx.HTTPError:
                    return False
                return {e["agent"] for e in d} == {"proxy-doctor", "pd-fl-client"}
            await wait_for(ready, timeout=30)
            r = (await http.post(f"http://127.0.0.1:{port}/send", json={"target_agent": "proxy-doctor", "payload": {
                "type": "ask", "question": "pip: CERTIFICATE_VERIFY_FAILED", "context": {"simulate": "untrusted_corp_ca"}}})).json()
            assert r["diagnosis"]["problem"] == "untrusted_corp_ca" and r["verified"] is False  # proposes, applies nothing
            assert (await http.get(f"http://127.0.0.1:{port}/")).status_code == 200
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


async def test_laptop_joins_the_hub_as_a_spoke(tmp_path):
    from conftest import ForwardProxy, run_gateway
    async with contextlib.AsyncExitStack() as stack, httpx.AsyncClient(timeout=30) as http:
        proxy = await stack.enter_async_context(ForwardProxy())
        gw, hub_local, hub_url = await stack.enter_async_context(
            run_gateway(stack, zone="az", role="hub", spoke_tokens={"laptop-alice": "tok-alice-123"}))
        port = free_port()
        task = asyncio.create_task(laptop.run("laptop-alice", hub_url=hub_url, hub_token="tok-alice-123",
                                              hub_proxy=proxy.url, port=port, mcp_port=0, data_dir=str(tmp_path),
                                              llm=False))
        try:
            async def joined():
                d = (await http.get(hub_local + "/directory", params={"capability": "proxy-doctor"})).json()
                return [e["zone"] for e in d] == ["laptop-alice"]
            await wait_for(joined, timeout=30)
            # the hub can ask the laptop's doctor, which gives advice but never checks the laptop for others
            r = (await http.post(hub_local + "/send", json={"target_zone": "laptop-alice", "target_agent": "proxy-doctor",
                                                           "payload": {"type": "ask", "question": "pip: 407"}})).json()
            assert r["mode"] == "advice" and not any(t["step"] == "diagnose" for t in r["trace"])
            assert gw.hub.zones()["laptop-alice"]["via_proxy"] is True and proxy.requests > 0
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
