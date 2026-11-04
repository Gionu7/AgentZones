"""Three live zones over HTTP: hub az, spoke cn (direct) and spoke b (only through an HTTP proxy)."""

import contextlib
import json

import httpx
import numpy as np
import pytest

from agentzones.gateway import GatewaySettings, HubClient
from agentzones.hub import RoutingError
from conftest import ForwardProxy, run_agent, run_gateway, wait_for

TOKENS = {"b": "tok-b-0123456789", "cn": "tok-cn-0123456789"}


@contextlib.asynccontextmanager
async def three_zones(stack, proxy):
    az, az_url, hub_url = await stack.enter_async_context(
        run_gateway(stack, zone="az", role="hub", spoke_tokens=TOKENS))
    cn, cn_url, _ = await stack.enter_async_context(
        run_gateway(stack, zone="cn", role="spoke", hub_url=hub_url, hub_token=TOKENS["cn"]))
    b, b_url, _ = await stack.enter_async_context(
        run_gateway(stack, zone="b", role="spoke", hub_url=hub_url, hub_token=TOKENS["b"], hub_proxy=proxy.url))
    yield {"az": az_url, "cn": cn_url, "b": b_url, "hub": hub_url}, {"az": az, "cn": cn, "b": b}


async def send(http, gw_url, zone, agent, payload, timeout=30):
    r = await http.post(gw_url + "/send", json={"target_zone": zone, "target_agent": agent,
                                               "payload": payload, "timeout": timeout})
    return r.status_code, r.json()


async def test_messaging_directory_and_auth(tmp_path):
    async with contextlib.AsyncExitStack() as stack, httpx.AsyncClient(timeout=60) as http:
        proxy = await stack.enter_async_context(ForwardProxy())
        urls, gws = await stack.enter_async_context(three_zones(stack, proxy))
        for z in ("az", "cn", "b"):
            await run_agent(stack, urls[z], "echo", "agentzones.agent:EchoAgent")

        async def everyone():
            d = (await http.get(urls["b"] + "/directory", params={"capability": "echo"})).json()
            return {e["zone"] for e in d} == {"az", "cn", "b"}
        await wait_for(everyone)

        # b -> hub -> cn and back; neither spoke accepts connections from outside
        code, r = await send(http, urls["b"], "cn", "echo", {"hello": 1})
        assert code == 200 and r["zone"] == "cn" and r["from"] == "operator@b"
        code, r = await send(http, urls["cn"], "b", "echo", {"hello": 2})
        assert code == 200 and r["zone"] == "b" and r["from"] == "operator@cn"
        code, r = await send(http, urls["az"], "b", "echo", "x")
        assert r["zone"] == "b"
        code, r = await send(http, urls["b"], "az", "echo", "y")
        assert r["zone"] == "az"
        assert proxy.requests > 0, "b's hub traffic must go through its proxy"
        assert gws["b"].status()["hub"]["connected"]

        code, r = await send(http, urls["b"], "cn", "nobody", {})
        assert "no agent" in r["error"]
        code, _ = await send(http, urls["b"], "mars", "echo", {})
        assert code == 404

        # a spoke with a wrong token is rejected and sees why
        bad = HubClient(GatewaySettings(zone="b", role="spoke", hub_url=urls["hub"], hub_token="wrong"))
        with pytest.raises(PermissionError):
            await bad.call("GET", "/hub/directory")
        await bad.close()
        # unsigned call to the hub API
        assert (await http.get(urls["hub"] + "/hub/poll")).status_code == 401

        # b cannot read cn's mail: poll only ever returns the caller's mailbox, and b cannot answer for cn
        b_client = HubClient(GatewaySettings(zone="b", role="spoke", hub_url=urls["hub"], hub_token=TOKENS["b"]))
        await gws["az"].hub.accept({"id": "q1", "kind": "request", "from_agent": "x", "to_zone": "cn",
                                    "to_agent": "echo", "payload": {}, "expires": 2e9}, sender="az")
        with pytest.raises(RoutingError):
            await b_client.call("POST", "/hub/send", {"id": "r1", "kind": "reply", "reply_to": "q1", "payload": "fake"})
        await b_client.close()


async def test_tampering_proxy_is_detected(tmp_path):
    async with contextlib.AsyncExitStack() as stack, httpx.AsyncClient(timeout=60) as http:
        proxy = await stack.enter_async_context(ForwardProxy(tamper=True))
        urls, gws = await stack.enter_async_context(three_zones(stack, proxy))
        await run_agent(stack, urls["cn"], "echo", "agentzones.agent:EchoAgent")
        await wait_for(lambda: gws["az"].hub.directory.get("cn"))
        code, r = await send(http, urls["b"], "cn", "echo", {"msg": "hello"}, timeout=5)
        assert code == 502 and "hub" in r["detail"]
        assert any("bad signature" in e["msg"] for e in gws["az"].events)


async def test_proxy_doctor_and_federated_learning_across_zones(tmp_path):
    async with contextlib.AsyncExitStack() as stack, httpx.AsyncClient(timeout=120) as http:
        proxy = await stack.enter_async_context(ForwardProxy())
        urls, gws = await stack.enter_async_context(three_zones(stack, proxy))
        profiles = {"az": "dev-laptops", "b": "build-servers", "cn": "k8s-cluster"}
        doctors = {}
        for z in ("az", "cn", "b"):
            model = tmp_path / z / "models" / "proxy-student.npz"
            doctors[z] = await run_agent(stack, urls[z], "proxy-doctor", "proxy_doctor.agent:ProxyDoctor",
                                         DATA_DIR=str(tmp_path / z / "pd"), PD_AUTO_APPLY="1",
                                         PD_EXPLAIN_WITH_LLM="0", MODEL_PROVIDER="none", PD_PROFILE=profiles[z],
                                         PD_MODEL_PATH=str(model))
            await run_agent(stack, urls[z], "pd-fl-client", "agentzones.fl.client:FLClient",
                            ZONE=z, FL_JOB="proxy-student", FL_TASK="tasks.proxy_student:ProxyStudentClient",
                            PD_PROFILE=profiles[z], PD_SAMPLES="300", FL_MODEL_OUT=str(model))
        await run_agent(stack, urls["b"], "pd-aggregator", "agentzones.fl.aggregator:FLAggregator",
                        FL_JOB="proxy-student", FL_STRATEGY="tasks.proxy_student:ProxyStudentStrategy",
                        FL_ROUNDS="3", FL_MIN_CLIENTS="3", FL_AUTOSTART="0", FL_POLL_S="0.2",
                        FL_MODEL_DIR=str(tmp_path / "b" / "models-out"))

        async def ready():
            d = (await http.get(urls["az"] + "/directory")).json()
            return len([e for e in d if e["capabilities"] == ["proxy-doctor"]]) == 3 and \
                len([e for e in d if e["capabilities"] == ["fl-client:proxy-student"]]) == 3
        await wait_for(ready)

        # ---- Proxy Doctor: b fixes a 407 locally and sends the verified fix to az and cn
        code, r = await send(http, urls["b"], "b", "proxy-doctor", {
            "type": "ask", "question": "407 Proxy Authentication Required",
            "context": {"simulate": "proxy_auth_required"}}, timeout=60)
        assert code == 200, r
        assert r["diagnosis"]["problem"] == "proxy_auth_required" and r["verified"]
        assert r["shared_with"] == ["az", "cn"]

        # az asks its own doctor, which consults the others and reuses b's verified fix
        code, r = await send(http, urls["az"], "az", "proxy-doctor", {
            "type": "ask", "question": "pip: 407 Proxy Authentication Required",
            "context": {"simulate": "proxy_auth_required"}}, timeout=60)
        assert r["diagnosis"]["problem"] == "proxy_auth_required" and r["verified"]
        assert "b" in r["fix"]["source"]
        code, fixes = await send(http, urls["cn"], "cn", "proxy-doctor", {"type": "fixes"})
        assert fixes[0]["verified_by"] == ["az", "b"]

        # other zones may consult, but cannot make a doctor diagnose its machine
        code, r = await send(http, urls["cn"], "b", "proxy-doctor", {"type": "ask", "question": "x"})
        assert "only diagnoses its own zone" in r["error"]
        # a zone cannot vouch for others when it pushes a fix
        code, r = await send(http, urls["cn"], "b", "proxy-doctor", {"type": "pd.fix", "fix": {
            "problem": "clock_skew", "summary": "s", "steps": ["sync clock"], "verified_by": ["az", "b"]}})
        code, fx = await send(http, urls["b"], "b", "proxy-doctor", {"type": "fixes", "problem": "clock_skew"})
        assert fx[0]["verified_by"] == ["cn"]

        # ---- Federated learning: started from cn, aggregator in b, clients in all three zones
        code, r = await send(http, urls["cn"], "b", "pd-aggregator", {"type": "fl.start"})
        assert r["ok"] and r["state"] in ("waiting_for_clients", "running")

        async def done():
            _, st = await send(http, urls["az"], "b", "pd-aggregator", {"type": "fl.status"})
            assert st["state"] not in ("failed", "stopped"), st
            return st if st["state"] == "done" else None
        st = await wait_for(done, timeout=60, interval=0.3)
        assert st["round"] == 3 and len(st["history"]) == 3
        assert all(len(h["clients"]) == 3 for h in st["history"])
        accs = [h["global_acc"] for h in st["history"]]
        assert accs[-1] > 0.6 and accs[-1] >= accs[0] - 0.05, accs

        saved = tmp_path / "b" / "models-out"
        assert json.loads((saved / "proxy-student-latest.json").read_text())["round"] == 3
        W = np.load(saved / "proxy-student-latest.npz")["W"]

        # every client got the final model; each zone's doctor switches to it
        for z in ("az", "b", "cn"):
            f = tmp_path / z / "models" / "proxy-student.npz"
            await wait_for(f.exists)
            assert np.allclose(np.load(f)["W"], W, atol=1e-6)
            code, s = await send(http, urls[z], z, "proxy-doctor", {"type": "status"})
            assert s["model"].endswith("proxy-student.npz")

        # a second start while done is allowed; stop works from any zone
        code, r = await send(http, urls["az"], "b", "pd-aggregator", {"type": "fl.start", "rounds": 50})
        assert r["ok"]
        code, r = await send(http, urls["az"], "b", "pd-aggregator", {"type": "fl.stop"})
        assert r["state"] == "stopped"
