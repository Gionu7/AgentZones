"""Three real zones over HTTP in one event loop: gossip, signed asks, fix sharing, FL, auth."""

import asyncio
import sys

import httpx

from agentzones.config import ZoneConfig
from agentzones.server import create_app
from conftest import free_port

SECRET = "test-secret"


def zone_cfg(name, port, seeds, profile, tmp_path):
    return ZoneConfig.model_validate({
        "name": name, "host": "127.0.0.1", "port": port, "advertise_url": f"http://127.0.0.1:{port}",
        "data_dir": str(tmp_path / name), "model": {"provider": "none"},
        "discovery": {"peers": seeds, "gossip_interval_s": 0.3, "peer_ttl_s": 10},
        "security": {"network_secret": SECRET},
        "mcp_servers": [{"name": "proxy", "command": sys.executable, "args": ["-m", "proxy_doctor.mcp_server"],
                         "expose": ["list_problems"]}],
        "agents": [{"name": "proxy-doctor", "kind": "python", "impl": "proxy_doctor.agent:ProxyDoctor",
                    "options": {"explain_with_llm": False, "learn_from_simulated": True}}],
        "federated": {"enabled": True, "data": {"profile": profile, "n_samples": 300}},
    })


async def wait_for(pred, timeout=15):
    t0 = asyncio.get_running_loop().time()
    while not await pred():
        if asyncio.get_running_loop().time() - t0 > timeout:
            raise AssertionError("timeout")
        await asyncio.sleep(0.2)


async def test_three_zones(tmp_path, serve):
    pa, pb, pc = free_port(), free_port(), free_port()
    ua, ub, uc = (f"http://127.0.0.1:{p}" for p in (pa, pb, pc))
    cfgs = [zone_cfg("zone-a", pa, [], "dev-laptops", tmp_path),
            zone_cfg("zone-b", pb, [ua], "build-servers", tmp_path),
            zone_cfg("zone-c", pc, [ub], "k8s-cluster", tmp_path)]
    apps = [create_app(c) for c in cfgs]
    async with serve(apps[0], pa), serve(apps[1], pb), serve(apps[2], pc):
        async with httpx.AsyncClient(timeout=60) as http:
            async def connected():
                views = [(await http.get(u + "/api/peers")).json() for u in (ua, ub, uc)]
                return all(len(v) == 2 for v in views)
            await wait_for(connected)  # c learned about a via b

            # unsigned peer call is rejected
            r = await http.post(ua + "/p2p/ask", json={"agent": "proxy-doctor", "question": "x"})
            assert r.status_code == 401

            # exposed tool callable by a peer, unexposed tool refused
            zc = apps[2].state.zone
            assert "untrusted_corp_ca" in await zc.call_peer_tool("zone-a", "proxy.list_problems")
            try:
                await zc.call_peer_tool("zone-a", "proxy.apply_fix", {"problem": "x"})
                raise AssertionError("should fail")
            except RuntimeError as exc:
                assert "not exposed" in str(exc)

            # zone-b fixes a 407 locally and shares the verified fix
            r = (await http.post(ub + "/api/agents/proxy-doctor/ask", json={
                "question": "407 Proxy Authentication Required", "context": {"simulate": "proxy_auth_required"}})).json()
            assert r["diagnosis"]["problem"] == "proxy_auth_required" and r["verified"]
            assert set(r["shared_with"]) == {"zone-a", "zone-c"}

            # zone-a, unsure, asks peers and reuses zone-b's verified fix
            r = (await http.post(ua + "/api/agents/proxy-doctor/ask", json={
                "question": "pip: 407 Proxy Authentication Required", "context": {"simulate": "proxy_auth_required"}})).json()
            assert r["diagnosis"]["peers"], "zone-a should have consulted peers"
            assert r["diagnosis"]["problem"] == "proxy_auth_required"
            assert "zone-b" in r["fix"]["source"] and r["verified"]
            fixes = (await http.get(uc + "/api/fixes")).json()
            assert fixes[0]["verified_by"] == ["zone-a", "zone-b"]

            # one synchronous FedAvg round -> identical models everywhere
            trains = [(await http.post(u + "/api/fl/train")).json() for u in (ua, ub, uc)]
            # learning loop: zone-a and zone-b each verified one case, which entered local training
            assert [t["new_verified_cases"] for t in trains] == [1, 1, 0]
            aggs = [(await http.post(u + "/api/fl/aggregate")).json() for u in (ua, ub, uc)]
            assert all(len(a["peers"]) == 2 for a in aggs)
            assert len({a["global_acc"] for a in aggs}) == 1
