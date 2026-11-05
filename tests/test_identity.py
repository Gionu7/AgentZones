"""Per-agent identity on the gateway, richer directory entries and GatewayClient.send_all."""

import contextlib

import httpx

from agentzones.agent import AgentHost
from conftest import RunningServer, free_port, run_agent, run_gateway, wait_for

TOKENS = {"cn": "tok-cn-0123456789"}


async def registered(gw, name):
    return await wait_for(lambda: gw.live_agents().get(name))


async def test_live_agent_name_cannot_be_taken_over():
    async with contextlib.AsyncExitStack() as stack, httpx.AsyncClient(timeout=10) as http:
        gw, url, _ = await stack.enter_async_context(run_gateway(stack, zone="z"))
        host = await run_agent(stack, url, "echo", "agentzones.agent:EchoAgent")
        real_url = (await registered(gw, "echo"))["url"]

        for headers in ({"x-agent-token": "someone-else"}, {}):
            r = await http.post(url + "/agents/register", headers=headers,
                                json={"name": "echo", "url": "http://127.0.0.1:1/evil", "capabilities": ["echo"]})
            assert r.status_code == 403
        assert gw.live_agents()["echo"]["url"] == real_url

        # the real agent keeps refreshing its own entry
        r = await http.post(url + "/agents/register", headers={"x-agent-token": host.agent.gw.agent_token},
                            json={"name": "echo", "url": real_url, "capabilities": ["echo"]})
        assert r.status_code == 200


async def test_from_agent_comes_from_the_token():
    async with contextlib.AsyncExitStack() as stack, httpx.AsyncClient(timeout=10) as http:
        gw, url, _ = await stack.enter_async_context(run_gateway(stack, zone="z"))
        a = await run_agent(stack, url, "a", "agentzones.agent:EchoAgent")
        await run_agent(stack, url, "b", "agentzones.agent:EchoAgent")
        await registered(gw, "a")
        await registered(gw, "b")

        def send(from_agent, headers=None):
            return http.post(url + "/send", headers=headers or {},
                             json={"from_agent": from_agent, "target_agent": "b", "payload": 1})

        r = await send("a")                                    # no token: may not use a registered name
        assert r.status_code == 403
        r = await send("operator")
        assert r.status_code == 200 and r.json()["from"] == "operator@z"
        r = await send("b", {"x-agent-token": a.agent.gw.agent_token})   # claims b, token says a
        assert r.status_code == 200 and r.json()["from"] == "a@z"
        assert (await a.agent.gw.send("z", "b", 2))["from"] == "a@z"


async def test_restarted_agent_gets_its_name_back():
    async with contextlib.AsyncExitStack() as stack:
        gw, url, _ = await stack.enter_async_context(run_gateway(stack, zone="z", agent_ttl_s=60))
        port = free_port()
        env = {"AGENT_NAME": "echo", "AGENT_IMPL": "agentzones.agent:EchoAgent", "AGENT_PORT": str(port),
               "AGENT_URL": f"http://127.0.0.1:{port}", "GATEWAY_URL": url, "REGISTER_INTERVAL_S": "0.3"}
        first = AgentHost(env)
        async with RunningServer(first.app(), port):
            await wait_for(lambda: gw.live_agents().get("echo", {}).get("identity") == first.agent.gw.identity)
        # same URL, new process, new token: well within the 60 s TTL of the old entry
        second = AgentHost(env)
        await stack.enter_async_context(RunningServer(second.app(), port))
        await wait_for(lambda: gw.live_agents()["echo"]["identity"] == second.agent.gw.identity)


async def test_directory_carries_version_and_accepts_and_send_all_reaches_every_zone():
    async with contextlib.AsyncExitStack() as stack:
        az, az_url, hub_url = await stack.enter_async_context(
            run_gateway(stack, zone="az", role="hub", spoke_tokens=TOKENS))
        cn, cn_url, _ = await stack.enter_async_context(
            run_gateway(stack, zone="cn", role="spoke", hub_url=hub_url, hub_token=TOKENS["cn"]))
        await run_agent(stack, az_url, "echo", "agentzones.agent:EchoAgent")
        sender = await run_agent(stack, cn_url, "echo", "agentzones.agent:EchoAgent")

        async def both():
            return {e["zone"]: e for e in await cn.directory("echo")} if len(await cn.directory("echo")) == 2 else None
        entries = await wait_for(both)
        remote = entries["az"]                                  # came through the hub
        assert remote["version"] and "*" in remote["accepts"]

        replies = await sender.agent.gw.send_all("echo", {"hi": 1}, timeout=10)
        assert sorted(r["zone"] for _, r in replies) == ["az", "cn"]
        assert all(r["from"] == "echo@cn" for _, r in replies)
        replies = await sender.agent.gw.send_all("echo", {"hi": 2}, timeout=10, other_zones_only=True)
        assert [e["zone"] for e, _ in replies] == ["az"]
        assert await sender.agent.gw.send_all("nobody", {}) == []


async def test_an_agent_cannot_pose_as_an_fl_operator(tmp_path):
    async with contextlib.AsyncExitStack() as stack, httpx.AsyncClient(timeout=10) as http:
        gw, url, _ = await stack.enter_async_context(run_gateway(stack, zone="z"))
        await run_agent(stack, url, "agg", "agentzones.fl.aggregator:FLAggregator", FL_JOB="pd",
                        FL_STRATEGY="tasks.proxy_student:ProxyStudentStrategy", FL_AUTOSTART="0",
                        FL_MODEL_DIR=str(tmp_path))
        rogue = await run_agent(stack, url, "assistant", "agentzones.agent:EchoAgent")
        await registered(gw, "agg")
        await registered(gw, "assistant")
        r = await http.post(url + "/send", headers={"x-agent-token": rogue.agent.gw.agent_token},
                            json={"from_agent": "operator", "target_agent": "agg", "payload": {"type": "fl.start"}})
        assert "assistant is not an operator" in r.json()["error"]
        assert gw.live_agents()["agg"]["accepts"]["fl.request"]
