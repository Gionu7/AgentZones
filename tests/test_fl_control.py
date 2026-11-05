"""FL controls: MCP tools that only request, operator approval, limits, audit log, update screening."""

import contextlib
import json

import httpx
import numpy as np
import pytest

from agentzones.agent import GatewayClient
from agentzones.fl import FitResult
from agentzones.fl.aggregator import FLAggregator
from agentzones.mcp_gateway import build_server, http_app
from agentzones.mcp_lite import HttpMCPClient
from conftest import RunningServer, free_port, run_agent, run_gateway, wait_for


async def test_llm_requests_need_operator_approval(tmp_path):
    async with contextlib.AsyncExitStack() as stack, httpx.AsyncClient(timeout=60) as http:
        gw, url, _ = await stack.enter_async_context(run_gateway(stack, zone="solo"))
        for i in range(2):
            await run_agent(stack, url, f"client{i}", "agentzones.fl.client:FLClient", ZONE=f"site{i}",
                            FL_JOB="pd", FL_TASK="tasks.proxy_student:ProxyStudentClient", PD_SAMPLES="200")
        await run_agent(stack, url, "agg", "agentzones.fl.aggregator:FLAggregator", FL_JOB="pd",
                        FL_STRATEGY="tasks.proxy_student:ProxyStudentStrategy", FL_MIN_CLIENTS="2",
                        FL_AUTOSTART="0", FL_MAX_ROUNDS="3", FL_POLL_S="0.2", FL_MODEL_DIR=str(tmp_path))
        await wait_for(lambda: len(gw.live_agents()) == 3)

        async def send(payload, sender="operator"):
            return (await http.post(url + "/send", json={"from_agent": sender, "target_agent": "agg",
                                                        "payload": payload})).json()

        # an LLM agent goes through the MCP tools; it can only ask
        mp = free_port()
        await stack.enter_async_context(RunningServer(http_app(build_server(url, name="assistant")), mp))
        mcp = HttpMCPClient("zone", f"http://127.0.0.1:{mp}/mcp")
        await mcp.start()
        assert {"fl_list_jobs", "fl_status", "fl_request_start", "fl_request_stop"} <= {t.name for t in mcp.tools}
        jobs = json.loads((await mcp.call_tool("fl_list_jobs", {}))[0])
        assert jobs == [{"job": "pd", "aggregator": "agg@solo", "clients": ["client0@solo", "client1@solo"]}]
        text, err = await mcp.call_tool("fl_request_start", {"job": "pd", "rounds": 10, "reason": "new incidents"})
        req = json.loads(text)
        assert not err and req["pending"] and req["state"] == "idle"
        # going around the tools does not help: fl.start is for operators
        r = await send({"type": "fl.start"}, sender="assistant")
        assert "not an operator" in r["error"]
        assert "only operators" in (await send({"type": "fl.approve", "id": req["pending"]}, "assistant"))["error"]

        # an operator approves; rounds are capped at FL_MAX_ROUNDS
        r = await send({"type": "fl.approve", "id": req["pending"]})
        assert r.get("ok"), {k: v for k, v in r.items() if k in ("error", "pending", "audit", "state")}
        assert r["rounds"] == 3 and r["pending"] == []
        st = await wait_for(lambda: _done(send), timeout=30)
        assert st["round"] == 3 and st["started_by"] == "assistant@solo"
        for rec in st["history"]:
            shares = [c["share"] for c in rec["contrib"] if c["status"] == "used"]
            assert len(shares) == 2 and sum(shares) == pytest.approx(1, abs=1e-3)
            assert all(c["update_norm"] > 0 for c in rec["contrib"])

        # a second request is rejected by the operator
        text, _ = await mcp.call_tool("fl_request_start", {"job": "pd", "rounds": 2, "reason": "again"})
        r = await send({"type": "fl.reject", "id": json.loads(text)["pending"], "reason": "not now"})
        assert r["ok"] and r["state"] == "done"
        await mcp.close()

        outcomes = [(a["action"], a["outcome"]) for a in r["audit"]]
        assert outcomes == [("request start", "pending approval"), ("fl.start", "refused"),
                            ("fl.approve", "refused"), ("approve start", "approved"), ("start", "started"),
                            ("request start", "pending approval"), ("reject start", "rejected")]
        assert r["audit"][4]["who"] == "assistant@solo" and "capped" in r["audit"][4]["note"]
        assert len((tmp_path / "pd-audit.jsonl").read_text().splitlines()) == 7


async def _done(send):
    st = await send({"type": "fl.status"})
    assert st["state"] not in ("failed", "stopped"), st
    return st if st["state"] == "done" else None


def test_screening_caps_samples_and_drops_outliers():
    agg = FLAggregator("agg", GatewayClient("http://127.0.0.1:1", "agg"),
                       {"FL_MAX_SAMPLES": "1000", "FL_MAX_UPDATE_RATIO": "5"})
    agg.parameters = {"w": np.zeros(4)}
    results = [FitResult("az", "c", {"w": np.full(4, 0.10)}, 500),
               FitResult("cn", "c", {"w": np.full(4, 0.12)}, 1_000_000),   # claims a huge dataset
               FitResult("b", "c", {"w": np.full(4, 0.11)}, 500),
               FitResult("x", "c", {"w": np.full(4, 9.0)}, 500)]            # a wild update
    used, report = agg._screen(results)
    by = {e["client"].split("@")[1]: e for e in report}
    assert [r.zone for r in used] == ["az", "cn", "b"]
    assert by["cn"]["samples"] == 1000 and by["cn"]["claimed"] == 1_000_000 and by["cn"]["share"] == 0.5
    assert by["x"]["status"] == "rejected" and "median" in by["x"]["why"] and by["x"]["share"] == 0
