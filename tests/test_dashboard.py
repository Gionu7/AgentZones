"""The zone dashboard (agentzones/dashboard.py). The hub view is covered in test_zones_e2e.py."""

import contextlib

import httpx

from conftest import run_gateway


async def test_dashboard_page_is_public_but_data_needs_the_api_token():
    async with contextlib.AsyncExitStack() as stack, httpx.AsyncClient(timeout=30) as http:
        gw, url, _ = await stack.enter_async_context(run_gateway(stack, zone="solo", api_token="t0k"))
        assert (await http.get(url + "/")).status_code == 200
        assert (await http.get(url + "/dashboard/data")).status_code == 401
        r = await http.get(url + "/dashboard/data", headers={"authorization": "Bearer t0k"})
        assert r.status_code == 200 and r.json()["zone"] == "solo" and r.json()["fl"] == []
