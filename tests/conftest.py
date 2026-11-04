import asyncio
import contextlib
import socket
import sys
from pathlib import Path

import pytest
import uvicorn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agentzones.agent import AgentHost  # noqa: E402
from agentzones.devproxy import ForwardProxy  # noqa: E402,F401
from agentzones.gateway import Gateway, GatewaySettings, hub_app, local_app  # noqa: E402


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class RunningServer:
    def __init__(self, app, port):
        self.app, self.port = app, port
        self.server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
        self.task = None

    @property
    def url(self):
        return f"http://127.0.0.1:{self.port}"

    async def __aenter__(self):
        self.task = asyncio.create_task(self.server.serve())
        while not self.server.started:
            if self.task.done():
                self.task.result()
            await asyncio.sleep(0.02)
        return self

    async def __aexit__(self, *exc):
        self.server.should_exit = True
        await self.task


@pytest.fixture
def serve():
    return RunningServer


FAST = dict(poll_wait_s=1.0, register_interval_s=0.3, visibility_s=2.0)


@contextlib.asynccontextmanager
async def run_gateway(stack: contextlib.AsyncExitStack, **kw):
    """Start a gateway (and its hub API for role hub). Yields (gateway, local_url, hub_url)."""
    s = GatewaySettings(host="127.0.0.1", port=free_port(), hub_port=free_port(), **{**FAST, **kw})
    gw = Gateway(s)
    local = await stack.enter_async_context(RunningServer(local_app(gw), s.port))
    hub_url = None
    if gw.hub:
        hub_url = (await stack.enter_async_context(RunningServer(hub_app(gw), s.hub_port))).url
    yield gw, local.url, hub_url


async def run_agent(stack: contextlib.AsyncExitStack, gateway_url: str, name: str, impl: str, **env):
    port = free_port()
    host = AgentHost({"AGENT_NAME": name, "AGENT_IMPL": impl, "AGENT_PORT": str(port),
                      "AGENT_URL": f"http://127.0.0.1:{port}", "GATEWAY_URL": gateway_url,
                      "REGISTER_INTERVAL_S": "0.3", **env})
    await stack.enter_async_context(RunningServer(host.app(), port))
    return host


async def wait_for(pred, timeout=20, interval=0.1):
    loop = asyncio.get_running_loop()
    t0 = loop.time()
    while True:
        res = pred()
        if asyncio.iscoroutine(res):
            res = await res
        if res:
            return res
        if loop.time() - t0 > timeout:
            raise AssertionError("timeout waiting for condition")
        await asyncio.sleep(interval)
