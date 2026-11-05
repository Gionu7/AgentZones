"""Run an ASGI app on 127.0.0.1 inside the current event loop.

Used by ``agentzone laptop`` (a whole zone in one process), ``scripts/demo.py`` and the tests.
"""

from __future__ import annotations

import asyncio
import socket

import uvicorn


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class LocalServer:
    """``async with LocalServer(app, port):`` serves ``app`` until the block ends."""

    def __init__(self, app, port: int):
        self.port = port
        self.server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
        self.task: asyncio.Task | None = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    async def __aenter__(self) -> "LocalServer":
        self.task = asyncio.create_task(self.server.serve())
        while not self.server.started:
            if self.task.done():
                self.task.result()  # raises, e.g. port in use
                raise RuntimeError(f"server on port {self.port} stopped during startup")
            await asyncio.sleep(0.02)
        return self

    async def __aexit__(self, *exc) -> None:
        self.server.should_exit = True
        await self.task
