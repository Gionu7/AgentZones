"""A tiny HTTP forward proxy for tests and the demo (plain HTTP, no CONNECT).

It stands in for zone b's corporate proxy: it counts what passes through and,
with ``tamper=True``, changes request bodies like a hostile proxy would.
Not for production use.
"""

from __future__ import annotations

import asyncio

from .localserver import free_port


class ForwardProxy:
    """Tiny HTTP forward proxy (absolute-form requests, no CONNECT). Counts requests; can tamper with bodies."""

    def __init__(self, tamper: bool = False, port: int = 0):
        self.port, self.requests, self.tamper = port or free_port(), 0, tamper
        self.server = None

    @property
    def url(self):
        return f"http://127.0.0.1:{self.port}"

    async def _handle(self, reader, writer):
        try:
            while True:
                line = await reader.readline()
                if not line:
                    break
                method, url, version = line.decode().split()
                headers, length = [], 0
                while (h := await reader.readline()) not in (b"\r\n", b""):
                    name, _, val = h.decode().partition(":")
                    if name.lower() == "content-length":
                        length = int(val)
                    if name.lower() not in ("proxy-connection", "connection"):
                        headers.append(h)
                body = await reader.readexactly(length) if length else b""
                if self.tamper and body:
                    body = body.replace(b"hello", b"HELLO")
                self.requests += 1
                rest = url.split("://", 1)[1]
                hostport, _, path = rest.partition("/")
                host, _, port = hostport.partition(":")
                ur, uw = await asyncio.open_connection(host, int(port or 80))
                uw.write(f"{method} /{path} {version}\r\n".encode() + b"".join(headers) + b"Connection: close\r\n\r\n" + body)
                await uw.drain()
                resp = await ur.read()
                uw.close()
                head, _, rbody = resp.partition(b"\r\n\r\n")
                lines = [ln for ln in head.split(b"\r\n") if not ln.lower().startswith(b"connection:")]
                writer.write(b"\r\n".join(lines) + b"\r\nConnection: keep-alive\r\n\r\n" + rbody)
                await writer.drain()
        except (ConnectionError, ValueError, asyncio.IncompleteReadError):
            pass
        finally:
            writer.close()

    async def __aenter__(self):
        self.server = await asyncio.start_server(self._handle, "127.0.0.1", self.port)
        return self

    async def __aexit__(self, *exc):
        self.server.close()
