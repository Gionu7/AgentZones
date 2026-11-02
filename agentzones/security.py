"""Zone-to-zone request authentication.

Every peer request carries:
    X-AZ-Zone       sender zone name
    X-AZ-Timestamp  unix seconds
    X-AZ-Signature  hex HMAC-SHA256(secret, "METHOD\\nPATH\\nTIMESTAMP\\nsha256(body)")

This gives integrity + authenticity within a zone network that shares one
secret, and replay protection within the allowed clock skew. Transport
encryption (TLS / mTLS) is expected from the deployment (ingress, service
mesh, or a reverse proxy) - see docs/SECURITY.md.
"""

from __future__ import annotations

import hashlib
import hmac
import ipaddress
import time


def _payload(method: str, path: str, ts: str, body: bytes) -> bytes:
    return f"{method.upper()}\n{path}\n{ts}\n{hashlib.sha256(body).hexdigest()}".encode()


def sign_headers(secret: str, zone: str, method: str, path: str, body: bytes) -> dict[str, str]:
    headers = {"X-AZ-Zone": zone}
    if not secret:
        return headers
    ts = str(int(time.time()))
    sig = hmac.new(secret.encode(), _payload(method, path, ts, body), hashlib.sha256).hexdigest()
    headers.update({"X-AZ-Timestamp": ts, "X-AZ-Signature": sig})
    return headers


def verify(secret: str, method: str, path: str, body: bytes, headers, max_skew_s: int = 120) -> str | None:
    """Return None if OK, else a reason string."""
    if not secret:
        return None
    ts = headers.get("x-az-timestamp")
    sig = headers.get("x-az-signature")
    if not ts or not sig:
        return "missing signature"
    try:
        if abs(time.time() - int(ts)) > max_skew_s:
            return "timestamp outside allowed skew"
    except ValueError:
        return "bad timestamp"
    expected = hmac.new(secret.encode(), _payload(method, path, ts, body), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, sig):
        return "bad signature"
    return None


def ip_allowed(ip: str | None, cidrs: list[str]) -> bool:
    if not cidrs:
        return True
    if not ip:
        return False
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return False
    return any(addr in ipaddress.ip_network(c, strict=False) for c in cidrs)
