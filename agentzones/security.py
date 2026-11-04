"""Authentication of spoke <-> hub traffic.

Each spoke zone shares one secret token with the hub (``SPOKE_TOKENS`` on the
hub, ``HUB_TOKEN`` on the spoke). The token is used as an HMAC key and never
travels on the wire, so a proxy on the path cannot read and reuse it.

Request headers (spoke -> hub)::

    X-AZ-Zone       sending zone
    X-AZ-Timestamp  unix seconds
    X-AZ-Nonce      random, single use
    X-AZ-Signature  hex HMAC-SHA256(token, "REQ\\nMETHOD\\nPATH?QUERY\\nTS\\nNONCE\\nsha256(body)")

The hub answers with ``X-AZ-Signature`` = HMAC(token, "RES\\n<request nonce>\\nsha256(body)"),
which binds every response to its request: a man in the middle can neither
change a message nor replay an old answer. It *can* still read the bodies
(confidentiality needs TLS, see docs/SECURITY.md).
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import time

MAX_SKEW_S = 120


def _digest(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def _mac(token: str, text: str) -> str:
    return hmac.new(token.encode(), text.encode(), hashlib.sha256).hexdigest()


def sign_request(token: str, zone: str, method: str, path: str, body: bytes) -> dict[str, str]:
    ts, nonce = str(int(time.time())), secrets.token_hex(16)
    sig = _mac(token, f"REQ\n{method.upper()}\n{path}\n{ts}\n{nonce}\n{_digest(body)}")
    return {"X-AZ-Zone": zone, "X-AZ-Timestamp": ts, "X-AZ-Nonce": nonce, "X-AZ-Signature": sig}


def sign_response(token: str, nonce: str, body: bytes) -> str:
    return _mac(token, f"RES\n{nonce}\n{_digest(body)}")


def verify_response(token: str, nonce: str, body: bytes, signature: str | None) -> bool:
    return bool(signature) and hmac.compare_digest(sign_response(token, nonce, body), signature or "")


class RequestVerifier:
    """Checks spoke requests on the hub: known zone, valid MAC, fresh, nonce not seen before."""

    def __init__(self, tokens: dict[str, str], max_skew_s: int = MAX_SKEW_S):
        self.tokens, self.max_skew_s = tokens, max_skew_s
        self._nonces: dict[str, float] = {}

    def verify(self, method: str, path: str, body: bytes, headers) -> tuple[str | None, str]:
        """Return (zone, "") if the request is authentic, else (None, reason)."""
        zone = headers.get("x-az-zone") or ""
        ts, nonce, sig = (headers.get(h) for h in ("x-az-timestamp", "x-az-nonce", "x-az-signature"))
        token = self.tokens.get(zone)
        if not token:
            return None, "unknown zone"
        if not (ts and nonce and sig):
            return None, "missing signature"
        try:
            if abs(time.time() - int(ts)) > self.max_skew_s:
                return None, "timestamp outside allowed skew"
        except ValueError:
            return None, "bad timestamp"
        expected = _mac(token, f"REQ\n{method.upper()}\n{path}\n{ts}\n{nonce}\n{_digest(body)}")
        if not hmac.compare_digest(expected, sig):
            return None, "bad signature"
        now = time.time()
        if len(self._nonces) > 10_000 or (self._nonces and next(iter(self._nonces.values())) < now - 2 * self.max_skew_s):
            self._nonces = {n: t for n, t in self._nonces.items() if t > now - 2 * self.max_skew_s}
        key = f"{zone}:{nonce}"
        if key in self._nonces:
            return None, "replayed request"
        self._nonces[key] = now
        return zone, ""


def parse_tokens(spec: str) -> dict[str, str]:
    """``"b=tok1,cn=tok2"`` -> ``{"b": "tok1", "cn": "tok2"}``."""
    out: dict[str, str] = {}
    for part in (spec or "").split(","):
        zone, sep, token = part.strip().partition("=")
        if sep and zone.strip() and token.strip():
            out[zone.strip()] = token.strip()
    return out
