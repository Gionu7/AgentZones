import time

from agentzones.security import ip_allowed, sign_headers, verify


def lower(h):
    return {k.lower(): v for k, v in h.items()}


def test_sign_and_verify_roundtrip():
    h = sign_headers("s3cret", "zone-a", "POST", "/p2p/ask", b'{"q":1}')
    assert verify("s3cret", "POST", "/p2p/ask", b'{"q":1}', lower(h)) is None


def test_tampered_body_or_wrong_secret_rejected():
    h = lower(sign_headers("s3cret", "zone-a", "POST", "/p2p/ask", b'{"q":1}'))
    assert verify("s3cret", "POST", "/p2p/ask", b'{"q":2}', h) == "bad signature"
    assert verify("other", "POST", "/p2p/ask", b'{"q":1}', h) == "bad signature"
    assert verify("s3cret", "POST", "/p2p/ask", b"", {}) == "missing signature"


def test_replay_outside_skew_rejected(monkeypatch):
    h = lower(sign_headers("k", "z", "GET", "/p2p/fl/weights", b""))
    monkeypatch.setattr(time, "time", lambda: int(h["x-az-timestamp"]) + 1000)
    assert "skew" in verify("k", "GET", "/p2p/fl/weights", b"", h, max_skew_s=120)


def test_no_secret_means_open():
    assert verify("", "GET", "/x", b"", {}) is None


def test_cidr_allowlist():
    assert ip_allowed("10.1.2.3", ["10.0.0.0/8"])
    assert not ip_allowed("192.168.1.1", ["10.0.0.0/8"])
    assert ip_allowed("1.2.3.4", [])
