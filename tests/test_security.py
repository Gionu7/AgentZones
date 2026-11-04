import time

from agentzones.security import RequestVerifier, parse_tokens, sign_request, sign_response, verify_response


def lower(h):
    return {k.lower(): v for k, v in h.items()}


def test_roundtrip_and_zone_binding():
    v = RequestVerifier({"b": "tok-b", "cn": "tok-cn"})
    h = lower(sign_request("tok-b", "b", "POST", "/hub/send", b'{"x":1}'))
    assert v.verify("POST", "/hub/send", b'{"x":1}', h) == ("b", "")
    # same request claiming to be cn: cn's token does not match
    h2 = lower(sign_request("tok-b", "b", "POST", "/hub/send", b"{}"))
    h2["x-az-zone"] = "cn"
    assert v.verify("POST", "/hub/send", b"{}", h2) == (None, "bad signature")


def test_tampering_unknown_zone_and_missing_headers():
    v = RequestVerifier({"b": "tok-b"})
    h = lower(sign_request("tok-b", "b", "POST", "/hub/send", b'{"x":1}'))
    assert v.verify("POST", "/hub/send", b'{"x":2}', h)[1] == "bad signature"
    assert v.verify("POST", "/hub/ack", b'{"x":1}', h)[1] == "bad signature"
    assert v.verify("GET", "/hub/poll", b"", {"x-az-zone": "zz"})[1] == "unknown zone"
    assert v.verify("GET", "/hub/poll", b"", {"x-az-zone": "b"})[1] == "missing signature"


def test_replay_rejected():
    v = RequestVerifier({"b": "tok-b"})
    h = lower(sign_request("tok-b", "b", "GET", "/hub/poll?wait=20", b""))
    assert v.verify("GET", "/hub/poll?wait=20", b"", h)[0] == "b"
    assert v.verify("GET", "/hub/poll?wait=20", b"", h)[1] == "replayed request"


def test_old_request_rejected(monkeypatch):
    v = RequestVerifier({"b": "k"})
    h = lower(sign_request("k", "b", "GET", "/hub/poll", b""))
    monkeypatch.setattr(time, "time", lambda: int(h["x-az-timestamp"]) + 1000)
    assert "skew" in v.verify("GET", "/hub/poll", b"", h)[1]


def test_response_bound_to_request():
    sig = sign_response("k", "nonce-1", b"[]")
    assert verify_response("k", "nonce-1", b"[]", sig)
    assert not verify_response("k", "nonce-2", b"[]", sig)      # replayed answer to another request
    assert not verify_response("k", "nonce-1", b"[{}]", sig)    # modified answer
    assert not verify_response("other", "nonce-1", b"[]", sig)
    assert not verify_response("k", "nonce-1", b"[]", None)


def test_parse_tokens():
    assert parse_tokens(" b=1 , cn=2,bad,=x,") == {"b": "1", "cn": "2"}
