import time

from agentzones.peers import PeerTable


def test_observe_candidates_and_self_filter():
    t = PeerTable("a", "http://a:1", ["http://b:1", "http://a:1"], ttl_s=10, max_peers=8)
    assert list(t.by_url) == ["http://b:1"]  # self removed from seeds
    t.observe({"name": "b", "url": "http://b:1"})
    t.add_candidate("http://c:1", name="c")
    t.add_candidate("http://a:1", name="a")
    assert [p.name for p in t.alive()] == ["b"]
    assert "http://c:1" in t.by_url and "http://a:1" not in t.by_url


def test_expire_keeps_seeds_drops_others(monkeypatch):
    t = PeerTable("a", "http://a:1", ["http://b:1"], ttl_s=10, max_peers=8)
    t.observe({"name": "b", "url": "http://b:1"})
    t.observe({"name": "c", "url": "http://c:1"})
    now = time.time()
    monkeypatch.setattr(time, "time", lambda: now + 60)
    t.expire()
    assert "http://b:1" in t.by_url and t.by_url["http://b:1"].name is None
    assert "http://c:1" not in t.by_url


def test_url_change_collapses_by_name():
    t = PeerTable("a", "http://a:1", [], ttl_s=10, max_peers=8)
    t.observe({"name": "b", "url": "http://10.0.0.5:1"})
    t.observe({"name": "b", "url": "http://10.0.0.9:1"})
    assert list(t.by_url) == ["http://10.0.0.9:1"]
