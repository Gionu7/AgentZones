import time

import pytest

from agentzones.hub import Hub, RoutingError


def make_hub():
    delivered = []

    async def local(msg):
        delivered.append(msg)
    hub = Hub("az", ["b", "cn"], local, visibility_s=0.2)
    for box in hub.mailboxes.values():  # both spokes are polling
        box.last_poll = time.time()
    return hub, delivered


def req(mid, to_zone, **kw):
    return {"id": mid, "kind": "request", "from_zone": "x", "from_agent": "a", "to_zone": to_zone,
            "to_agent": "t", "payload": {}, "expires": time.time() + 30, **kw}


async def test_request_routing_forces_sender_zone():
    hub, _ = make_hub()
    await hub.accept(req("m1", "cn", from_zone="az"), sender="b")  # b claims to be az
    [m] = await hub.poll("cn", 0)
    assert m["from_zone"] == "b"


async def test_unknown_zone_rejected():
    hub, _ = make_hub()
    with pytest.raises(RoutingError):
        await hub.accept(req("m1", "nowhere"), sender="b")


async def test_reply_only_from_target_and_only_to_requester():
    hub, delivered = make_hub()
    await hub.accept(req("m1", "cn"), sender="b")
    bad = {"id": "r1", "kind": "reply", "reply_to": "m1", "to_zone": "cn", "payload": "x"}
    with pytest.raises(RoutingError):
        await hub.accept(bad, sender="b")          # b may not answer a request sent to cn
    good = {"id": "r2", "kind": "reply", "reply_to": "m1", "to_zone": "az", "payload": "ok"}
    await hub.accept(good, sender="cn")            # claimed to_zone is ignored: goes back to b
    await hub.poll("cn", 0)
    [r] = await hub.poll("b", 0)
    assert r["to_zone"] == "b" and r["from_zone"] == "cn" and r["payload"] == "ok"
    with pytest.raises(RoutingError):              # a request is answered once
        await hub.accept({**good, "id": "r3"}, sender="cn")
    assert not delivered


async def test_messages_for_hub_zone_delivered_locally():
    hub, delivered = make_hub()
    await hub.accept(req("m1", "az"), sender="cn")
    assert delivered[0]["from_zone"] == "cn"


async def test_unacked_messages_are_redelivered_acked_are_not():
    hub, _ = make_hub()
    await hub.accept(req("m1", "cn"), sender="b")
    await hub.accept(req("m2", "cn"), sender="b")
    first = await hub.poll("cn", 0)
    assert [m["id"] for m in first] == ["m1", "m2"]
    assert hub.ack("cn", ["m1"]) == 1
    assert await hub.poll("cn", 0) == []           # m2 still in flight
    again = await hub.poll("cn", 1)                # long poll wakes when m2's ack times out
    assert [m["id"] for m in again] == ["m2"]


async def test_expired_messages_dropped_and_long_poll_wakes_on_new_mail():
    import asyncio
    hub, _ = make_hub()
    await hub.accept(req("old", "cn", expires=time.time() - 1), sender="b")
    assert await hub.poll("cn", 0) == []
    waiter = asyncio.create_task(hub.poll("cn", 5))
    await asyncio.sleep(0.05)
    await hub.accept(req("new", "cn"), sender="b")
    assert [m["id"] for m in await asyncio.wait_for(waiter, 1)] == ["new"]


async def test_offline_spoke_fails_fast_and_leaves_the_directory():
    hub, _ = make_hub()
    hub.register("cn", [{"agent": "echo", "capabilities": ["echo"]}])
    assert [e["zone"] for e in hub.remote_entries()] == ["cn"]
    hub.mailboxes["cn"].last_poll = time.time() - 60      # cn stopped polling
    assert hub.remote_entries() == [] and not hub.zones()["cn"]["online"]
    with pytest.raises(RoutingError, match="offline"):
        await hub.accept(req("m1", "cn"), sender="b")


async def test_retried_send_is_routed_once():
    hub, delivered = make_hub()
    await hub.accept(req("m1", "cn"), sender="b")
    await hub.accept(req("m1", "cn"), sender="b")          # response was lost, b sent it again
    assert len(await hub.poll("cn", 0)) == 1
    reply = {"id": "r1", "kind": "reply", "reply_to": "m1", "payload": "ok"}
    await hub.accept(reply, sender="cn")
    await hub.accept(reply, sender="cn")                   # retried reply: no 403, not delivered twice
    assert len(await hub.poll("b", 0)) == 1
    assert hub.totals["b>cn"]["messages"] == 1 and sum(r[1] + r[2] + r[3] for r in hub.rate) == 2
