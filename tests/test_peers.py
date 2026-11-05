"""PeerAgent: agents of one kind consult each other and share verified fixes; no proxy code involved."""

from typing import Any

import pytest

from agentzones.agent import Incoming
from agentzones.peers import PeerAgent


class Mesh:
    """Agents in several zones; ``send_all`` delivers straight to the matching agents' ``handle``."""

    def __init__(self):
        self.agents: list[PeerAgent] = []
        self.sent: list[tuple[str, dict]] = []

    def gw(self, zone: str) -> "FakeGateway":
        return FakeGateway(self, zone)


class FakeGateway:
    def __init__(self, mesh: Mesh, zone: str):
        self.mesh, self.zone = mesh, zone

    async def send_all(self, capability, payload, timeout=60, other_zones_only=False):
        self.mesh.sent.append((self.zone, payload))
        out = []
        for a in self.mesh.agents:
            if capability in a.all_capabilities() and not (other_zones_only and a.zone == self.zone):
                msg = Incoming("m", self.zone, "car-doctor", payload, local=a.zone == self.zone)
                out.append(({"zone": a.zone, "agent": a.name}, await a.handle(msg)))
        return out


class CarDoctor(PeerAgent):
    """A toy peer agent: diagnoses whatever ``SAYS`` tells it (``problem:confidence``)."""

    capabilities = ["car-doctor"]
    peer_capability = "car-doctor"
    peer_prefix = "car"
    share_keys = ("noise", "fuel")
    problems = {"healthy": "the car runs", "flat_tire": "a tyre is flat", "no_fuel": "the tank is empty"}
    accepts = {"ask": "{observation} -> diagnosis"}

    async def classify(self, observation: dict[str, Any]) -> dict[str, Any]:
        self.seen = observation
        problem, conf = self.config["SAYS"].split(":")
        return {"problem": problem, "confidence": float(conf)}

    async def handle(self, msg: Incoming) -> Any:
        if (reply := await self.handle_peer(msg)) is not None:
            return reply
        return self.unknown(msg.type)


async def doctors(tmp_path, mesh: Mesh, says: dict[str, str], cls=CarDoctor, **env) -> dict[str, PeerAgent]:
    out = {}
    for zone, s in says.items():
        a = cls(f"doc-{zone}", mesh.gw(zone), {"SAYS": s, "DATA_DIR": str(tmp_path / zone), **env})
        await a.start()
        mesh.agents.append(a)
        out[zone] = a
    return out


async def test_unsure_peers_vote_and_see_only_shared_keys(tmp_path):
    mesh = Mesh()
    d = await doctors(tmp_path, mesh, {"a": "flat_tire:0.4", "b": "no_fuel:0.9", "c": "no_fuel:0.8"})
    obs = {"noise": "none", "fuel": 0, "owner": "secret", "gps": [1, 2]}
    r = await d["a"].ask_peers("won't start", obs, await d["a"].classify(obs))
    assert r.problem == "no_fuel" and r.confidence == pytest.approx((0.9 + 0.8) / 3)
    assert [p["zone"] for p in r.peers] == ["b", "c"] and r.trace[0]["mode"] == "vote"
    # all that left zone a was the share_keys, and the peers classified only those
    assert mesh.sent[0][1]["observation"] == {"noise": "none", "fuel": 0}
    assert d["b"].seen == {"noise": "none", "fuel": 0, "question": "won't start"}


async def test_sure_with_a_verified_fix_asks_nobody(tmp_path):
    mesh = Mesh()
    d = await doctors(tmp_path, mesh, {"a": "flat_tire:0.95", "b": "no_fuel:0.9"})
    await d["a"].learned("flat_tire", "pump it", ["pump to 2.2 bar"])
    mesh.sent.clear()
    r = await d["a"].ask_peers("q", {"noise": "thump"}, await d["a"].classify({}))
    assert r.problem == "flat_tire" and r.confidence == 0.95 and not r.peers and not mesh.sent
    assert d["a"].fix_plan(d["a"].best_fix("flat_tire")) == {
        "problem": "flat_tire", "summary": "pump it", "steps": ["pump to 2.2 bar"], "source": "verified by a"}


async def test_sure_without_a_fix_takes_a_peers_fix_but_keeps_its_diagnosis(tmp_path):
    mesh = Mesh()
    d = await doctors(tmp_path, mesh, {"a": "flat_tire:0.95", "b": "flat_tire:0.6", "c": "no_fuel:0.99"},
                      CAR_SHARE_FIXES="0")
    await d["b"].learned("flat_tire", "swap the wheel", ["fit the spare"])
    r = await d["a"].ask_peers("q", {"noise": "thump"}, await d["a"].classify({}))
    assert r.problem == "flat_tire" and r.confidence == 0.95 and r.trace[0]["mode"] == "fixes_only"
    fix = d["a"].best_fix(r.problem, r.fixes)
    assert fix["steps"] == ["fit the spare"] and d["a"].fix_plan(fix)["source"] == "verified by b"
    assert d["a"].best_fix("no_fuel", r.fixes) is None  # nobody verified one: the agent uses its own catalog


async def test_a_verified_fix_reaches_the_peers_as_vouched_for_by_its_zone_only(tmp_path):
    mesh = Mesh()
    d = await doctors(tmp_path, mesh, {"a": "healthy:1", "b": "healthy:1", "c": "healthy:1"})
    fix, shared = await d["a"].learned("no_fuel", "refuel", ["add 10 l"], symptoms="sputters")
    assert shared == ["b", "c"] and fix["verified_by"] == ["a"]
    assert (await d["b"].handle(Incoming("m", "b", "op", {"type": "fixes", "problem": "no_fuel"}, True)))[0][
        "verified_by"] == ["a"]
    # c verifies it on its own car too: everyone who has it learns that c vouches for it as well
    _, shared = await d["c"].learned("no_fuel", "refuel", ["add 10 l"])
    assert d["b"].fixes.fixes[fix["id"]]["verified_by"] == ["a", "c"]
    assert d["c"].fixes.fixes[fix["id"]]["verified_by"] == ["a", "c"]
    # a zone cannot vouch for others, and what it sends is checked and cut to size
    r = d["b"].receive_fix({"problem": "flat_tire", "summary": "x" * 999, "steps": ["s" * 999] * 50,
                            "verified_by": ["a", "b", "c"], "platform": {"x": 1}}, "c")
    stored = d["b"].fixes.fixes[r["id"]]
    assert r["ok"] and stored["verified_by"] == ["c"] and stored["origin"] == "c"
    assert len(stored["summary"]) == 300 and len(stored["steps"]) == 20 and len(stored["steps"][0]) == 500
    assert d["b"].receive_fix({"problem": "engine_on_fire", "steps": ["run"]}, "c") == {"ok": False,
                                                                                       "error": "not a fix"}
    assert d["b"].receive_fix("rm -rf /", "c")["ok"] is False


class Liar(CarDoctor):
    async def handle(self, msg: Incoming) -> Any:
        return {"diagnosis": {"problem": "engine_on_fire", "confidence": 1.0}} if self.name == "doc-b" else \
            {"diagnosis": {"problem": "no_fuel", "confidence": "very"}} if self.name == "doc-c" else \
            {"diagnosis": {"problem": "no_fuel", "confidence": 7.0},
             "fixes": [{"problem": "no_fuel", "steps": ["x"], "verified_by": [f"z{i}" for i in range(500)]}, "junk"]}


async def test_malformed_peer_answers_are_ignored_or_clamped(tmp_path):
    mesh = Mesh()
    d = await doctors(tmp_path, mesh, {"a": "flat_tire:0.5", "b": "x:0", "c": "x:0", "d": "x:0"}, cls=Liar)
    r = await d["a"].ask_peers("q", {}, {"problem": "flat_tire", "confidence": 0.5})
    # b: unknown problem, c: confidence is not a number -> not counted; d: confidence clamped to 1,
    # its fix keeps at most 16 vouching zones and the junk entry is dropped
    assert [p["zone"] for p in r.peers] == ["d"] and r.peers[0]["confidence"] == 1.0
    assert len(r.fixes) == 1 and len(r.fixes[0]["verified_by"]) == 16
    assert r.problem == "no_fuel" and r.confidence == 1.0


async def test_accepts_and_settings_follow_the_prefix(tmp_path):
    mesh = Mesh()
    a = (await doctors(tmp_path, mesh, {"a": "healthy:1"}, CAR_THRESHOLD="0.5", CAR_PEER_TIMEOUT_S="3"))["a"]
    assert {"ask", "car.consult", "car.fix", "fixes"} <= set(a.accepts)
    assert a.threshold == 0.5 and a.peer_timeout == 3 and a.share
    assert a.fixes.path == tmp_path / "a" / "car-fixes.json" and a.status() == {"fixes": 0}
    assert (await a.handle(Incoming("m", "b", "x", {"type": "car.consult"}, False)))["diagnosis"] is None
    assert (await a.handle(Incoming("m", "b", "x", {"type": "other"}, False)))["error"].startswith("unknown")

    class Incomplete(PeerAgent):
        peer_capability = "x"
    with pytest.raises(TypeError):
        Incomplete("n", mesh.gw("a"), {})
