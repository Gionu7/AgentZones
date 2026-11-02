"""Learning loop (verified cases -> local training), robust aggregation, FL persistence."""

import json

import numpy as np

from agentzones.federated import FederatedNode, encode_weights, robust_aggregate
from proxy_doctor.student import ProxyStudentTask
from proxy_doctor.synth import make_dataset

PROFILES = ["dev-laptops", "build-servers", "k8s-cluster"]


def nodes(tmp_path, **kw):
    return [FederatedNode(ProxyStudentTask(str(tmp_path / f"z{i}"), f"z{i}", profile=p, n_samples=300, seed=i),
                          5, 0.5, zone_name=f"z{i}", **kw) for i, p in enumerate(PROFILES)]


def run_rounds(ns, rounds, poison=None):
    for _ in range(rounds):
        for n in ns:
            n.train()
        shared = [dict(n.shared(), zone=n.zone_name) for n in ns]
        if poison:
            shared.append(poison(ns[0]))
        for n in ns:
            n.aggregate([s for s in shared if s["zone"] != n.zone_name])
    return [n.task.evaluate()["global_acc"] for n in ns]


def poisoned(node):
    rng = np.random.default_rng(0)
    w = {k: rng.normal(0, 50, v.shape) for k, v in node.snapshot.items()}
    return {"task": node.task.name, "round": 0, "n_samples": 1_000_000, "weights": encode_weights(w), "zone": "evil"}


def test_poisoned_peer_is_rejected(tmp_path):
    accs = run_rounds(nodes(tmp_path), 3, poison=poisoned)
    assert min(accs) > 0.85
    rec = nodes(tmp_path / "x")[0]
    rec.train()
    out = rec.aggregate([poisoned(rec), dict(rec.shared(), zone="peer1"), dict(rec.shared(), zone="peer2")])
    assert [r["zone"] for r in out["rejected"]] == ["evil"]


def test_naive_fedavg_would_have_been_destroyed(tmp_path):
    accs = run_rounds(nodes(tmp_path, max_update_ratio=0, max_sample_ratio=1e12), 2, poison=poisoned)
    assert max(accs) < 0.5  # what the defences prevent


def test_sample_count_cap_and_median():
    a, b, c = ({"W": np.full(3, v)} for v in (0.0, 1.0, 2.0))
    out, info = robust_aggregate([(a, 100, "a"), (b, 100, "b"), (c, 100_000, "c")], max_update_ratio=0)
    assert info["capped"] == [{"zone": "c", "claimed": 100_000, "counted": 300}]
    assert np.allclose(out["W"], (0 * 100 + 1 * 100 + 2 * 300) / 500)
    out, _ = robust_aggregate([(a, 1, "a"), (b, 1, "b"), (c, 1, "c")], method="median", max_update_ratio=0)
    assert np.allclose(out["W"], 1.0)
    out, _ = robust_aggregate([(a, 1, "a"), (b, 1, "b"), (c, 1, "c")], method="trimmed_mean",
                              trim_fraction=0.34, max_update_ratio=0)
    assert np.allclose(out["W"], 1.0)


def test_state_survives_restart(tmp_path):
    t = ProxyStudentTask(str(tmp_path), "z", profile="dev-laptops", n_samples=200)
    n = FederatedNode(t, 5, 0.5, state_path=tmp_path / "fl.npz", zone_name="z")
    n.train()
    n.aggregate([])
    w = t.get_weights()["W"].copy()
    t2 = ProxyStudentTask(str(tmp_path), "z", profile="dev-laptops", n_samples=200)
    n2 = FederatedNode(t2, 5, 0.5, state_path=tmp_path / "fl.npz", zone_name="z")
    assert n2.round == 1 and np.allclose(t2.get_weights()["W"], w)
    assert n2.history[-1]["phase"] == "restored"


def test_verified_cases_teach_the_local_model(tmp_path):
    # a dev-laptop site has (almost) never seen proxy auth problems
    task = ProxyStudentTask(str(tmp_path), "laptops", profile="dev-laptops", n_samples=300, seed=1)
    cases, labels = make_dataset(200, "uniform", seed=77, label_noise=0.0)
    target = [(c, lab) for c, lab in zip(cases, labels) if lab == "proxy_auth_required"]
    test = target[len(target) // 2:]

    def acc():
        return np.mean([task.model.classify(c)["problem"] == "proxy_auth_required" for c, _ in test])

    before = acc()
    with (tmp_path / "verified_cases.jsonl").open("w") as f:
        for c, lab in target[:len(target) // 2]:
            f.write(json.dumps({"obs": c, "label": lab}) + "\n")
        f.write("not json\n" + json.dumps({"obs": {}, "label": "made_up"}) + "\n")  # ignored
    m = task.train(5, 0.5)
    assert m["new_verified_cases"] == len(target) // 2
    assert task.train(5, 0.5)["new_verified_cases"] == 0  # only new lines are read
    assert task.evaluate()["verified_cases"] == len(target) // 2
    assert acc() > before + 0.3, (before, acc())
