import numpy as np

from agentzones.federated import FederatedNode, decode_weights, encode_weights, fedavg
from proxy_doctor.student import ProxyStudentTask


def test_fedavg_weighted():
    w = fedavg([({"W": np.ones(2)}, 1), ({"W": np.zeros(2)}, 3)])
    assert np.allclose(w["W"], 0.25)


def test_weights_codec_roundtrip():
    w = {"W": np.random.default_rng(0).normal(size=(3, 4))}
    assert np.allclose(decode_weights(encode_weights(w))["W"], w["W"], atol=1e-6)


def test_federation_beats_local_only_on_skewed_sites(tmp_path):
    profiles = ["dev-laptops", "build-servers", "k8s-cluster"]
    nodes = [FederatedNode(ProxyStudentTask(str(tmp_path), f"z{i}", profile=p, n_samples=300, seed=i), 5, 0.5)
             for i, p in enumerate(profiles)]
    local_only = [n.task.evaluate()["global_acc"] for n in nodes]
    for _ in range(3):
        for n in nodes:
            n.train()
        shared = [dict(n.shared(), zone=i) for i, n in enumerate(nodes)]
        for i, n in enumerate(nodes):
            n.aggregate([s for j, s in enumerate(shared) if j != i])
    fed = [n.task.evaluate()["global_acc"] for n in nodes]
    assert max(local_only) < 0.7
    assert min(fed) > 0.85
    assert len({round(x, 6) for x in fed}) == 1  # synchronous FedAvg -> identical global model
