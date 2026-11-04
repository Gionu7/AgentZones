import numpy as np
import pytest

from agentzones.fl import FedAvg, FitResult, check_shapes, decode, encode


def test_fedavg_weights_by_samples_and_averages_metrics():
    s = FedAvg({})
    a = FitResult("az", "c", {"w": np.array([0.0, 0.0])}, 100, {"acc": 0.5})
    b = FitResult("b", "c", {"w": np.array([3.0, 6.0])}, 200, {"acc": 0.8})
    params, metrics = s.aggregate_fit(1, [a, b], [], {"w": np.zeros(2)})
    assert np.allclose(params["w"], [2.0, 4.0])
    assert metrics["num_samples"] == 300 and metrics["client_acc"] == pytest.approx(0.7)


def test_encode_roundtrip_and_shape_check():
    p = {"W": np.arange(6, dtype=np.float32).reshape(2, 3), "b": np.ones(3)}
    q = decode(encode(p))
    assert set(q) == {"W", "b"} and np.allclose(q["W"], p["W"])
    check_shapes(p, q)
    with pytest.raises(ValueError):
        check_shapes(p, {"W": np.zeros((3, 2)), "b": np.ones(3)})
    with pytest.raises(ValueError):
        check_shapes(p, {"W": p["W"], "b": np.array([1, np.nan, 1])})
    with pytest.raises(ValueError):
        check_shapes(p, {"W": p["W"]})
