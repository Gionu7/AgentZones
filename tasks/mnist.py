"""Example FL plugin: a numpy MLP (784-128-10) on MNIST.

Each client holds one shard of the training set: ``FL_CLIENT_INDEX`` of
``FL_NUM_CLIENTS`` (default: 3 shards). Set ``LABELS=0,1,2,3`` on a client to
keep only those digits (non-IID data). The aggregator evaluates the global
model on the 10k test set.

Data: ``MNIST_URL`` is downloaded once to ``MNIST_PATH`` and checked against
its SHA-256. Behind a TLS-inspecting proxy, point ``SSL_CERT_FILE`` at the
corporate CA bundle (zone/base.yml mounts ``HOST_CA_BUNDLE`` there).

Other settings: FL_LOCAL_EPOCHS (1), FL_LR (0.1), FL_BATCH (64), MNIST_LIMIT (cap on train samples).
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path

import numpy as np

from agentzones.fl import ClientTask, FedAvg, Metrics, Parameters

log = logging.getLogger(__name__)

URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
SHA256 = "731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1"
HIDDEN = 128


def load_mnist(config: dict[str, str]) -> dict[str, np.ndarray]:
    path = Path(config.get("MNIST_PATH", "/data/mnist.npz"))
    if not path.exists():
        import httpx

        url = config.get("MNIST_URL", URL)
        log.info("downloading %s", url)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        with httpx.stream("GET", url, follow_redirects=True, timeout=120) as r:  # honours HTTPS_PROXY, SSL_CERT_FILE
            r.raise_for_status()
            with tmp.open("wb") as f:
                for chunk in r.iter_bytes():
                    f.write(chunk)
        digest = hashlib.sha256(tmp.read_bytes()).hexdigest()
        if config.get("MNIST_SHA256", SHA256) and digest != config.get("MNIST_SHA256", SHA256):
            tmp.unlink()
            raise ValueError(f"MNIST download has sha256 {digest}, expected {SHA256}")
        tmp.replace(path)
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _prep(x: np.ndarray) -> np.ndarray:
    return (x.reshape(len(x), -1) / 255.0).astype(np.float32)


def init_parameters(seed: int = 0) -> Parameters:
    rng = np.random.default_rng(seed)
    return {"W1": (rng.standard_normal((784, HIDDEN)) * np.sqrt(2 / 784)).astype(np.float32),
            "b1": np.zeros(HIDDEN, np.float32),
            "W2": (rng.standard_normal((HIDDEN, 10)) * np.sqrt(2 / HIDDEN)).astype(np.float32),
            "b2": np.zeros(10, np.float32)}


def forward(p: Parameters, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h = np.maximum(X @ p["W1"] + p["b1"], 0)
    z = h @ p["W2"] + p["b2"]
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return h, e / e.sum(axis=1, keepdims=True)


def evaluate(p: Parameters, X: np.ndarray, y: np.ndarray) -> Metrics:
    if not len(X):
        return {}
    _, P = forward(p, X)
    return {"loss": float(-np.log(P[np.arange(len(y)), y] + 1e-9).mean()), "acc": float((P.argmax(1) == y).mean())}


def sgd(p: Parameters, X: np.ndarray, y: np.ndarray, epochs: int, lr: float, batch: int, seed: int) -> Parameters:
    p = {k: v.astype(np.float32).copy() for k, v in p.items()}
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        order = rng.permutation(len(X))
        for i in range(0, len(X), batch):
            idx = order[i:i + batch]
            xb, yb = X[idx], y[idx]
            h, P = forward(p, xb)
            d2 = P
            d2[np.arange(len(yb)), yb] -= 1
            d2 /= len(yb)
            dh = (d2 @ p["W2"].T) * (h > 0)
            p["W2"] -= lr * h.T @ d2
            p["b2"] -= lr * d2.sum(0)
            p["W1"] -= lr * xb.T @ dh
            p["b1"] -= lr * dh.sum(0)
    return p


class MnistClient(ClientTask):
    def __init__(self, config: dict[str, str]):
        super().__init__(config)
        data = load_mnist(config)
        idx, n = int(config.get("FL_CLIENT_INDEX", "0")), int(config.get("FL_NUM_CLIENTS", "3"))
        X, y = _prep(data["x_train"])[idx::n], data["y_train"][idx::n].astype(np.int64)
        labels = [int(v) for v in config.get("LABELS", "").split(",") if v.strip()]
        if labels:
            keep = np.isin(y, labels)
            X, y = X[keep], y[keep]
        limit = int(config.get("MNIST_LIMIT", "0"))
        if limit:
            X, y = X[:limit], y[:limit]
        split = int(len(X) * 0.9)
        self.X, self.y, self.Xv, self.yv = X[:split], y[:split], X[split:], y[split:]
        self.epochs = int(config.get("FL_LOCAL_EPOCHS", "1"))
        self.lr, self.batch = float(config.get("FL_LR", "0.1")), int(config.get("FL_BATCH", "64"))
        self.seed = idx
        log.info("mnist client shard %d/%d: %d train samples, labels %s", idx, n, len(self.X), labels or "all")

    def fit(self, parameters: Parameters, config: dict) -> tuple[Parameters, int, Metrics]:
        rnd = int(config.get("round", 0))
        new = sgd(parameters, self.X, self.y, int(config.get("epochs", self.epochs)), float(config.get("lr", self.lr)),
                  self.batch, self.seed * 1000 + rnd)
        m = evaluate(new, self.Xv, self.yv)
        return new, len(self.X), {"val_loss": m.get("loss", 0.0), "val_acc": m.get("acc", 0.0)}

    def evaluate(self, parameters: Parameters, config: dict) -> tuple[int, Metrics]:
        return len(self.Xv), evaluate(parameters, self.Xv, self.yv)


class MnistStrategy(FedAvg):
    def __init__(self, config: dict[str, str]):
        super().__init__(config)
        data = load_mnist(config)
        self.Xt, self.yt = _prep(data["x_test"]), data["y_test"].astype(np.int64)

    def initial_parameters(self) -> Parameters:
        return init_parameters(int(self.config.get("FL_SEED", "0")))

    def configure_fit(self, rnd: int, parameters: Parameters, clients: list) -> dict:
        return {"round": rnd}

    def evaluate(self, rnd: int, parameters: Parameters) -> Metrics:
        m = evaluate(parameters, self.Xt, self.yt)
        return {"test_loss": m["loss"], "test_acc": m["acc"]}


if __name__ == "__main__":  # quick offline check: python -m tasks.mnist
    cfg = {**os.environ, "MNIST_LIMIT": os.environ.get("MNIST_LIMIT", "6000")}
    s, clients = MnistStrategy(cfg), [MnistClient({**cfg, "FL_CLIENT_INDEX": str(i)}) for i in range(3)]
    p = s.initial_parameters()
    for r in range(1, 4):
        from agentzones.fl import FitResult
        res = [FitResult("local", f"c{i}", *c.fit(p, {"round": r})) for i, c in enumerate(clients)]
        p, _ = s.aggregate_fit(r, res, [], p)
        print(r, s.evaluate(r, p))
