"""Decentralized federated averaging (FedAvg without a coordinator).

A round on a zone has two phases:

1. ``train``     - train the current model on *local* data only and keep a
                   snapshot of the trained weights (the only thing peers see).
2. ``aggregate`` - pull the trained snapshots of all live peers running the
                   same task, average them weighted by sample count, and load
                   the result.

If every zone trains before anyone aggregates, all zones end the round with
the same global model (classic FedAvg). If zones run out of step it degrades
gracefully into gossip averaging, which still converges. Raw data never
leaves the zone; optionally Gaussian noise is added to shared weights.
"""

from __future__ import annotations

import base64
import importlib
import io
import logging
import time
from pathlib import Path
from typing import Any, Protocol

import numpy as np

log = logging.getLogger(__name__)

Weights = dict[str, np.ndarray]


class FederatedTask(Protocol):
    name: str
    n_samples: int

    def get_weights(self) -> Weights: ...
    def set_weights(self, w: Weights) -> None: ...
    def train(self, epochs: int, lr: float) -> dict[str, float]: ...
    def evaluate(self) -> dict[str, float]: ...


def load_task(spec: str, **kwargs: Any) -> FederatedTask:
    mod, _, cls = spec.partition(":")
    return getattr(importlib.import_module(mod), cls)(**kwargs)


def encode_weights(w: Weights) -> str:
    buf = io.BytesIO()
    np.savez_compressed(buf, **{k: v.astype(np.float32) for k, v in w.items()})
    return base64.b64encode(buf.getvalue()).decode()


def decode_weights(s: str) -> Weights:
    with np.load(io.BytesIO(base64.b64decode(s)), allow_pickle=False) as z:
        return {k: z[k].astype(np.float64) for k in z.files}


def fedavg(items: list[tuple[Weights, int]]) -> Weights:
    total = sum(n for _, n in items) or 1
    keys = items[0][0].keys()
    return {k: sum(w[k] * (n / total) for w, n in items) for k in keys}


class FederatedNode:
    def __init__(self, task: FederatedTask, epochs: int, lr: float, dp_noise_std: float = 0.0):
        self.task, self.epochs, self.lr, self.dp_noise_std = task, epochs, lr, dp_noise_std
        self.round = 0
        self.snapshot: Weights = task.get_weights()
        self.history: list[dict[str, Any]] = [{"round": 0, "phase": "init", "t": time.time(), **task.evaluate()}]

    def shared(self) -> dict[str, Any]:
        w = self.snapshot
        if self.dp_noise_std > 0:
            rng = np.random.default_rng()
            w = {k: v + rng.normal(0, self.dp_noise_std, v.shape) for k, v in w.items()}
        return {"task": self.task.name, "round": self.round, "n_samples": self.task.n_samples,
                "weights": encode_weights(w)}

    def train(self) -> dict[str, Any]:
        metrics = self.task.train(self.epochs, self.lr)
        self.snapshot = self.task.get_weights()
        rec = {"round": self.round + 1, "phase": "train", "t": time.time(), **metrics, **self.task.evaluate()}
        self.history.append(rec)
        return rec

    def aggregate(self, peer_payloads: list[dict[str, Any]]) -> dict[str, Any]:
        items = [(self.snapshot, self.task.n_samples)]
        used = []
        for p in peer_payloads:
            if p.get("task") != self.task.name:
                continue
            try:
                items.append((decode_weights(p["weights"]), int(p["n_samples"])))
                used.append(p.get("zone"))
            except Exception as exc:
                log.warning("bad weights from %s: %s", p.get("zone"), exc)
        self.task.set_weights(fedavg(items))
        self.round += 1
        rec = {"round": self.round, "phase": "aggregate", "t": time.time(), "peers": used,
               "total_samples": sum(n for _, n in items), **self.task.evaluate()}
        self.history.append(rec)
        return rec

    def status(self) -> dict[str, Any]:
        return {"task": self.task.name, "round": self.round, "n_samples": self.task.n_samples,
                "metrics": self.task.evaluate(), "history": self.history[-50:]}


def task_kwargs(data: dict[str, Any], data_dir: Path, zone_name: str) -> dict[str, Any]:
    return {**data, "data_dir": str(data_dir), "zone_name": zone_name}
