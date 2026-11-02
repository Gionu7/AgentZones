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
import json
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


def _flat(w: Weights) -> np.ndarray:
    return np.concatenate([w[k].ravel() for k in sorted(w)])


def robust_aggregate(items: list[tuple[Weights, int, str]], method: str = "fedavg", trim_fraction: float = 0.2,
                     max_sample_ratio: float = 3.0, max_update_ratio: float = 5.0,
                     protect: str | None = None) -> tuple[Weights, dict[str, Any]]:
    """Aggregate (weights, n_samples, zone) triples, resisting poisoned or broken peers.

    1. Outlier filter: drop snapshots whose L2 distance to the coordinate-wise median model is more
       than ``max_update_ratio`` x the median distance (``protect``, our own zone, is never dropped).
    2. Sample-count cap (fedavg): nobody may count for more than ``max_sample_ratio`` x the median
       reported count - a zone claiming a million samples cannot take over the average.
    3. Combine: ``fedavg`` (weighted mean), ``median`` or ``trimmed_mean`` (per coordinate).
    """
    info: dict[str, Any] = {"method": method, "rejected": [], "capped": []}
    keys = sorted(items[0][0])
    if len(items) >= 3 and max_update_ratio > 0:
        med = {k: np.median(np.stack([w[k] for w, _, _ in items]), axis=0) for k in keys}
        mflat = _flat(med)
        dists = [float(np.linalg.norm(_flat(w) - mflat)) for w, _, _ in items]
        ref = max(float(np.median(dists)), 1e-6)
        kept = []
        for (w, n, z), d in zip(items, dists):
            if d > max_update_ratio * ref and z != protect:
                info["rejected"].append({"zone": z, "distance_ratio": round(d / ref, 1)})
            else:
                kept.append((w, n, z))
        items = kept
    counts = [max(int(n), 0) for _, n, _ in items]
    cap = max_sample_ratio * float(np.median(counts)) if counts else 0
    capped = []
    for (w, n, z) in items:
        if cap and n > cap:
            info["capped"].append({"zone": z, "claimed": n, "counted": int(cap)})
            n = int(cap)
        capped.append((w, n, z))
    stack = {k: np.stack([w[k] for w, _, _ in capped]) for k in keys}
    if method == "median":
        out = {k: np.median(v, axis=0) for k, v in stack.items()}
    elif method == "trimmed_mean":
        m = len(capped)
        cut = min(int(m * trim_fraction), (m - 1) // 2)
        out = {k: np.sort(v, axis=0)[cut:m - cut].mean(axis=0) for k, v in stack.items()}
    else:
        out = fedavg([(w, n) for w, n, _ in capped])
    info["used"] = [z for _, _, z in capped]
    return out, info


class FederatedNode:
    def __init__(self, task: FederatedTask, epochs: int, lr: float, dp_noise_std: float = 0.0,
                 state_path: Path | None = None, aggregation: str = "fedavg", trim_fraction: float = 0.2,
                 max_sample_ratio: float = 3.0, max_update_ratio: float = 5.0, zone_name: str = ""):
        self.task, self.epochs, self.lr, self.dp_noise_std = task, epochs, lr, dp_noise_std
        self.aggregation, self.trim_fraction = aggregation, trim_fraction
        self.max_sample_ratio, self.max_update_ratio = max_sample_ratio, max_update_ratio
        self.zone_name = zone_name
        self.state_path = state_path
        self.round = 0
        self.snapshot: Weights = task.get_weights()
        self.history: list[dict[str, Any]] = []
        if not self._load():
            self.history = [{"round": 0, "phase": "init", "t": time.time(), **task.evaluate()}]

    # persistence: the model survives restarts (weights + snapshot + round + history)
    def _save(self) -> None:
        if not self.state_path:
            return
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_name(self.state_path.name + ".tmp.npz")
        arrays = {f"model__{k}": v for k, v in self.task.get_weights().items()}
        arrays.update({f"snap__{k}": v for k, v in self.snapshot.items()})
        np.savez_compressed(tmp, **arrays, meta=np.array(json.dumps(
            {"task": self.task.name, "round": self.round, "history": self.history[-200:]}, default=float)))
        tmp.replace(self.state_path)

    def _load(self) -> bool:
        if not self.state_path or not self.state_path.exists():
            return False
        try:
            with np.load(self.state_path, allow_pickle=False) as z:
                meta = json.loads(str(z["meta"]))
                if meta.get("task") != self.task.name:
                    return False
                model = {k[7:]: z[k] for k in z.files if k.startswith("model__")}
                snap = {k[6:]: z[k] for k in z.files if k.startswith("snap__")}
            cur = self.task.get_weights()
            if set(model) != set(cur) or any(model[k].shape != cur[k].shape for k in cur):
                return False
            self.task.set_weights(model)
            self.snapshot = snap or model
            self.round = int(meta.get("round", 0))
            self.history = list(meta.get("history", []))
            self.history.append({"round": self.round, "phase": "restored", "t": time.time(), **self.task.evaluate()})
            return True
        except Exception as exc:
            log.warning("could not restore FL state from %s: %s", self.state_path, exc)
            return False

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
        self._save()
        return rec

    def aggregate(self, peer_payloads: list[dict[str, Any]]) -> dict[str, Any]:
        me = self.zone_name or "self"
        items = [(self.snapshot, self.task.n_samples, me)]
        for p in peer_payloads:
            if p.get("task") != self.task.name:
                continue
            try:
                w = decode_weights(p["weights"])
                if set(w) != set(self.snapshot) or any(w[k].shape != self.snapshot[k].shape for k in w):
                    raise ValueError("shape mismatch")
                if not all(np.isfinite(v).all() for v in w.values()):
                    raise ValueError("non-finite weights")
                items.append((w, int(p["n_samples"]), p.get("zone") or "?"))
            except Exception as exc:
                log.warning("bad weights from %s: %s", p.get("zone"), exc)
        agg, info = robust_aggregate(items, self.aggregation, self.trim_fraction, self.max_sample_ratio,
                                     self.max_update_ratio, protect=me)
        self.task.set_weights(agg)
        self.round += 1
        rec = {"round": self.round, "phase": "aggregate", "t": time.time(),
               "peers": [z for z in info["used"] if z != me], "rejected": info["rejected"], "capped": info["capped"],
               "method": info["method"], "total_samples": sum(n for _, n, _ in items), **self.task.evaluate()}
        self.history.append(rec)
        self._save()
        return rec

    def status(self) -> dict[str, Any]:
        return {"task": self.task.name, "round": self.round, "n_samples": self.task.n_samples,
                "aggregation": self.aggregation, "metrics": self.task.evaluate(), "history": self.history[-50:]}


def task_kwargs(data: dict[str, Any], data_dir: Path, zone_name: str) -> dict[str, Any]:
    return {**data, "data_dir": str(data_dir), "zone_name": zone_name}
