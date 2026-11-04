"""Generic federated learning over agent zones.

The agents in this package (:mod:`.client`, :mod:`.aggregator`) only do
communication and the training lifecycle. Everything model specific is a
plugin, selected with ``FL_TASK`` (clients) and ``FL_STRATEGY`` (aggregator):

* :class:`ClientTask` - runs on each client, on that client's local data.
* :class:`Strategy` / :class:`FedAvg` - runs on the aggregator.

Parameters are ``dict[str, numpy.ndarray]``, so any framework fits (numpy,
a PyTorch ``state_dict`` converted with ``.numpy()``, Keras weights, ...).
"""

from __future__ import annotations

import base64
import importlib
import io
from dataclasses import dataclass, field
from typing import Any

import numpy as np

Parameters = dict[str, np.ndarray]
Metrics = dict[str, float]


class ClientTask:
    """Plugin for a client. ``config`` is the environment of the client agent."""

    def __init__(self, config: dict[str, str]):
        self.config = config

    def fit(self, parameters: Parameters, config: dict[str, Any]) -> tuple[Parameters, int, Metrics]:
        """Train from ``parameters`` on local data; return (new parameters, number of samples, metrics)."""
        raise NotImplementedError

    def evaluate(self, parameters: Parameters, config: dict[str, Any]) -> tuple[int, Metrics]:
        """Optional: evaluate ``parameters`` on local data; return (number of samples, metrics)."""
        return 0, {}

    def on_global(self, parameters: Parameters) -> None:
        """Optional: called with the final global model when a run is done."""


@dataclass
class FitResult:
    zone: str
    agent: str
    parameters: Parameters
    num_samples: int
    metrics: Metrics = field(default_factory=dict)


class Strategy:
    """Plugin for the aggregator. ``config`` is the environment of the aggregator agent."""

    def __init__(self, config: dict[str, str]):
        self.config = config

    def initial_parameters(self) -> Parameters:
        raise NotImplementedError

    def configure_fit(self, rnd: int, parameters: Parameters, clients: list[dict[str, Any]]) -> dict[str, Any]:
        """Optional: per-round config sent to every client with ``fl.fit``."""
        return {}

    def aggregate_fit(self, rnd: int, results: list[FitResult], failures: list[str],
                      current: Parameters) -> tuple[Parameters, Metrics]:
        raise NotImplementedError

    def evaluate(self, rnd: int, parameters: Parameters) -> Metrics:
        """Optional: central evaluation of the global model."""
        return {}


class FedAvg(Strategy):
    """Average client parameters weighted by their number of samples."""

    def aggregate_fit(self, rnd: int, results: list[FitResult], failures: list[str],
                      current: Parameters) -> tuple[Parameters, Metrics]:
        total = sum(r.num_samples for r in results)
        if total <= 0:
            raise ValueError("clients reported no samples")
        keys = current.keys() if current else results[0].parameters.keys()
        agg = {k: sum(r.parameters[k].astype(np.float64) * (r.num_samples / total) for r in results)
               for k in keys}
        metrics: Metrics = {"num_samples": float(total)}
        for name in {m for r in results for m in r.metrics}:  # sample-weighted client metrics
            vals = [(r.metrics[name], r.num_samples) for r in results if name in r.metrics]
            metrics[f"client_{name}"] = float(sum(v * n for v, n in vals) / max(sum(n for _, n in vals), 1))
        return agg, metrics


def encode(params: Parameters) -> str:
    buf = io.BytesIO()
    np.savez_compressed(buf, **{k: np.asarray(v, dtype=np.float32) for k, v in params.items()})
    return base64.b64encode(buf.getvalue()).decode()


def decode(blob: str) -> Parameters:
    with np.load(io.BytesIO(base64.b64decode(blob)), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def load_plugin(spec: str, config: dict[str, str]) -> Any:
    """``"tasks.mnist:MnistClient"`` -> instance (constructed with the agent's config)."""
    mod, _, cls = spec.partition(":")
    if not cls:
        raise ValueError(f"plugin spec must look like module:Class, got {spec!r}")
    return getattr(importlib.import_module(mod), cls)(config)


def check_shapes(reference: Parameters, params: Parameters) -> None:
    if set(reference) != set(params):
        raise ValueError(f"parameter names differ: {sorted(set(reference) ^ set(params))[:5]}")
    for k, v in reference.items():
        if params[k].shape != v.shape:
            raise ValueError(f"parameter {k}: shape {params[k].shape} != {v.shape}")
        if not np.all(np.isfinite(params[k])):
            raise ValueError(f"parameter {k} has NaN or inf values")
