"""Student model: softmax regression over diagnostic features.

Tiny on purpose (~2.4k parameters): trains in milliseconds on a laptop CPU,
can be distilled from a large teacher LLM (soft labels) and averaged across
zones with FedAvg. The zone's LLM is used for explanations, not for the
classification itself, so the doctor works even without a GPU or a model.
"""

from __future__ import annotations

import json
import zlib
from pathlib import Path
from typing import Any

import numpy as np

from .catalog import CLASSES
from .features import DIM, featurize, featurize_many
from .synth import make_dataset


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


class Student:
    def __init__(self, W: np.ndarray | None = None, l2: float = 1e-3):
        self.W = np.zeros((DIM, len(CLASSES))) if W is None else W
        self.l2 = l2

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return softmax(X @ self.W)

    def fit(self, X: np.ndarray, Y: np.ndarray, epochs: int = 50, lr: float = 0.5,
            sample_weight: np.ndarray | None = None) -> float:
        """Y: (n, C) one-hot or soft targets (distillation). Full-batch GD, optional per-sample weights."""
        w = np.ones(len(X)) if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
        total = max(float(w.sum()), 1e-9)
        loss = 0.0
        for _ in range(epochs):
            P = self.predict_proba(X)
            loss = float(-(w[:, None] * Y * np.log(P + 1e-12)).sum() / total)
            grad = X.T @ ((P - Y) * w[:, None]) / total + self.l2 * self.W
            self.W -= lr * grad
        return loss

    def classify(self, obs: dict[str, Any]) -> dict[str, Any]:
        p = self.predict_proba(featurize(obs)[None, :])[0]
        order = np.argsort(-p)
        return {"problem": CLASSES[order[0]], "confidence": float(p[order[0]]),
                "top": [{"problem": CLASSES[i], "p": round(float(p[i]), 4)} for i in order[:3]]}

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, W=self.W, classes=np.array(CLASSES))

    @classmethod
    def load(cls, path: str | Path) -> "Student":
        with np.load(path, allow_pickle=False) as z:
            if list(z["classes"]) != CLASSES:
                raise ValueError("class list mismatch")
            return cls(z["W"])


def one_hot(labels: list[str]) -> np.ndarray:
    Y = np.zeros((len(labels), len(CLASSES)))
    for i, lbl in enumerate(labels):
        Y[i, CLASSES.index(lbl)] = 1.0
    return Y


def accuracy(model: Student, X: np.ndarray, labels: list[str]) -> float:
    if not len(X):
        return 0.0
    pred = model.predict_proba(X).argmax(axis=1)
    return float(np.mean([CLASSES[i] == lbl for i, lbl in zip(pred, labels)]))


VERIFIED_CASES = "verified_cases.jsonl"


class ProxyStudentTask:
    """Federated task: each zone trains the student on its own cases.

    Local data = synthetic cases of the site's profile (a bootstrap stand-in) plus
    every *verified* diagnosis the Proxy Doctor made on this site: when a fix is
    applied and the checks pass, the observation and the confirmed problem class are
    appended to ``<data_dir>/verified_cases.jsonl``. They are picked up at every
    training phase, weighted ``verified_weight`` times a synthetic case, and never
    leave the zone - only model weights do. Over time, lower ``n_samples`` so real
    incidents replace the synthetic bootstrap.

    data options (zone.yaml ``federated.data``):
      profile:   site profile in proxy_doctor.synth.PROFILES (dev-laptops, ...)
      n_samples: number of synthetic bootstrap cases (0 = only real verified cases)
      seed:      RNG seed for the synthetic cases
      cases_path: JSONL of real local cases {"obs":..., "label":...} (default: <data_dir>/verified_cases.jsonl)
      verified_weight: weight of a real case relative to a synthetic one (default 3)
      init_from: optional pretrained student .npz (e.g. the distilled one)
      warm_start: train on local data at start-up (default true), so the zone
                  has a usable local-only model before the first FL round
    """

    name = "proxy-doctor-student/v1"

    def __init__(self, data_dir: str, zone_name: str, profile: str = "uniform", n_samples: int = 400,
                 seed: int | None = None, cases_path: str | None = None, init_from: str | None = None,
                 test_n: int = 900, warm_start: bool = True, verified_weight: float = 3.0, **_: Any):
        seed = seed if seed is not None else zlib.crc32(zone_name.encode()) % 10_000  # stable per zone
        cases, labels = make_dataset(n_samples, profile, seed) if n_samples > 0 else ([], [])
        split = int(len(cases) * 0.8)
        self.X, self.y = featurize_many(cases[:split]), labels[:split]
        self.Xh, self.yh = featurize_many(cases[split:]), labels[split:]
        gc, gl = make_dataset(test_n, "uniform", seed=424242, label_noise=0.0)  # shared benchmark
        self.Xg, self.yg = featurize_many(gc), gl
        self.cases_path = Path(cases_path) if cases_path else Path(data_dir) / VERIFIED_CASES
        self.verified_weight = float(verified_weight)
        self.Xv, self.yv = np.zeros((0, DIM)), []
        self._lines_read = 0
        self.refresh()
        self.model = Student.load(init_from) if init_from and Path(init_from).exists() else Student()
        self.profile = profile
        if warm_start and not (init_from and Path(init_from).exists()) and self.n_samples:
            self._fit(epochs=50, lr=0.5)

    @property
    def n_samples(self) -> int:
        return len(self.y) + len(self.yv)

    def refresh(self) -> int:
        """Load verified cases appended since the last call. Returns how many are new."""
        if not self.cases_path.exists():
            return 0
        lines = self.cases_path.read_text().splitlines()
        new_obs, new_labels = [], []
        for line in lines[self._lines_read:]:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("label") in CLASSES and isinstance(rec.get("obs"), dict):
                new_obs.append(rec["obs"])
                new_labels.append(rec["label"])
        self._lines_read = len(lines)
        if new_obs:
            self.Xv = np.vstack([self.Xv, featurize_many(new_obs)])
            self.yv = self.yv + new_labels
        return len(new_obs)

    def _fit(self, epochs: int, lr: float) -> float:
        X = np.vstack([self.X, self.Xv]) if len(self.yv) else self.X
        y = list(self.y) + list(self.yv)
        if not y:
            return 0.0
        w = np.concatenate([np.ones(len(self.y)), np.full(len(self.yv), self.verified_weight)])
        return self.model.fit(X, one_hot(y), epochs=epochs, lr=lr, sample_weight=w)

    def get_weights(self) -> dict[str, np.ndarray]:
        return {"W": self.model.W.copy()}

    def set_weights(self, w: dict[str, np.ndarray]) -> None:
        self.model.W = w["W"].copy()

    def train(self, epochs: int, lr: float) -> dict[str, float]:
        new = self.refresh()
        loss = self._fit(epochs=epochs * 10, lr=lr)
        return {"loss": round(loss, 4), "new_verified_cases": new}

    def evaluate(self) -> dict[str, float]:
        out = {"local_acc": round(accuracy(self.model, self.Xh, self.yh), 4),
               "global_acc": round(accuracy(self.model, self.Xg, self.yg), 4),
               "verified_cases": len(self.yv)}
        if self.yv:  # how well the current model fits this site's real, confirmed cases
            out["verified_acc"] = round(accuracy(self.model, self.Xv, self.yv), 4)
        return out
