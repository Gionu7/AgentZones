"""Student model: softmax regression over diagnostic features.

Tiny on purpose (~2.4k parameters): trains in milliseconds on a laptop CPU,
can be distilled from a large teacher LLM (soft labels) and averaged across
zones with FedAvg. The zone's LLM is used for explanations, not for the
classification itself, so the doctor works even without a GPU or a model.
"""

from __future__ import annotations

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

    def fit(self, X: np.ndarray, Y: np.ndarray, epochs: int = 50, lr: float = 0.5) -> float:
        """Y: (n, C) one-hot or soft targets (distillation). Full-batch GD."""
        n = max(len(X), 1)
        loss = 0.0
        for _ in range(epochs):
            P = self.predict_proba(X)
            loss = float(-(Y * np.log(P + 1e-12)).sum() / n)
            grad = X.T @ (P - Y) / n + self.l2 * self.W
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


class ProxyStudentTask:
    """Federated task: each zone trains the student on its own (skewed) cases.

    options (tasks/proxy_student.py passes PD_PROFILE, PD_SAMPLES and PD_CASES_PATH):
      profile:   site profile in proxy_doctor.synth.PROFILES (dev-laptops, ...)
      n_samples: number of local cases (synthetic stand-in for local incident logs)
      seed:      RNG seed for the local data
      cases_path: optional JSONL of real local cases {"obs":..., "label":...}
      init_from: optional pretrained student .npz (e.g. the distilled one)
      warm_start: train on local data at start-up (default true), so the zone
                  has a usable local-only model before the first FL round
    """

    name = "proxy-doctor-student/v1"

    def __init__(self, data_dir: str, zone_name: str, profile: str = "uniform", n_samples: int = 400,
                 seed: int | None = None, cases_path: str | None = None, init_from: str | None = None,
                 test_n: int = 900, warm_start: bool = True, **_: Any):
        seed = seed if seed is not None else zlib.crc32(zone_name.encode()) % 10_000  # stable per zone
        cases, labels = make_dataset(n_samples, profile, seed)
        if cases_path and Path(cases_path).exists():
            import json
            for line in Path(cases_path).read_text().splitlines():
                if line.strip():
                    rec = json.loads(line)
                    cases.append(rec["obs"])
                    labels.append(rec["label"])
        split = int(len(cases) * 0.8)
        self.X, self.y = featurize_many(cases[:split]), labels[:split]
        self.Xh, self.yh = featurize_many(cases[split:]), labels[split:]
        gc, gl = make_dataset(test_n, "uniform", seed=424242, label_noise=0.0)  # shared benchmark
        self.Xg, self.yg = featurize_many(gc), gl
        self.model = Student.load(init_from) if init_from and Path(init_from).exists() else Student()
        self.n_samples = len(self.y)
        self.profile = profile
        if warm_start and not (init_from and Path(init_from).exists()):
            self.model.fit(self.X, one_hot(self.y), epochs=50, lr=0.5)

    def get_weights(self) -> dict[str, np.ndarray]:
        return {"W": self.model.W.copy()}

    def set_weights(self, w: dict[str, np.ndarray]) -> None:
        self.model.W = w["W"].copy()

    def train(self, epochs: int, lr: float) -> dict[str, float]:
        loss = self.model.fit(self.X, one_hot(self.y), epochs=epochs * 10, lr=lr)
        return {"loss": round(loss, 4)}

    def evaluate(self) -> dict[str, float]:
        return {"local_acc": round(accuracy(self.model, self.Xh, self.yh), 4),
                "global_acc": round(accuracy(self.model, self.Xg, self.yg), 4)}
