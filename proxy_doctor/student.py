"""Student model: softmax regression over diagnostic features.

Tiny on purpose (~2.4k parameters): trains in milliseconds on a laptop CPU,
can be distilled from a large teacher LLM (soft labels) and averaged across
zones with FedAvg. The zone's LLM is used for explanations, not for the
classification itself, so the doctor works even without a GPU or a model.
"""

from __future__ import annotations

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


def benchmark(n: int = 900) -> tuple[np.ndarray, list[str]]:
    """The shared, uniform benchmark every zone and the aggregator score models on: (features, labels)."""
    cases, labels = make_dataset(n, "uniform", seed=424242, label_noise=0.0)
    return featurize_many(cases), labels


def accuracy(model: Student, X: np.ndarray, labels: list[str]) -> float:
    if not len(X):
        return 0.0
    pred = model.predict_proba(X).argmax(axis=1)
    return float(np.mean([CLASSES[i] == lbl for i, lbl in zip(pred, labels)]))
