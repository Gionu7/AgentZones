"""FL plugin for the Proxy Doctor student model (softmax regression over diagnostic features).

Each client trains on its own site's incident cases: ``PD_PROFILE`` picks a site profile from
proxy_doctor.synth (a synthetic stand-in, ``PD_SAMPLES`` cases, seeded per zone), and
``PD_CASES_PATH`` adds real ones (JSONL, one ``{"obs": ..., "label": ...}`` per line). The aggregator
evaluates on a shared, uniform benchmark. Runs offline in milliseconds, which makes it the job used by
the tests and ``scripts/demo.py``.

Point ``FL_MODEL_OUT`` of the client and ``PD_MODEL_PATH`` of the zone's
proxy-doctor agent at the same file, and the doctor picks up the federated
model when a run is done.
"""

from __future__ import annotations

import json
import os
import zlib
from pathlib import Path

import numpy as np

from agentzones.fl import ClientTask, FedAvg, Metrics, Parameters
from proxy_doctor.catalog import CLASSES
from proxy_doctor.features import DIM, featurize_many
from proxy_doctor.student import Student, accuracy, benchmark, one_hot
from proxy_doctor.synth import make_dataset


class ProxyStudentClient(ClientTask):
    def __init__(self, config: dict[str, str]):
        super().__init__(config)
        zone = config.get("ZONE") or config.get("AGENT_NAME", "zone")
        cases, labels = make_dataset(int(config.get("PD_SAMPLES", "400")), config.get("PD_PROFILE", "uniform"),
                                     seed=zlib.crc32(zone.encode()) % 10_000)  # stable per zone
        path = Path(config.get("PD_CASES_PATH") or "")
        if path.is_file():
            for line in path.read_text().splitlines():
                if line.strip():
                    rec = json.loads(line)
                    cases.append(rec["obs"])
                    labels.append(rec["label"])
        split = int(len(cases) * 0.8)  # the rest is held out for local_acc
        self.X, self.Y = featurize_many(cases[:split]), one_hot(labels[:split])
        self.Xh, self.yh = featurize_many(cases[split:]), labels[split:]
        self.Xb, self.yb = benchmark()
        self.model = Student()
        self.epochs = int(config.get("FL_LOCAL_EPOCHS", "5"))
        self.lr = float(config.get("FL_LR", "0.5"))

    def _metrics(self) -> Metrics:
        return {"local_acc": round(accuracy(self.model, self.Xh, self.yh), 4),
                "global_acc": round(accuracy(self.model, self.Xb, self.yb), 4)}

    def fit(self, parameters: Parameters, config: dict) -> tuple[Parameters, int, Metrics]:
        self.model.W = parameters["W"].astype(np.float64)  # a copy: training does not touch the input
        loss = self.model.fit(self.X, self.Y, epochs=int(config.get("epochs", self.epochs)) * 10,
                              lr=float(config.get("lr", self.lr)))
        return {"W": self.model.W.copy()}, len(self.X), {"loss": round(loss, 4), **self._metrics()}

    def evaluate(self, parameters: Parameters, config: dict) -> tuple[int, Metrics]:
        self.model.W = parameters["W"].astype(np.float64)
        return len(self.yh), self._metrics()


class ProxyStudentStrategy(FedAvg):
    def __init__(self, config: dict[str, str]):
        super().__init__(config)
        self.X, self.y = benchmark(int(config.get("PD_BENCHMARK", "900")))

    def initial_parameters(self) -> Parameters:
        """Start from the distilled student (PD_INIT_FROM, e.g. models/student.npz) or from zeros."""
        path = self.config.get("PD_INIT_FROM", "")
        if path and os.path.exists(path):
            return {"W": Student.load(path).W}
        return {"W": np.zeros((DIM, len(CLASSES)))}

    def evaluate(self, rnd: int, parameters: Parameters) -> Metrics:
        return {"global_acc": accuracy(Student(parameters["W"].astype(np.float64)), self.X, self.y)}


if __name__ == "__main__":
    print(ProxyStudentStrategy(dict(os.environ)).evaluate(0, {"W": np.zeros((DIM, len(CLASSES)))}))
