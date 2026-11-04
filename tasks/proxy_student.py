"""FL plugin for the Proxy Doctor student model (softmax regression over diagnostic features).

Each client trains on its own site's incident cases (synthetic stand-in:
``PD_PROFILE`` picks a site profile from proxy_doctor.synth). The aggregator
evaluates on a shared, uniform benchmark. Runs offline in milliseconds, which
makes it the job used by the tests and ``scripts/demo.py``.

Point ``FL_MODEL_OUT`` of the client and ``PD_MODEL_PATH`` of the zone's
proxy-doctor agent at the same file, and the doctor picks up the federated
model when a run is done.
"""

from __future__ import annotations

import os

import numpy as np

from agentzones.fl import ClientTask, FedAvg, Metrics, Parameters
from proxy_doctor.catalog import CLASSES
from proxy_doctor.features import DIM, featurize_many
from proxy_doctor.student import ProxyStudentTask, Student, accuracy
from proxy_doctor.synth import make_dataset


class ProxyStudentClient(ClientTask):
    def __init__(self, config: dict[str, str]):
        super().__init__(config)
        self.local = ProxyStudentTask(data_dir=config.get("DATA_DIR", "/data"),
                                      zone_name=config.get("ZONE") or config.get("AGENT_NAME", "zone"),
                                      profile=config.get("PD_PROFILE", "uniform"),
                                      n_samples=int(config.get("PD_SAMPLES", "400")),
                                      cases_path=config.get("PD_CASES_PATH") or None, warm_start=False)
        self.epochs = int(config.get("FL_LOCAL_EPOCHS", "5"))
        self.lr = float(config.get("FL_LR", "0.5"))

    def fit(self, parameters: Parameters, config: dict) -> tuple[Parameters, int, Metrics]:
        self.local.set_weights({"W": parameters["W"].astype(np.float64)})
        m = self.local.train(int(config.get("epochs", self.epochs)), float(config.get("lr", self.lr)))
        return self.local.get_weights(), self.local.n_samples, {**m, **self.local.evaluate()}

    def evaluate(self, parameters: Parameters, config: dict) -> tuple[int, Metrics]:
        self.local.set_weights({"W": parameters["W"].astype(np.float64)})
        return len(self.local.yh), self.local.evaluate()


class ProxyStudentStrategy(FedAvg):
    def __init__(self, config: dict[str, str]):
        super().__init__(config)
        cases, labels = make_dataset(int(config.get("PD_BENCHMARK", "900")), "uniform", seed=424242, label_noise=0.0)
        self.X, self.y = featurize_many(cases), labels

    def initial_parameters(self) -> Parameters:
        return {"W": np.zeros((DIM, len(CLASSES)))}

    def evaluate(self, rnd: int, parameters: Parameters) -> Metrics:
        return {"global_acc": accuracy(Student(parameters["W"].astype(np.float64)), self.X, self.y)}


if __name__ == "__main__":
    print(ProxyStudentStrategy(dict(os.environ)).evaluate(0, {"W": np.zeros((DIM, len(CLASSES)))}))
