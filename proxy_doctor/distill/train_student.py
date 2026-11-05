"""Step 2a - distill the teacher into the tiny student classifier (CPU, seconds).

    python -m proxy_doctor.distill.train_student --data data/distill.jsonl --out models/student.npz

Loss = alpha * CE(teacher soft labels) + (1 - alpha) * CE(generator labels).
Use the result as ``PD_MODEL_PATH`` of the proxy-doctor agent, or as ``PD_INIT_FROM`` of the
proxy-student aggregator so federated rounds start from the distilled model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..features import featurize_many
from ..student import Student, accuracy, benchmark, one_hot


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/distill.jsonl")
    ap.add_argument("--out", default="models/student.npz")
    ap.add_argument("--alpha", type=float, default=0.7)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--lr", type=float, default=0.5)
    args = ap.parse_args()

    rows = [json.loads(line) for line in Path(args.data).read_text().splitlines() if line.strip()]
    X = featurize_many([r["obs"] for r in rows])
    Y = args.alpha * np.array([r["teacher"]["probs"] for r in rows]) + \
        (1 - args.alpha) * one_hot([r["label"] for r in rows])
    m = Student()
    loss = m.fit(X, Y, epochs=args.epochs, lr=args.lr)
    acc = accuracy(m, *benchmark())
    m.save(args.out)
    print(f"student: {m.W.size} parameters, loss {loss:.3f}, benchmark accuracy {acc:.1%} -> {args.out}")


if __name__ == "__main__":
    main()
