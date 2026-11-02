"""Step 1 - generate synthetic failure cases and label them with the teacher.

    # with a teacher model (GPU box running vLLM/Ollama, or Azure OpenAI)
    python -m proxy_doctor.distill.make_dataset --n 3000 --out data/distill.jsonl \\
        --teacher-url http://gpu-box:8000/v1 --teacher-model Qwen/Qwen2.5-32B-Instruct

    # offline (no teacher): ground truth + label smoothing
    python -m proxy_doctor.distill.make_dataset --n 3000 --out data/distill.jsonl

Each JSONL row: {"obs", "label" (generator truth), "teacher": {"probs", "problem", "explanation"}}.
Rows where the teacher disagrees with the generator are kept but counted, which
is a useful quality signal for both the generator and the teacher.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..synth import make_dataset
from .teacher import Teacher, oracle_label


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--profile", default="uniform")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", default="data/distill.jsonl")
    ap.add_argument("--teacher-url", default=os.environ.get("TEACHER_URL"))
    ap.add_argument("--teacher-model", default=os.environ.get("TEACHER_MODEL", "qwen2.5:32b"))
    ap.add_argument("--teacher-key", default=os.environ.get("TEACHER_API_KEY", "local"))
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    cases, labels = make_dataset(args.n, args.profile, args.seed, label_noise=0.0)
    teacher = Teacher(args.teacher_url, args.teacher_model, args.teacher_key) if args.teacher_url else None

    def lab(i: int) -> dict:
        if teacher is None:
            return oracle_label(labels[i])
        try:
            return teacher.label(cases[i])
        except Exception as exc:  # keep going; fall back to truth for this row
            print(f"teacher failed on row {i}: {exc}", file=sys.stderr)
            return {**oracle_label(labels[i]), "fallback": True}

    with ThreadPoolExecutor(args.workers) as ex:
        teach = list(ex.map(lab, range(len(cases))))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    disagree = 0
    with out.open("w") as f:
        for obs, y, t in zip(cases, labels, teach):
            disagree += t.get("problem") != y
            f.write(json.dumps({"obs": obs, "label": y, "teacher": t}, default=str) + "\n")
    print(f"wrote {len(cases)} rows to {out}; teacher disagreed with generator on {disagree} "
          f"({disagree / max(len(cases), 1):.1%})")


if __name__ == "__main__":
    main()
