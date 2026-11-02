"""Step 2b (optional, GPU) - turn the teacher dataset into chat SFT data for a small LLM.

    python -m proxy_doctor.distill.make_sft --data data/distill.jsonl --out data/sft.jsonl

The small model learns to produce the teacher's diagnosis + explanation from an
observation, so the zone's local LLM (e.g. Qwen2.5-0.5B/1.5B) explains like
the teacher while the student classifier stays the source of truth.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..catalog import render_fix
from .teacher import SYSTEM, brief


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/distill.jsonl")
    ap.add_argument("--out", default="data/sft.jsonl")
    args = ap.parse_args()
    rows = [json.loads(line) for line in Path(args.data).read_text().splitlines() if line.strip()]
    with Path(args.out).open("w") as f:
        for r in rows:
            t = r["teacher"]
            problem = t.get("problem") or r["label"]
            answer = {"problem": problem, "confidence": round(max(t["probs"]), 2),
                      "explanation": t.get("explanation", ""), "fix": render_fix(problem, r["obs"])["steps"]}
            f.write(json.dumps({"messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": json.dumps(brief(r["obs"]), default=str)},
                {"role": "assistant", "content": json.dumps(answer)},
            ]}) + "\n")
    print(f"wrote {len(rows)} SFT examples to {args.out}")


if __name__ == "__main__":
    main()
