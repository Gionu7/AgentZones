"""Shared store of *verified fixes*.

A fix enters the store when an agent applied it and a re-run of the
diagnostics passed. It is then pushed to peers. Receiving zones keep the
fix with the list of zones that verified it; they never auto-apply it - a
local agent must re-verify before adding its own name to ``verified_by``.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from pathlib import Path
from typing import Any

_WORD = re.compile(r"[a-z0-9_]{3,}")


def _tokens(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


def fix_id(problem: str, steps: list[str]) -> str:
    return hashlib.sha256((problem + "\n" + "\n".join(steps)).encode()).hexdigest()[:16]


class FixStore:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.fixes: dict[str, dict[str, Any]] = {}
        if self.path.exists():
            try:
                self.fixes = json.loads(self.path.read_text())
            except json.JSONDecodeError:
                self.fixes = {}

    def _save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.fixes, indent=2))
        tmp.replace(self.path)

    def add(self, *, problem: str, summary: str, steps: list[str], symptoms: str = "",
            verified_by: list[str] | None = None, origin: str = "", platform: str = "any",
            evidence: dict[str, Any] | None = None) -> dict[str, Any]:
        fid = fix_id(problem, steps)
        with self._lock:
            cur = self.fixes.get(fid)
            if cur:
                cur["verified_by"] = sorted(set(cur["verified_by"]) | set(verified_by or []))
                cur["updated"] = time.time()
            else:
                cur = self.fixes[fid] = {
                    "id": fid, "problem": problem, "summary": summary, "steps": steps,
                    "symptoms": symptoms, "platform": platform, "origin": origin,
                    "verified_by": sorted(set(verified_by or [])), "evidence": evidence or {},
                    "created": time.time(), "updated": time.time(),
                }
            self._save()
            return cur

    def merge(self, fix: dict[str, Any]) -> dict[str, Any]:
        """Merge a fix received from a peer."""
        return self.add(problem=fix["problem"], summary=fix.get("summary", ""), steps=list(fix["steps"]),
                        symptoms=fix.get("symptoms", ""), verified_by=fix.get("verified_by", []),
                        origin=fix.get("origin", ""), platform=fix.get("platform", "any"),
                        evidence=fix.get("evidence"))

    def search(self, problem: str | None = None, text: str = "", limit: int = 5) -> list[dict[str, Any]]:
        q = _tokens(text)
        scored = []
        for f in self.fixes.values():
            score = len(f["verified_by"]) * 0.1
            if problem and f["problem"] == problem:
                score += 2.0
            elif problem:
                continue
            if q:
                ft = _tokens(f["summary"] + " " + f["symptoms"] + " " + " ".join(f["steps"]))
                score += len(q & ft) / max(len(q), 1)
            scored.append((score, f))
        scored.sort(key=lambda x: -x[0])
        return [f for _, f in scored[:limit]]

    def all(self) -> list[dict[str, Any]]:
        return sorted(self.fixes.values(), key=lambda f: -f["updated"])
