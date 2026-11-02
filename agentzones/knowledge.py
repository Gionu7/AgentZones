"""Shared store of *verified fixes*, backed by signed attestations.

A fix enters the store when an agent applied it and a re-run of the
diagnostics passed. The zone then signs an *attestation* over the fix
(id, problem, hash of the steps) with its Ed25519 key and pushes the fix to
peers. ``verified_by`` is never taken from the sender: it is recomputed from
the attestations whose signatures check out against each attesting zone's
pinned public key. A fix with no valid attestation is refused.

Receiving zones never auto-apply a fix. Domain validators (e.g. Proxy Doctor
only accepts steps that match its catalog templates) run before a peer's fix
is stored, and a local agent must re-verify before adding its own attestation.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable

_WORD = re.compile(r"[a-z0-9_]{3,}")

# (zone, pubkey_b64, data, sig_b64) -> bool
SigVerifier = Callable[[str, bytes, str], bool]


def _tokens(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


def fix_id(problem: str, steps: list[str]) -> str:
    return hashlib.sha256((problem + "\n" + "\n".join(steps)).encode()).hexdigest()[:16]


def attestation_payload(fid: str, problem: str, steps: list[str], zone: str, ts: float) -> bytes:
    steps_hash = hashlib.sha256(json.dumps(steps).encode()).hexdigest()
    return f"agentzones-fix-attestation/v1\n{fid}\n{problem}\n{steps_hash}\n{zone}\n{ts:.3f}".encode()


class FixRejected(ValueError):
    pass


class FixStore:
    def __init__(self, path: Path, verify_attestation: SigVerifier | None = None):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        # (zone, data, sig) -> bool; None = accept everything (only for tools/tests)
        self.verify_attestation = verify_attestation
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

    def _valid(self, fix: dict[str, Any], att: dict[str, Any]) -> bool:
        try:
            data = attestation_payload(fix["id"], fix["problem"], fix["steps"], att["zone"], float(att["ts"]))
        except (KeyError, TypeError, ValueError):
            return False
        if self.verify_attestation is None:
            return True
        return self.verify_attestation(att["zone"], data, att.get("sig", ""))

    def _upsert(self, base: dict[str, Any], attestations: list[dict[str, Any]]) -> tuple[dict[str, Any], int, int]:
        """Insert/merge a fix, keeping only valid attestations. Returns (fix, n_new_valid, n_invalid)."""
        fid = fix_id(base["problem"], base["steps"])
        probe = {**base, "id": fid}
        good = [a for a in attestations if self._valid(probe, a)]
        with self._lock:
            cur = self.fixes.get(fid)
            if cur is None:
                if not good:
                    raise FixRejected("no valid attestation")
                cur = self.fixes[fid] = {**probe, "attestations": [], "created": time.time()}
            have = {a["zone"] for a in cur["attestations"]}
            new = [a for a in good if a["zone"] not in have]
            cur["attestations"].extend({"zone": a["zone"], "ts": a["ts"], "sig": a["sig"]} for a in new)
            cur["verified_by"] = sorted({a["zone"] for a in cur["attestations"]})
            cur["updated"] = time.time()
            self._save()
            return cur, len(new), len(attestations) - len(good)

    def add_verified(self, *, problem: str, summary: str, steps: list[str], zone: str,
                     sign: Callable[[bytes], str], symptoms: str = "", origin: str = "", platform: str = "any",
                     evidence: dict[str, Any] | None = None) -> dict[str, Any]:
        """Record a fix this zone verified itself, signing the attestation."""
        fid = fix_id(problem, steps)
        ts = round(time.time(), 3)
        att = {"zone": zone, "ts": ts, "sig": sign(attestation_payload(fid, problem, steps, zone, ts))}
        base = {"problem": problem, "summary": summary, "steps": list(steps), "symptoms": symptoms,
                "platform": platform, "origin": origin or zone, "evidence": evidence or {}}
        if fid in self.fixes:  # keep the original metadata, just add our attestation
            base = {k: self.fixes[fid][k] for k in base}
        return self._upsert(base, [att])[0]

    def merge(self, fix: dict[str, Any]) -> tuple[dict[str, Any], int, int]:
        """Merge a fix received from a peer. Raises FixRejected if nothing in it can be trusted."""
        try:
            base = {"problem": str(fix["problem"]), "summary": str(fix.get("summary", ""))[:300],
                    "steps": [str(s) for s in fix["steps"]], "symptoms": str(fix.get("symptoms", ""))[:500],
                    "platform": str(fix.get("platform", "any")), "origin": str(fix.get("origin", "")),
                    "evidence": fix.get("evidence") if isinstance(fix.get("evidence"), dict) else {}}
        except (KeyError, TypeError) as exc:
            raise FixRejected(f"malformed fix: {exc}") from exc
        return self._upsert(base, list(fix.get("attestations") or []))

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
