"""Teacher labelling: ask a large model (OpenAI-compatible endpoint) to diagnose observations.

The teacher can be a big open-weight model on the GPU machine (vLLM / Ollama:
qwen2.5:32b, llama3.3:70b, ...) or Azure OpenAI. Its output is turned into a
probability distribution over classes (soft labels) plus an explanation.
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from ..catalog import CLASSES, DESCRIPTIONS

SYSTEM = (
    "You are a senior network engineer diagnosing corporate proxy and TLS-interception problems.\n"
    "Classes:\n" + "\n".join(f"- {c}: {d}" for c, d in DESCRIPTIONS.items()) +
    "\nGiven a diagnostic observation (JSON), answer ONLY with JSON: "
    '{"problem": <class>, "confidence": <0..1>, "alternatives": {<class>: <prob>, ...}, '
    '"explanation": <2-3 sentences citing the evidence>}'
)


def brief(obs: dict[str, Any]) -> dict[str, Any]:
    keep = ("env", "proxy_url", "proxy_tcp_ok", "external", "external_noverify", "internal", "dns_external_ok",
            "tls", "ca", "clock_skew_s", "tool_errors")
    return {k: obs.get(k) for k in keep}


def to_probs(ans: dict[str, Any]) -> list[float]:
    p = [0.0] * len(CLASSES)
    top = ans.get("problem")
    conf = float(ans.get("confidence", 0.8))
    if top in CLASSES:
        p[CLASSES.index(top)] = conf
    for c, v in (ans.get("alternatives") or {}).items():
        if c in CLASSES and c != top:
            p[CLASSES.index(c)] += max(float(v), 0.0)
    s = sum(p)
    if s <= 0:
        return [1.0 / len(CLASSES)] * len(CLASSES)
    return [x / s for x in p]


class Teacher:
    def __init__(self, base_url: str, model: str, api_key: str = "local", timeout: float = 180):
        self.base_url, self.model, self.api_key = base_url.rstrip("/"), model, api_key
        self.http = httpx.Client(timeout=timeout)

    def label(self, obs: dict[str, Any]) -> dict[str, Any]:
        body = {"model": self.model, "temperature": 0.0, "response_format": {"type": "json_object"},
                "messages": [{"role": "system", "content": SYSTEM},
                             {"role": "user", "content": json.dumps(brief(obs), default=str)}]}
        r = self.http.post(f"{self.base_url}/chat/completions", json=body,
                           headers={"Authorization": f"Bearer {self.api_key}"})
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"]
        start, end = text.find("{"), text.rfind("}")
        ans = json.loads(text[start:end + 1])
        return {"probs": to_probs(ans), "problem": ans.get("problem"), "explanation": ans.get("explanation", "")}


def oracle_label(label: str, smoothing: float = 0.1) -> dict[str, Any]:
    """Offline stand-in for the teacher: ground truth with label smoothing."""
    p = [smoothing / (len(CLASSES) - 1)] * len(CLASSES)
    p[CLASSES.index(label)] = 1.0 - smoothing
    return {"probs": p, "problem": label, "explanation": DESCRIPTIONS[label]}
