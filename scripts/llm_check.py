#!/usr/bin/env python3
"""Measure whether a local model is good enough for an Agent Zone.

    python scripts/llm_check.py --url http://localhost:11434/v1 --model qwen2.5:1.5b qwen2.5:3b qwen2.5:7b

For each model it runs two kinds of task, N times each:

  tool   the model must call the right tool with valid arguments (as LLM agents do)
  json   the model must act as Proxy Doctor's investigator: read a simulated observation and
         return valid JSON naming the right failure class

and reports success rates and latency, so you can pick the smallest model that is reliable
(rule of thumb: >= 90% tool, >= 80% json). No zone needs to be running.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from proxy_doctor.agent import INVESTIGATOR_PROMPT, parse_json_object, sanitize  # noqa: E402
from proxy_doctor.catalog import CLASSES, DESCRIPTIONS  # noqa: E402
from proxy_doctor.synth import make_case  # noqa: E402

TOOLS = [
    {"type": "function", "function": {"name": "proxy__check_https", "description": "HEAD request to a URL through the proxy",
                                      "parameters": {"type": "object", "required": ["url"], "properties": {
                                          "url": {"type": "string"}, "verify": {"type": "boolean"}}}}},
    {"type": "function", "function": {"name": "proxy__dns_lookup", "description": "Check whether a host resolves",
                                      "parameters": {"type": "object", "required": ["host"], "properties": {
                                          "host": {"type": "string"}}}}},
    {"type": "function", "function": {"name": "proxy__inspect_tls", "description": "Issuer of the TLS certificate of a host",
                                      "parameters": {"type": "object", "required": ["host"], "properties": {
                                          "host": {"type": "string"}}}}},
]
TOOL_TASKS = [
    ("Does github.com resolve from here?", "proxy__dns_lookup", "host", "github.com"),
    ("Who issued the TLS certificate that pypi.org presents to us?", "proxy__inspect_tls", "host", "pypi.org"),
    ("Can we reach https://huggingface.co through the proxy?", "proxy__check_https", "url", "huggingface.co"),
]


def chat(http, url, model, messages, tools=None, json_mode=False):
    body = {"model": model, "messages": messages, "temperature": 0}
    if tools:
        body["tools"] = tools
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    t0 = time.time()
    r = http.post(f"{url.rstrip('/')}/chat/completions", json=body, headers={"Authorization": "Bearer local"})
    r.raise_for_status()
    return r.json()["choices"][0]["message"], time.time() - t0


def run_model(http, url, model, n, rng):
    tool_ok, json_ok, lat = 0, 0, []
    for i in range(n):
        q, name, arg, val = TOOL_TASKS[i % len(TOOL_TASKS)]
        try:
            msg, dt = chat(http, url, model, [{"role": "system", "content": "Use a tool to answer."},
                                              {"role": "user", "content": q}], TOOLS)
            lat.append(dt)
            calls = msg.get("tool_calls") or []
            if calls:
                fn = calls[0]["function"]
                args = fn["arguments"] if isinstance(fn["arguments"], dict) else json.loads(fn["arguments"] or "{}")
                tool_ok += fn["name"] == name and val in str(args.get(arg, ""))
        except Exception as exc:
            print(f"    tool task error: {exc}", file=sys.stderr)
    for _ in range(n):
        truth = rng.choice([c for c in CLASSES if c != "healthy"])
        obs = sanitize(make_case(truth, rng))
        system = INVESTIGATOR_PROMPT.format(tools="", classes="\n".join(f"- {c}: {d}" for c, d in DESCRIPTIONS.items()))
        try:
            msg, dt = chat(http, url, model, [{"role": "system", "content": system},
                                              {"role": "user", "content": f"Observation: {json.dumps(obs)}"}],
                           json_mode=True)
            lat.append(dt)
            ans = parse_json_object(msg.get("content"))
            json_ok += bool(ans) and ans.get("problem") == truth
        except Exception as exc:
            print(f"    json task error: {exc}", file=sys.stderr)
    return tool_ok / n, json_ok / n, statistics.median(lat) if lat else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:11434/v1")
    ap.add_argument("--model", nargs="+", default=["qwen2.5:1.5b"])
    ap.add_argument("-n", type=int, default=12)
    args = ap.parse_args()
    rng = random.Random(5)
    print(f"{'model':<28} {'tool calls':>10} {'diagnosis json':>15} {'median s':>9}")
    with httpx.Client(timeout=300) as http:
        for m in args.model:
            t, j, l = run_model(http, args.url, m, args.n, rng)
            flag = "ok" if t >= 0.9 and j >= 0.8 else "weak"
            print(f"{m:<28} {t:>10.0%} {j:>15.0%} {l:>9.1f}  {flag}")


if __name__ == "__main__":
    main()
