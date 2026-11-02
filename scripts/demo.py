#!/usr/bin/env python3
"""End-to-end hackathon demo on one machine: three zones, no central registry.

    python scripts/demo.py            # starts zone-a/b/c locally and runs the story
    python scripts/demo.py --no-launch --zones http://h1:7700 http://h2:7700 http://h3:7700

Story
  1. Discovery   zone-c only knows zone-b, zone-b only knows zone-a; gossip connects all.
  2. Local fix   zone-b (build servers) hits "407 proxy auth", fixes it, verifies, shares the fix.
  3. Ask peers   zone-a (dev laptops) hits the same problem, which it has rarely seen; it is
                 unsure, asks its peers, applies zone-b's verified fix and confirms it.
  4. Federated   four FedAvg rounds: every zone's model improves on problems it never saw
                 locally, without exchanging any raw data.
  5. After FL    zone-a now diagnoses a Kubernetes-style NO_PROXY problem on its own.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
B, DIM, G, R, Y, X = "\033[1m", "\033[2m", "\033[32m", "\033[31m", "\033[33m", "\033[0m"


def h(title: str) -> None:
    print(f"\n{B}=== {title} ==={X}")


def get(url, path):
    return httpx.get(url + path, timeout=30).json()


def post(url, path, body=None, timeout=300):
    return httpx.post(url + path, json=body or {}, timeout=timeout).json()


def ask(url, scenario, question):
    r = post(url, "/api/agents/proxy-doctor/ask", {"question": question, "context": {"simulate": scenario}})
    d = r["diagnosis"]
    loc = d["local"]
    print(f"  local model : {loc['problem']} ({loc['confidence']:.0%})")
    for p in d.get("peers", []):
        print(f"  peer {p['zone']:<7}: {p['problem']} ({p['confidence']:.0%}), {p['verified_fixes']} verified fix(es)")
    ok = G if d["problem"] == scenario else R
    print(f"  decision    : {ok}{d['problem']}{X} ({d['confidence']:.0%})  [truth: {scenario}]")
    if r.get("fix"):
        print(f"  fix         : {r['fix']['summary']}  {DIM}({r['fix']['source']}){X}")
        print(f"  verified    : {G + 'yes' if r.get('verified') else R + 'no'}{X}"
              + (f", shared with {', '.join(r.get('shared_with', []))}" if r.get("shared_with") else ""))
    return r


def accs(urls):
    out = {}
    for u in urls:
        m = get(u, "/api/fl")["metrics"]
        out[get(u, "/api/info")["name"]] = m
    return out


def show_acc(urls, label):
    a = accs(urls)
    cells = "  ".join(f"{n}: local {m['local_acc']:.0%} / benchmark {B}{m['global_acc']:.0%}{X}" for n, m in a.items())
    print(f"  {label:<14} {cells}")


def wait_health(urls, timeout=60):
    t0 = time.time()
    for u in urls:
        while True:
            try:
                if httpx.get(u + "/health", timeout=2).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            if time.time() - t0 > timeout:
                sys.exit(f"zone {u} did not start")
            time.sleep(0.5)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-launch", action="store_true")
    ap.add_argument("--zones", nargs=3, default=["http://127.0.0.1:7701", "http://127.0.0.1:7702",
                                                 "http://127.0.0.1:7703"])
    ap.add_argument("--keep", action="store_true", help="leave zones running after the demo")
    ap.add_argument("--fl-rounds", type=int, default=4)
    args = ap.parse_args()
    a, b, c = args.zones
    procs = []
    if not args.no_launch:
        data = ROOT / ".zone-data"
        if data.exists():
            import shutil
            shutil.rmtree(data)
        env = {**os.environ, "PYTHONPATH": str(ROOT), "LOG_LEVEL": os.environ.get("LOG_LEVEL", "WARNING")}
        env.setdefault("MODEL_PROVIDER", "none")  # set MODEL_PROVIDER=openai to use Ollama for explanations
        for z in "abc":
            log = open(ROOT / f".zone-{z}.log", "w")
            procs.append(subprocess.Popen([sys.executable, "-m", "agentzones.cli", "up", "-c",
                                           f"examples/demo/zone-{z}.yaml"], cwd=ROOT, env=env,
                                          stdout=log, stderr=subprocess.STDOUT))
        print(f"{DIM}started 3 zones, logs in .zone-[abc].log{X}")
    try:
        wait_health(args.zones)

        h("1. Discovery by gossip (no registry)")
        print(f"  seeds: zone-c -> zone-b -> zone-a")
        t0 = time.time()
        while time.time() - t0 < 40:
            views = {get(u, "/api/info")["name"]: sorted(p["name"] for p in get(u, "/api/peers")) for u in args.zones}
            if all(len(v) == 2 for v in views.values()):
                break
            time.sleep(1)
        for n, v in views.items():
            print(f"  {n} knows {v}")
        print(f"  {DIM}fully connected after {time.time() - t0:.1f}s{X}")
        show_acc(args.zones, "local-only")

        h("2. zone-b (build servers): 407 from proxy")
        ask(b, "proxy_auth_required", "az login fails: Tunnel connection failed: 407 Proxy Authentication Required")

        h("3. zone-a (dev laptops): same problem, rarely seen there -> asks peers")
        ask(a, "proxy_auth_required", "pip install fails with 407 Proxy Authentication Required")
        fx = get(c, "/api/fixes")
        print(f"  zone-c now holds {len(fx)} fix(es): " +
              "; ".join(f"{f['summary']} [verified by {', '.join(f['verified_by'])}]" for f in fx))

        h(f"4. Federated learning: {args.fl_rounds} FedAvg rounds, no raw data leaves a zone")
        show_acc(args.zones, "before")
        for r in range(1, args.fl_rounds + 1):
            for u in args.zones:  # phase 1: everyone trains locally
                post(u, "/api/fl/train")
            for u in args.zones:  # phase 2: everyone averages the peers' trained snapshots
                post(u, "/api/fl/aggregate")
            show_acc(args.zones, f"after round {r}")

        h("5. zone-a after FL: a Kubernetes-style problem it never saw locally")
        ask(a, "no_proxy_internal", "git clone https://git.corp.example fails: 502 Bad Gateway from proxy")

        print(f"\n{G}Demo complete.{X} Dashboards: " + "  ".join(args.zones))
        if args.keep and procs:
            print("zones keep running - Ctrl+C to stop")
            while True:
                time.sleep(3600)
    except KeyboardInterrupt:
        pass
    finally:
        for p in procs:
            p.terminate()


if __name__ == "__main__":
    main()
