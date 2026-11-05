#!/usr/bin/env python3
"""Regenerate the charts in docs/images from real runs of the federated-learning plugins.

    python scripts/make_doc_charts.py                      # Proxy Doctor charts (offline, seconds)
    MNIST_PATH=/path/to/mnist.npz python scripts/make_doc_charts.py   # also the MNIST chart

The clients and the strategy are the same plugin classes the zones run (tasks/*.py); here they are called
in one process, round by round, exactly as the aggregator does (fit on each client, then FedAvg), without
the network in between. Prints the numbers as markdown tables for the docs.

Charts are SVG with light and dark colours (they follow the reader's theme on GitHub). Colours are the
validated reference palette: series 1 blue, series 2 orange; gray marks are context.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "docs" / "images"

from agentzones.fl import FitResult  # noqa: E402

STYLE = """<style>
  .bg { fill: #fcfcfb; } .t1 { fill: #0b0b0b; } .t2 { fill: #52514e; } .grid { stroke: #e1e0d9; } .axis { stroke: #c3c2b7; }
  .s1 { stroke: #2a78d6; } .f1 { fill: #2a78d6; } .s2 { stroke: #eb6834; } .f2 { fill: #eb6834; } .fg { fill: #c3c2b7; }
  .ring { stroke: #fcfcfb; }
  @media (prefers-color-scheme: dark) {
    .bg { fill: #1a1a19; } .t1 { fill: #f0efec; } .t2 { fill: #c3c2b7; } .grid { stroke: #2c2c2a; } .axis { stroke: #383835; }
    .s1 { stroke: #3987e5; } .f1 { fill: #3987e5; } .s2 { stroke: #d95926; } .f2 { fill: #d95926; } .fg { fill: #5c5b56; }
    .ring { stroke: #1a1a19; }
  }
  text { font-family: system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif; }
</style>"""


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def svg(w: int, h: int, title: str, sub: str, body: str) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" '
            f'aria-label="{esc(title)}">{STYLE}<rect class="bg" width="{w}" height="{h}" rx="10"/>'
            f'<text class="t1" x="24" y="32" font-size="16" font-weight="650">{esc(title)}</text>'
            f'<text class="t2" x="24" y="52" font-size="12.5">{esc(sub)}</text>{body}</svg>\n')


def line_chart(series: list[dict], title: str, sub: str, y_lo: float, y_hi: float, step: float) -> str:
    """series: [{name, cls ('1'|'2'), points: [(x, y)]}]; y as a fraction, shown in %."""
    W, H, L, R, T, B = 720, 330, 60, 150, 76, 46
    xs = [x for s in series for x, _ in s["points"]]
    x0, x1 = min(xs), max(xs)
    X = lambda v: L + (W - L - R) * (v - x0) / max(x1 - x0, 1)  # noqa: E731
    Y = lambda v: T + (H - T - B) * (1 - (v - y_lo) / (y_hi - y_lo))  # noqa: E731
    g = ""
    v = y_lo
    while v <= y_hi + 1e-9:
        g += (f'<line class="grid" x1="{L}" x2="{W - R}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke-width="1"/>'
              f'<text class="t2" x="{L - 8}" y="{Y(v) + 4:.1f}" font-size="11.5" text-anchor="end">{v * 100:.0f}%</text>')
        v += step
    g += f'<line class="axis" x1="{L}" x2="{W - R}" y1="{H - B}" y2="{H - B}" stroke-width="1"/>'
    for x in range(int(x0), int(x1) + 1):
        g += f'<text class="t2" x="{X(x):.1f}" y="{H - B + 18}" font-size="11.5" text-anchor="middle">{x}</text>'
    g += f'<text class="t2" x="{(L + W - R) / 2:.0f}" y="{H - 8}" font-size="11.5" text-anchor="middle">round</text>'
    for s in series:
        pts = " ".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in s["points"])
        g += f'<polyline class="s{s["cls"]}" points="{pts}" fill="none" stroke-width="2" stroke-linejoin="round"/>'
        for x, y in s["points"]:  # 8px markers with a 2px surface ring
            g += (f'<circle class="f{s["cls"]} ring" cx="{X(x):.1f}" cy="{Y(y):.1f}" r="4" stroke-width="2">'
                  f'<title>{esc(s["name"])}, round {x}: {y * 100:.1f}%</title></circle>')
        lx, ly = s["points"][-1]  # direct label at the end of the line: name, then the value
        dy = s.get("label_dy", 0)
        g += (f'<text class="t1" x="{X(lx) + 10:.1f}" y="{Y(ly) + 4 + dy:.1f}" font-size="12.5" font-weight="600">'
              f'{ly * 100:.1f}%</text><text class="t2" x="{X(lx) + 10:.1f}" y="{Y(ly) + 19 + dy:.1f}" font-size="11.5">'
              f'{esc(s["name"])}</text>')
    if len(series) > 1:  # legend for two or more series
        lx = L
        for s in series:
            g += (f'<circle class="f{s["cls"]}" cx="{lx + 5}" cy="66" r="4"/>'
                  f'<text class="t2" x="{lx + 14}" y="70" font-size="11.5">{esc(s["name"])}</text>')
            lx += 14 + 7 * len(s["name"]) + 24
    return svg(W, H, title, sub, g)


def bar_chart(rows: list[tuple[str, float, bool]], title: str, sub: str) -> str:
    """Horizontal bars, value as a fraction. rows: (label, value, highlight)."""
    W, L, R, T, bh, gap = 720, 230, 70, 74, 26, 14
    H = T + len(rows) * (bh + gap) + 30
    X = lambda v: L + (W - L - R) * v  # noqa: E731
    g = ""
    for k in range(0, 11, 2):
        v = k / 10
        g += (f'<line class="grid" x1="{X(v):.1f}" x2="{X(v):.1f}" y1="{T - 6}" y2="{H - 30}" stroke-width="1"/>'
              f'<text class="t2" x="{X(v):.1f}" y="{H - 12}" font-size="11.5" text-anchor="middle">{k * 10}%</text>')
    for i, (label, v, hl) in enumerate(rows):
        y = T + i * (bh + gap)
        w = X(v) - L
        # square at the baseline, 4px rounded at the data end
        path = (f"M{L},{y} h{w - 4:.1f} a4,4 0 0 1 4,4 v{bh - 8} a4,4 0 0 1 -4,4 h{-(w - 4):.1f} z")
        g += (f'<path class="{"f1" if hl else "fg"}" d="{path}"><title>{esc(label)}: {v * 100:.1f}%</title></path>'
              f'<text class="t1" x="{L - 10}" y="{y + bh / 2 + 4.5:.1f}" font-size="12.5" text-anchor="end"'
              f'{" font-weight=" + chr(34) + "650" + chr(34) if hl else ""}>{esc(label)}</text>'
              f'<text class="t1" x="{X(v) + 8:.1f}" y="{y + bh / 2 + 4.5:.1f}" font-size="12.5" font-weight="600">'
              f'{v * 100:.0f}%</text>')
    g += f'<line class="axis" x1="{L}" x2="{L}" y1="{T - 6}" y2="{H - 30}" stroke-width="1"/>'
    return svg(W, H, title, sub, g)


def run_rounds(strategy, clients: dict, rounds: int, eval_key: str) -> list[float]:
    """The aggregator's loop without the network: fl.fit on every client, then FedAvg, then evaluate."""
    params, accs = strategy.initial_parameters(), []
    for rnd in range(1, rounds + 1):
        cfg = strategy.configure_fit(rnd, params, list(clients))
        results = [FitResult(z, "client", *c.fit(params, cfg)) for z, c in clients.items()]
        params, _ = strategy.aggregate_fit(rnd, results, [], params)
        accs.append(strategy.evaluate(rnd, params)[eval_key])
    return accs


def proxy_doctor(rounds: int = 5) -> dict:
    from tasks.proxy_student import ProxyStudentClient, ProxyStudentStrategy
    sites = {"az": "dev-laptops", "b": "build-servers", "cn": "k8s-cluster"}
    make = lambda z: ProxyStudentClient({"ZONE": z, "PD_PROFILE": sites[z]})  # noqa: E731
    strategy = ProxyStudentStrategy({})
    alone = {}
    for z in sites:  # the same amount of training, on the site's own data only
        c, p = make(z), strategy.initial_parameters()
        for rnd in range(1, rounds + 1):
            p, _, _ = c.fit(p, {"round": rnd})
        alone[z] = strategy.evaluate(0, p)["global_acc"]
    fed = run_rounds(strategy, {z: make(z) for z in sites}, rounds, "global_acc")
    return {"sites": sites, "alone": alone, "federated": fed}


def mnist(rounds: int = 5) -> dict | None:
    path = os.environ.get("MNIST_PATH", "")
    if not path or not Path(path).exists():
        return None
    from tasks.mnist import MnistClient, MnistStrategy
    base = {"MNIST_PATH": path, "FL_NUM_CLIENTS": "3"}
    strategy = MnistStrategy(base)
    split = {"az": "0,1,2,3", "b": "4,5,6", "cn": "7,8,9"}
    iid = {z: MnistClient({**base, "FL_CLIENT_INDEX": str(i)}) for i, z in enumerate(split)}
    non = {z: MnistClient({**base, "FL_CLIENT_INDEX": str(i), "LABELS": split[z]}) for i, z in enumerate(split)}
    return {"iid": run_rounds(strategy, iid, rounds, "test_acc"), "noniid": run_rounds(strategy, non, rounds, "test_acc"),
            "split": split}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pd = proxy_doctor()
    fed = pd["federated"]
    rows = [(f"{z} alone ({pd['sites'][z]})", pd["alone"][z], False) for z in pd["sites"]]
    rows.append(("federated: az + b + cn", fed[-1], True))
    (OUT / "fl-why-federate.svg").write_text(bar_chart(
        rows, "Proxy Doctor's classifier on problems from all sites",
        f"Accuracy on the shared benchmark after {len(fed)} rounds of training. No site shared its incident data."))
    (OUT / "fl-rounds.svg").write_text(line_chart(
        [{"name": "global model", "cls": "1", "points": list(enumerate(fed, 1))}],
        "Federated rounds: Proxy Doctor's classifier", "Accuracy of the averaged model on the shared benchmark, per round",
        0.0, 1.0, 0.25))
    print("| | benchmark accuracy |\n|---|---|")
    for label, v, _ in rows:
        print(f"| {label} | {v * 100:.1f}% |")
    print("\n| round | " + " | ".join(str(i) for i in range(1, len(fed) + 1)) + " |\n|---|" + "---|" * len(fed))
    print("| global model | " + " | ".join(f"{v * 100:.1f}%" for v in fed) + " |")

    mn = mnist()
    if mn:
        lo = max(0.0, np.floor(min(mn["iid"] + mn["noniid"]) * 10) / 10 - 0.1)
        (OUT / "fl-mnist.svg").write_text(line_chart(
            [{"name": "evenly split data", "cls": "1", "points": list(enumerate(mn["iid"], 1)), "label_dy": -12},
             {"name": "each zone: different digits", "cls": "2", "points": list(enumerate(mn["noniid"], 1)), "label_dy": 12}],
            "MNIST across three zones", "Test accuracy of the global model per round (10,000 test images)",
            lo, 1.0, 0.1))
        print("\n| round | " + " | ".join(str(i) for i in range(1, len(mn["iid"]) + 1)) + " |\n|---|" + "---|" * len(mn["iid"]))
        print("| evenly split | " + " | ".join(f"{v * 100:.1f}%" for v in mn["iid"]) + " |")
        print("| " + ", ".join(f"{z}: {d}" for z, d in mn["split"].items()) + " | "
              + " | ".join(f"{v * 100:.1f}%" for v in mn["noniid"]) + " |")
    else:
        print("\n(MNIST chart skipped: set MNIST_PATH to an mnist.npz)")
    print(f"\nwrote {', '.join(p.name for p in sorted(OUT.glob('fl-*.svg')))}")


if __name__ == "__main__":
    main()
