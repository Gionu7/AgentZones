"""Observation -> fixed-size feature vector for the student classifier."""

from __future__ import annotations

import re
import zlib
from typing import Any

import numpy as np

HASH_DIM = 256
_TOK = re.compile(r"[a-z_]{3,}|\d{3}")

STRUCT_NAMES = [
    "bias",
    "https_proxy_set", "no_proxy_has_domain",
    "proxy_tcp_ok", "proxy_tcp_fail",
    "external_ok", "external_fail", "status_407", "status_403", "status_5xx",
    "noverify_ok", "noverify_fail",
    "internal_checked", "internal_ok", "internal_fail",
    "dns_ok", "dns_fail",
    "tls_intercepted", "tls_not_intercepted",
    "sys_has_issuer", "sys_lacks_issuer",
    "py_has_issuer", "py_lacks_issuer", "ca_env_set",
    "skew_large", "skew_small",
    "proxy_socks",
]
DIM = len(STRUCT_NAMES) + HASH_DIM


def _tri(v: Any) -> tuple[float, float]:
    if v is None:
        return 0.0, 0.0
    return (1.0, 0.0) if v else (0.0, 1.0)


def text_of(obs: dict[str, Any]) -> str:
    parts = list(obs.get("tool_errors") or [])
    for k in ("external", "external_noverify", "internal"):
        e = (obs.get(k) or {}).get("error")
        if e:
            parts.append(e)
    if obs.get("question"):
        parts.append(obs["question"])
    return " ".join(parts).lower()


def featurize(obs: dict[str, Any]) -> np.ndarray:
    env = obs.get("env") or {}
    ext = obs.get("external") or {}
    nov = obs.get("external_noverify") or {}
    internal = obs.get("internal") or {}
    tls = obs.get("tls") or {}
    ca = obs.get("ca") or {}
    no_proxy = env.get("NO_PROXY") or ""
    skew = obs.get("clock_skew_s")
    status = ext.get("status") or 0
    ext_ok = ext.get("ok") if ext else None
    nov_ok = nov.get("ok") if nov else None
    int_ok = internal.get("ok") if internal.get("checked") else None
    v = [
        1.0,
        float(bool(env.get("HTTPS_PROXY"))),
        float(any(p.strip().startswith(".") and p.strip() not in (".svc",) for p in no_proxy.split(","))
              or "corp" in no_proxy or "intra" in no_proxy),
        *_tri(obs.get("proxy_tcp_ok")),
        *_tri(ext_ok),
        float(status == 407), float(status == 403), float(500 <= status < 600),
        *_tri(nov_ok),
        float(bool(internal.get("checked"))), *_tri(int_ok),
        *_tri(obs.get("dns_external_ok")),
        *_tri(tls.get("intercepted")),
        *_tri(ca.get("system_has_issuer")),
        *_tri(ca.get("python_has_issuer")),
        float(bool(ca.get("ca_env_set"))),
        *(_tri(abs(skew) > 600) if skew is not None else (0.0, 0.0)),
        float(str(obs.get("proxy_url") or "").startswith("socks")),
    ]
    h = np.zeros(HASH_DIM)
    for tok in _TOK.findall(text_of(obs)):
        h[zlib.crc32(tok.encode()) % HASH_DIM] += 1.0
    if h.any():
        h = h / np.linalg.norm(h)
    return np.concatenate([np.array(v, dtype=np.float64), h])


def featurize_many(cases: list[dict[str, Any]]) -> np.ndarray:
    return np.stack([featurize(c) for c in cases]) if cases else np.zeros((0, DIM))
