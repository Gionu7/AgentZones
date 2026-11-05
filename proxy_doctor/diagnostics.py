"""Real proxy / TLS diagnostics using only the Python standard library.

Returns an *observation* dict (same schema as :mod:`proxy_doctor.synth`).
Nothing is sent anywhere except the test requests themselves; the
observation contains no credentials (proxy user:pass is redacted).
"""

from __future__ import annotations

import email.utils
import os
import platform
import re
import socket
import ssl
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

PUBLIC_CA_MARKERS = [
    "digicert", "let's encrypt", "isrg", "globalsign", "sectigo", "comodo", "usertrust", "amazon",
    "google trust", "gts ", "microsoft", "baltimore", "geotrust", "entrust", "godaddy", "starfield",
    "thawte", "verisign", "identrust", "certum", "buypass", "quovadis", "actalis", "harica", "ssl.com",
]
CA_ENV_VARS = ["REQUESTS_CA_BUNDLE", "SSL_CERT_FILE", "NODE_EXTRA_CA_CERTS", "CURL_CA_BUNDLE", "PIP_CERT"]


def _env(name: str) -> str | None:
    return os.environ.get(name) or os.environ.get(name.lower())


def redact(url: str | None) -> str | None:
    if not url:
        return url
    return re.sub(r"//[^/@]+@", "//***@", url)


def proxy_env() -> dict[str, Any]:
    return {
        "HTTPS_PROXY": bool(_env("HTTPS_PROXY")),
        "HTTP_PROXY": bool(_env("HTTP_PROXY")),
        "NO_PROXY": _env("NO_PROXY") or "",
        "proxy_url": redact(_env("HTTPS_PROXY") or _env("HTTP_PROXY")),
        "ca_env": {k: os.environ.get(k) for k in CA_ENV_VARS if os.environ.get(k)},
    }


def tcp_check(url: str | None, timeout: float = 3.0) -> bool | None:
    if not url:
        return None
    p = urllib.parse.urlparse(url if "://" in url else "http://" + url)
    try:
        with socket.create_connection((p.hostname, p.port or 80), timeout=timeout):
            return True
    except OSError:
        return False


def dns_check(host: str) -> bool:
    try:
        socket.getaddrinfo(host, 443)
        return True
    except OSError:
        return False


def _certifi_ctx() -> ssl.SSLContext | None:
    try:
        import certifi  # type: ignore
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return None


def https_get(url: str, ctx: ssl.SSLContext | None, timeout: float = 8.0) -> dict[str, Any]:
    handler = urllib.request.HTTPSHandler(context=ctx) if ctx else urllib.request.HTTPSHandler()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler(), handler)
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "proxy-doctor/0.1"})
    t0 = time.time()
    try:
        with opener.open(req, timeout=timeout) as r:
            return {"ok": True, "status": r.status, "error": None, "date": r.headers.get("Date"),
                    "ms": int((time.time() - t0) * 1000)}
    except urllib.error.HTTPError as e:
        # 4xx/5xx from the origin still proves connectivity; 407 is the proxy speaking
        ok = e.code not in (407, 502, 503, 504) and e.code < 500
        return {"ok": ok, "status": e.code, "error": None if ok else f"HTTP {e.code} {e.reason}",
                "date": e.headers.get("Date") if e.headers else None}
    except Exception as e:
        msg = str(getattr(e, "reason", e))
        m = re.search(r"Tunnel connection failed: (\d{3})", msg)
        return {"ok": False, "status": int(m.group(1)) if m else None, "error": f"{type(e).__name__}: {msg}",
                "date": None}


def _noverify_ctx() -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE  # diagnosis only: lets us see the presented chain
    return ctx


def peer_issuer(host: str, port: int = 443, timeout: float = 8.0) -> str | None:
    """Issuer of the certificate presented for host (through the proxy if set)."""
    proxy = _env("HTTPS_PROXY")
    try:
        if proxy and not _bypass(host):
            p = urllib.parse.urlparse(proxy)
            sock = socket.create_connection((p.hostname, p.port or 8080), timeout=timeout)
            sock.sendall(f"CONNECT {host}:{port} HTTP/1.1\r\nHost: {host}:{port}\r\n\r\n".encode())
            resp = b""
            while b"\r\n\r\n" not in resp:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                resp += chunk
            if b" 200" not in resp.split(b"\r\n", 1)[0]:
                sock.close()
                return None
        else:
            sock = socket.create_connection((host, port), timeout=timeout)
        with _noverify_ctx().wrap_socket(sock, server_hostname=host) as s:
            der = s.getpeercert(binary_form=True)
        if not der:
            return None
        with tempfile.NamedTemporaryFile("w", suffix=".pem", delete=False) as f:
            f.write(ssl.DER_cert_to_PEM_cert(der))
            path = f.name
        try:
            info = ssl._ssl._test_decode_cert(path)  # type: ignore[attr-defined]
        finally:
            os.unlink(path)
        return ", ".join(f"{k}={v}" for rdn in info.get("issuer", ()) for k, v in rdn)
    except Exception:
        return None


def intercepted(issuer: str | None) -> bool | None:
    """True if a certificate issuer is not a public CA, i.e. a proxy re-signs TLS. None if unknown."""
    return None if issuer is None else not any(m in issuer.lower() for m in PUBLIC_CA_MARKERS)


def _bypass(host: str) -> bool:
    try:
        return bool(urllib.request.proxy_bypass(host))
    except Exception:
        return False


def _skew(date_header: str | None) -> float | None:
    if not date_header:
        return None
    try:
        return time.time() - email.utils.parsedate_to_datetime(date_header).timestamp()
    except Exception:
        return None


def diagnose(target: str = "https://pypi.org", internal_url: str = "", error_text: str = "") -> dict[str, Any]:
    host = urllib.parse.urlparse(target).hostname or target
    env = proxy_env()
    sys_ctx = ssl.create_default_context()
    certifi_ctx = _certifi_ctx()
    # "external" = what a typical tool sees: certifi bundle unless a CA env var overrides it
    tool_ctx = certifi_ctx if (certifi_ctx and not env["ca_env"]) else sys_ctx
    external = https_get(target, tool_ctx)
    system = https_get(target, sys_ctx) if tool_ctx is not sys_ctx else external
    noverify = https_get(target, _noverify_ctx())
    issuer = peer_issuer(host)

    def verify_ok(res: dict) -> bool | None:
        if res["ok"]:
            return True
        err = (res.get("error") or "").lower()
        return False if "certificate" in err or "ssl" in err else None

    internal = {"checked": False, "ok": None, "status": None, "error": None}
    if internal_url:
        r = https_get(internal_url, sys_ctx) if internal_url.startswith("https") else https_get(internal_url, None)
        internal = {"checked": True, "ok": r["ok"], "status": r["status"], "error": r["error"]}

    return {
        "os": platform.system().lower(),
        "target": target,
        "env": {"HTTPS_PROXY": env["HTTPS_PROXY"], "HTTP_PROXY": env["HTTP_PROXY"], "NO_PROXY": env["NO_PROXY"]},
        "proxy_url": env["proxy_url"],
        "proxy_tcp_ok": tcp_check(_env("HTTPS_PROXY") or _env("HTTP_PROXY")),
        "external": {k: external[k] for k in ("ok", "status", "error")},
        "external_noverify": {k: noverify[k] for k in ("ok", "status", "error")},
        "internal": internal,
        "dns_external_ok": dns_check(host),
        "tls": {"issuer": issuer, "intercepted": intercepted(issuer)},
        "ca": {
            "system_has_issuer": verify_ok(system) if noverify["ok"] else None,
            "python_has_issuer": verify_ok(https_get(target, certifi_ctx)) if (certifi_ctx and noverify["ok"]) else None,
            "ca_env_set": bool(env["ca_env"]),
            "system_bundle_path": ssl.get_default_verify_paths().cafile or "/etc/ssl/certs/ca-certificates.crt",
        },
        "clock_skew_s": _skew(noverify.get("date")),
        "tool_errors": [error_text] if error_text else [],
        "simulated": False,
    }
