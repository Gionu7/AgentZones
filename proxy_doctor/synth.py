"""Synthetic proxy/TLS failure cases.

Produces *observations* in exactly the format :mod:`proxy_doctor.diagnostics`
returns, labelled with the true failure class. Used to

* train / distill the student classifier,
* give each demo zone its own, *skewed* local dataset (non-IID: a dev-laptop
  site mostly sees CA problems, a build farm mostly proxy auth, a Kubernetes
  site mostly NO_PROXY/DNS) - which is exactly when federated learning helps,
* drive the simulator in demos and tests.
"""

from __future__ import annotations

import random
from typing import Any

from .catalog import CLASSES

TOOLS = ["pip", "npm", "git", "curl", "python-requests", "java", "az", "docker", "conda", "dotnet"]
HOSTS = ["pypi.org", "registry.npmjs.org", "github.com", "huggingface.co", "login.microsoftonline.com",
         "management.azure.com", "repo.maven.apache.org", "ghcr.io"]
INTERNAL = ["git.corp.example", "artifactory.corp.example", "wiki.intra.example", "10.20.0.15", "jenkins.corp.example"]

ERRORS: dict[str, list[str]] = {
    "missing_proxy_env": [
        "Connection timed out after 30001 milliseconds",
        "ConnectTimeoutError: connect timeout to {host}:443",
        "Failed to connect to {host} port 443: Network is unreachable",
        "connect ETIMEDOUT 151.101.0.223:443",
        "dial tcp {host}:443: i/o timeout",
        "Max retries exceeded with url: /simple/ (Caused by ConnectTimeoutError)",
    ],
    "proxy_unreachable": [
        "ProxyError: Cannot connect to proxy. Connection refused",
        "Failed to connect to proxy.corp.example port 8080: Connection refused",
        "Could not resolve proxy: proxy.corp.example",
        "proxyconnect tcp: dial tcp 10.0.0.1:8080: connect: connection refused",
        "tunneling socket could not be established, cause=connect ECONNREFUSED",
    ],
    "proxy_auth_required": [
        "ProxyError: Tunnel connection failed: 407 Proxy Authentication Required",
        "Received HTTP code 407 from proxy after CONNECT",
        "Unable to tunnel through proxy. Proxy returns HTTP/1.1 407",
        "proxyconnect tcp: 407 Proxy Authentication Required",
        "statusCode=407 authentication required by proxy",
    ],
    "untrusted_corp_ca": [
        "SSL: CERTIFICATE_VERIFY_FAILED certificate verify failed: self-signed certificate in certificate chain",
        "SSL certificate problem: unable to get local issuer certificate",
        "PKIX path building failed: unable to find valid certification path to requested target",
        "x509: certificate signed by unknown authority",
        "SEC_E_UNTRUSTED_ROOT the certificate chain was issued by an authority that is not trusted",
        "SELF_SIGNED_CERT_IN_CHAIN self signed certificate in certificate chain",
    ],
    "no_proxy_internal": [
        "HTTP 502 Bad Gateway from proxy for {internal}",
        "403 Forbidden ERR_ACCESS_DENIED {internal}",
        "504 Gateway Timeout proxy could not reach {internal}",
        "fatal: unable to access 'https://{internal}/': The requested URL returned error: 502",
        "Received HTTP code 403 from proxy after CONNECT to {internal}",
    ],
    "dns_failure": [
        "getaddrinfo ENOTFOUND {host}",
        "Could not resolve host: {host}",
        "[Errno -2] Name or service not known",
        "nodename nor servname provided, or not known",
        "dial tcp: lookup {host}: no such host",
        "Temporary failure in name resolution",
    ],
    "clock_skew": [
        "SSL: CERTIFICATE_VERIFY_FAILED certificate verify failed: certificate has expired",
        "x509: certificate has expired or is not yet valid",
        "NET::ERR_CERT_DATE_INVALID",
        "certificate is not yet valid",
        "CertificateExpiredException: NotAfter",
    ],
}
ERRORS["tool_ca_bundle"] = ERRORS["untrusted_corp_ca"]

# class priors per site profile (index = CLASSES order)
PROFILES: dict[str, dict[str, float]] = {
    "uniform": {c: 1.0 for c in CLASSES},
    # developer laptops behind TLS interception
    "dev-laptops": {"healthy": 2, "untrusted_corp_ca": 5, "tool_ca_bundle": 5, "clock_skew": 2,
                    "missing_proxy_env": 0.15, "proxy_unreachable": 0.15, "proxy_auth_required": 0.1,
                    "no_proxy_internal": 0.1, "dns_failure": 0.1},
    # build servers / VMs with service accounts
    "build-servers": {"healthy": 2, "missing_proxy_env": 4, "proxy_auth_required": 5, "proxy_unreachable": 3,
                      "untrusted_corp_ca": 0.15, "tool_ca_bundle": 0.15, "no_proxy_internal": 0.1,
                      "dns_failure": 0.1, "clock_skew": 0.1},
    # Kubernetes / containers
    "k8s-cluster": {"healthy": 2, "no_proxy_internal": 5, "dns_failure": 4, "tool_ca_bundle": 2,
                    "missing_proxy_env": 0.15, "proxy_unreachable": 0.1, "proxy_auth_required": 0.1,
                    "untrusted_corp_ca": 0.1, "clock_skew": 0.1},
}


def _maybe(rng: random.Random, value: Any, p_unknown: float = 0.08) -> Any:
    return None if rng.random() < p_unknown else value


def make_case(label: str, rng: random.Random) -> dict[str, Any]:
    host, internal, tool = rng.choice(HOSTS), rng.choice(INTERNAL), rng.choice(TOOLS)
    proxy = rng.choice(["http://proxy.corp.example:8080", "http://10.0.0.1:3128", "http://webproxy:80"])
    no_proxy = rng.choice(["localhost,127.0.0.1", "localhost", "", "localhost,127.0.0.1,.svc"])
    obs: dict[str, Any] = {
        "os": rng.choice(["linux", "linux", "darwin", "windows"]),
        "target": f"https://{host}",
        "env": {"HTTPS_PROXY": True, "HTTP_PROXY": rng.random() < 0.9, "NO_PROXY": no_proxy},
        "proxy_url": proxy,
        "proxy_tcp_ok": True,
        "external": {"ok": True, "status": 200, "error": None},
        "external_noverify": {"ok": True, "status": 200, "error": None},
        "internal": {"checked": rng.random() < 0.6, "ok": True, "status": 200, "error": None},
        "dns_external_ok": rng.random() < 0.85,  # many corp networks do resolve external names
        "tls": {"issuer": "CN=DigiCert Global G2", "intercepted": False},
        "ca": {"system_has_issuer": True, "python_has_issuer": True, "ca_env_set": rng.random() < 0.3,
               "system_bundle_path": "/etc/ssl/certs/ca-certificates.crt"},
        "clock_skew_s": rng.gauss(0, 5),
        "tool_errors": [],
    }
    intercepted = rng.random() < 0.7
    if intercepted:
        obs["tls"] = {"issuer": rng.choice(["CN=Corp Root CA", "CN=Zscaler Root CA", "CN=BlueCoat ProxySG",
                                            "CN=Palo Alto Decryption CA"]), "intercepted": True}

    def fail(err: str, status: int | None = None, noverify_ok: bool = False):
        e = err.format(host=host, internal=internal)
        obs["external"] = {"ok": False, "status": status, "error": e}
        obs["external_noverify"] = ({"ok": True, "status": 200, "error": None} if noverify_ok
                                    else {"ok": False, "status": status, "error": e})
        obs["tool_errors"].append(f"{tool}: {e}")

    if label == "healthy":
        if rng.random() < 0.25:  # direct internet, no proxy
            obs["env"] = {"HTTPS_PROXY": False, "HTTP_PROXY": False, "NO_PROXY": ""}
            obs["proxy_url"], obs["proxy_tcp_ok"] = None, None
            obs["tls"]["intercepted"] = False
    elif label == "missing_proxy_env":
        obs["env"] = {"HTTPS_PROXY": False, "HTTP_PROXY": False, "NO_PROXY": ""}
        obs["proxy_url"], obs["proxy_tcp_ok"] = None, None
        obs["tls"] = {"issuer": None, "intercepted": None}
        obs["hint_proxy"] = proxy
        fail(rng.choice(ERRORS[label]))
    elif label == "proxy_unreachable":
        obs["proxy_tcp_ok"] = False
        obs["tls"] = {"issuer": None, "intercepted": None}
        fail(rng.choice(ERRORS[label]))
    elif label == "proxy_auth_required":
        obs["tls"] = {"issuer": None, "intercepted": None}
        fail(rng.choice(ERRORS[label]), status=407)
    elif label in ("untrusted_corp_ca", "tool_ca_bundle"):
        obs["tls"] = {"issuer": rng.choice(["CN=Corp Root CA", "CN=Zscaler Root CA", "CN=BlueCoat ProxySG"]),
                      "intercepted": True}
        obs["ca"]["system_has_issuer"] = label == "tool_ca_bundle"
        obs["ca"]["python_has_issuer"] = False
        obs["ca"]["ca_env_set"] = False
        fail(rng.choice(ERRORS[label]), noverify_ok=True)
    elif label == "no_proxy_internal":
        obs["env"]["NO_PROXY"] = rng.choice(["localhost,127.0.0.1", "", "localhost"])
        obs["internal"] = {"checked": True, "ok": False, "status": rng.choice([403, 502, 504]),
                           "error": rng.choice(ERRORS[label]).format(host=host, internal=internal)}
        obs["tool_errors"].append(f"{tool}: {obs['internal']['error']}")
        obs["hint_internal_suffix"] = "." + internal.split(".", 1)[-1] if "." in internal else internal
    elif label == "dns_failure":
        obs["dns_external_ok"] = False
        if rng.random() < 0.5:
            obs["env"] = {"HTTPS_PROXY": False, "HTTP_PROXY": False, "NO_PROXY": ""}
            obs["proxy_url"], obs["proxy_tcp_ok"] = None, None
        else:
            obs["proxy_url"] = "socks5://proxy.corp.example:1080"
        obs["tls"] = {"issuer": None, "intercepted": None}
        fail(rng.choice(ERRORS[label]))
    elif label == "clock_skew":
        obs["clock_skew_s"] = rng.choice([-1, 1]) * rng.uniform(3600, 86400 * 400)
        fail(rng.choice(ERRORS[label]), noverify_ok=True)
    else:
        raise ValueError(label)

    # observation noise: unknown fields, irrelevant log lines
    obs["proxy_tcp_ok"] = _maybe(rng, obs["proxy_tcp_ok"])
    obs["dns_external_ok"] = _maybe(rng, obs["dns_external_ok"])
    obs["ca"]["system_has_issuer"] = _maybe(rng, obs["ca"]["system_has_issuer"], 0.15)
    obs["ca"]["python_has_issuer"] = _maybe(rng, obs["ca"]["python_has_issuer"], 0.15)
    obs["clock_skew_s"] = _maybe(rng, obs["clock_skew_s"], 0.2)
    if rng.random() < 0.3:
        obs["tool_errors"].append(rng.choice(["WARNING: Retrying (Retry(total=4))", "npm notice",
                                              "Collecting requests", "remote: Enumerating objects"]))
    return obs


def sample_labels(n: int, profile: str, rng: random.Random) -> list[str]:
    pri = PROFILES[profile]
    weights = [pri.get(c, 0.0) for c in CLASSES]
    return rng.choices(CLASSES, weights=weights, k=n)


def make_dataset(n: int, profile: str = "uniform", seed: int = 0, label_noise: float = 0.02):
    rng = random.Random(seed)
    labels = sample_labels(n, profile, rng)
    cases = [make_case(lbl, rng) for lbl in labels]
    noisy = [rng.choice(CLASSES) if rng.random() < label_noise else lbl for lbl in labels]
    return cases, noisy


if __name__ == "__main__":  # quick look: python -m proxy_doctor.synth
    import json
    import sys
    c, y = make_dataset(int(sys.argv[1]) if len(sys.argv) > 1 else 3, seed=1)
    for case, label in zip(c, y):
        print(label, json.dumps(case, indent=1))
