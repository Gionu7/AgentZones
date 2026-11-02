"""Failure classes and fix plans."""

from __future__ import annotations

CLASSES = [
    "healthy",
    "missing_proxy_env",
    "proxy_unreachable",
    "proxy_auth_required",
    "untrusted_corp_ca",
    "tool_ca_bundle",
    "no_proxy_internal",
    "dns_failure",
    "clock_skew",
]

DESCRIPTIONS = {
    "healthy": "Outbound HTTPS works; no proxy or certificate problem detected.",
    "missing_proxy_env": "Direct internet access is blocked and no proxy is configured (HTTPS_PROXY unset).",
    "proxy_unreachable": "A proxy is configured but cannot be reached (wrong host/port, VPN off, proxy down).",
    "proxy_auth_required": "The proxy answers 407 Proxy Authentication Required - credentials missing or wrong.",
    "untrusted_corp_ca": "TLS is intercepted by the corporate proxy and its root CA is not in the system trust store.",
    "tool_ca_bundle": "The system trusts the corporate CA but the tool uses its own CA bundle "
                      "(Python certifi, Node, Java, conda, git).",
    "no_proxy_internal": "Internal hosts are being sent through the proxy - NO_PROXY is missing internal domains.",
    "dns_failure": "External names cannot be resolved locally; the client must let the proxy resolve them.",
    "clock_skew": "The system clock is wrong, so valid certificates look expired or not yet valid.",
}

# Fix plans: ordered steps (shell for linux/macOS; Windows notes where relevant).
# {proxy}, {ca_path}, {internal} are filled in by the agent from the observation.
FIXES: dict[str, dict] = {
    "missing_proxy_env": {
        "summary": "Configure the corporate proxy in the environment",
        "steps": [
            "export HTTPS_PROXY={proxy} HTTP_PROXY={proxy}",
            "export NO_PROXY=localhost,127.0.0.1,.svc,.cluster.local,{internal}",
            "persist in ~/.profile (or System > Environment Variables on Windows)",
        ],
    },
    "proxy_unreachable": {
        "summary": "Point to a reachable proxy (check VPN / PAC file)",
        "steps": [
            "connect the corporate VPN if off-site",
            "read the proxy from the PAC file: curl -s $PAC_URL | grep -i proxy",
            "export HTTPS_PROXY={proxy} HTTP_PROXY={proxy}",
        ],
    },
    "proxy_auth_required": {
        "summary": "Authenticate to the proxy",
        "steps": [
            "use a local auth helper (cntlm / px) on 127.0.0.1:3128 for NTLM/Kerberos proxies",
            "export HTTPS_PROXY=http://127.0.0.1:3128 HTTP_PROXY=http://127.0.0.1:3128",
            "or (basic auth only) export HTTPS_PROXY=http://$USER:$PASS@{proxy_host} - avoid storing passwords",
        ],
    },
    "untrusted_corp_ca": {
        "summary": "Install the corporate root CA into the system trust store",
        "steps": [
            "obtain the corporate root CA (IT portal) as corp-root.crt",
            "Debian/Ubuntu: sudo cp corp-root.crt /usr/local/share/ca-certificates/ && sudo update-ca-certificates",
            "RHEL: sudo cp corp-root.crt /etc/pki/ca-trust/source/anchors/ && sudo update-ca-trust",
            "macOS: sudo security add-trusted-cert -d -r trustRoot -k /Library/Keychains/System.keychain corp-root.crt",
            "Windows: certutil -addstore -f Root corp-root.crt",
        ],
    },
    "tool_ca_bundle": {
        "summary": "Point tool-specific CA bundles at the system store",
        "steps": [
            "export REQUESTS_CA_BUNDLE={ca_path} SSL_CERT_FILE={ca_path}",
            "export NODE_EXTRA_CA_CERTS={ca_path}",
            "pip config set global.cert {ca_path}",
            "git config --global http.sslCAInfo {ca_path}",
            "conda config --set ssl_verify {ca_path}",
        ],
    },
    "no_proxy_internal": {
        "summary": "Bypass the proxy for internal domains",
        "steps": [
            "export NO_PROXY=localhost,127.0.0.1,.svc,.cluster.local,{internal}",
            "persist NO_PROXY next to HTTPS_PROXY (and lowercase no_proxy for curl/wget)",
        ],
    },
    "dns_failure": {
        "summary": "Let the proxy resolve external names",
        "steps": [
            "make sure HTTPS_PROXY is set so the proxy does DNS (CONNECT host:443)",
            "for SOCKS use socks5h:// (not socks5://) so names resolve at the proxy",
            "check /etc/resolv.conf points at the corporate resolver when on VPN",
        ],
    },
    "clock_skew": {
        "summary": "Synchronise the system clock",
        "steps": [
            "Linux: sudo timedatectl set-ntp true",
            "macOS: sudo sntp -sS time.apple.com",
            "Windows: w32tm /resync",
        ],
    },
}


def render_fix(problem: str, obs: dict | None = None) -> dict:
    obs = obs or {}
    plan = FIXES.get(problem)
    if not plan:
        return {"problem": problem, "summary": "No fix needed", "steps": []}
    proxy = obs.get("proxy_url") or obs.get("hint_proxy") or "http://proxy.corp.example:8080"
    vals = {
        "proxy": proxy,
        "proxy_host": proxy.split("://")[-1],
        "ca_path": obs.get("ca", {}).get("system_bundle_path") or "/etc/ssl/certs/ca-certificates.crt",
        "internal": obs.get("hint_internal_suffix") or ".corp.example",
    }
    return {"problem": problem, "summary": plan["summary"], "steps": [s.format(**vals) for s in plan["steps"]]}
