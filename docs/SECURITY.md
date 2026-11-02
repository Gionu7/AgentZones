# Security and allowed traffic between zones

## Traffic matrix

| From | To | Port / protocol | Required | Notes |
|---|---|---|---|---|
| Zone | Peer zones | TCP 7700, HTTP(S) `/p2p/*` | yes | Gossip, ask, tool calls, fixes, FL weights. HMAC-signed |
| User / CLI / browser | Own zone | TCP 7700 `/api/*`, `/` | yes | Bind to localhost or require `api_token` |
| Zone | Local model | TCP 11434 (Ollama) / 8000 (vLLM) | yes | Same host or pod network only |
| Zone | MCP servers | stdio (child process) or HTTP | yes | Keep HTTP MCP servers on localhost or the pod network |
| Ollama / build | Model and image registries | HTTPS via corporate proxy | setup only | `registry.ollama.ai`, `huggingface.co`, `ghcr.io`, Docker Hub, PyPI |
| Proxy Doctor | Test targets | HTTPS via proxy | when diagnosing | HEAD requests to e.g. `pypi.org`, plus the internal URL you name |

Zones never need inbound internet access. A zone that only asks questions (a laptop behind NAT) still works:
it gossips outward and peers answer on the same connection. It is simply not reachable for questions from
other zones.

## Controls built in

* **Request signing:** `security.network_secret` is shared by all zones in one network. Every peer request
  carries `X-AZ-Timestamp` and `X-AZ-Signature = HMAC-SHA256(secret, method, path, timestamp, sha256(body))`.
  Requests older than `max_clock_skew_s` (default 120 s) are rejected, which limits replay.
* **Source allow-list:** `security.allowed_cidrs` restricts which networks may connect at all.
* **Local API token:** `security.api_token` must be sent as a bearer token on `/api/*` from any non-loopback
  address. The dashboard accepts `#token=...` in its URL.
* **Least exposure:**
  * MCP tools are private unless listed in `expose`.
  * Agents can be `public: false`.
  * Fixes from peers are stored but never auto-applied. The local agent re-verifies them.
  * Proxy Doctor applies real changes only with `PROXY_DOCTOR_ALLOW_APPLY=1`, and only user-level ones:
    an env file, `pip config` and `git config`. System CA installs and clock fixes are returned as steps
    for a person to run.
* **Data minimisation:**
  * Only diagnostic observations leave a zone, with proxy credentials redacted.
  * Federated learning shares model weights, never data. You can add `dp_noise_std` noise to the weights.
* **Loop and abuse limits:** hop limit, trace, request-id deduplication, and timeouts on every peer call.
* **Container:** runs as a non-root user (uid 10001), corporate CAs are baked in from `deploy/certs/`, and the
  NetworkPolicy in `deploy/k8s/` lets only the zone reach the model.

## Recommended for anything beyond a hackathon

1. **TLS between zones:** put the zone behind an ingress or a reverse proxy with TLS, or use mTLS
   from a service mesh. HMAC protects integrity, not confidentiality.
2. **Per-zone identity:** replace the shared secret with per-zone key pairs (Ed25519) and a signed list of
   allowed zones, so one compromised zone cannot impersonate others. The `X-AZ-Zone` header is already there
   for this.
3. **Policy for remote tool calls:** per-peer allow-lists, rate limits and an audit log of `/p2p/tools/call`.
4. **Prompt-injection hygiene:** treat peer answers as untrusted data. Proxy Doctor only uses peer
   *classifications* and fix *steps* from a known catalog format, and it re-verifies everything. A generic LLM
   agent should do the same.
5. **Secure aggregation for FL:** masking or Paillier so no zone sees another zone's individual update.
