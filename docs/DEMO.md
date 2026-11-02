# Hackathon demo script (about 5 minutes)

## Setup

Install the package, then start the demo and leave the zones running:

```bash
pip install -e .
```

```bash
python scripts/demo.py --keep
```

Open three browser tabs: http://127.0.0.1:7701, :7702 and :7703. These are the zone-a, zone-b and zone-c dashboards.

Optional, for LLM-written explanations:

```bash
ollama pull qwen2.5:1.5b
```

```bash
MODEL_PROVIDER=openai python scripts/demo.py --keep
```

Optional, the same three zones in Docker:

```bash
docker compose -f deploy/docker-compose.demo.yml up -d --build
```

```bash
python scripts/demo.py --no-launch --zones http://localhost:7701 http://localhost:7702 http://localhost:7703
```

Optional, across real machines. Run one zone per laptop or VM with `ZONE_SEED` set to a neighbour's URL and the
same `ZONE_SECRET` everywhere (see `.env.example`).

## Story

1. **One command per zone, no registry.** zone-c only knows zone-b, and zone-b only knows zone-a. Show the
   *Peer zones* panel: within seconds everyone knows everyone, through gossip.
2. **A local fix.** zone-b is a build server and hits `407 Proxy Authentication Required`. Its own model is
   confident. It applies the fix, re-runs the checks, marks the fix **verified** and shares it.
   The *Verified fixes* panel fills in on zone-a and zone-c.
3. **Asking for help.** zone-a is a developer laptop and hits the same 407, which it has hardly ever seen.
   Its local model is unsure, so it asks its peers and sends only the diagnostic observation. zone-b is
   confident and has a verified fix. zone-a applies it, verifies it locally and adds itself to `verified_by`.
4. **Federated learning.** Click *Full round* on each dashboard, or let the script do it. The chart shows
   benchmark accuracy going from about 45% (each site only knows its own problems) to about 95%. No incident
   data left any zone.
5. **Real machine.** Pick *real machine* in the dashboard and ask about `pypi.org`. The doctor runs real
   checks: proxy env, TCP to the proxy, HTTPS with and without verification, certificate issuer, CA bundles
   and clock skew. A nice live trick is to unset `REQUESTS_CA_BUNDLE` behind a TLS-intercepting proxy. The
   doctor then reports `tool_ca_bundle` and gives the exact `pip config` and `git config` lines.

## Talking points

* Costs nothing per token, uses idle machines, and the data stays on site. The model runs on a laptop CPU
  (student) or a small local LLM.
* Standard parts: MCP for tools, an OpenAI-compatible API for models, HTTP and JSON between zones. Any MCP
  server, such as a future Azure ML one, plugs straight in.
* Safe by default: signed peer calls, tools private unless exposed, dry-run unless allowed, verify before share.

## Simulated failure classes

`--simulate` / dashboard: `untrusted_corp_ca`, `tool_ca_bundle`, `proxy_auth_required`, `missing_proxy_env`,
`proxy_unreachable`, `no_proxy_internal`, `dns_failure`, `clock_skew`, `healthy`.
