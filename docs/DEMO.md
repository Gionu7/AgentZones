# Hackathon demo script (about 5 minutes)

## Setup

Without Docker (one process, about 10 seconds):

```bash
pip install -e .
```

```bash
python scripts/demo.py --keep
```

With Docker, on one host and with b's network constraints enforced:

```bash
scripts/zone.sh az
```

```bash
scripts/zone.sh cn
```

```bash
COMPOSE_PROFILES=sim scripts/zone.sh b
```

Add `COMPOSE_PROFILES=llm` (for example `COMPOSE_PROFILES=sim,llm` for b) to get Ollama and LLM-written
explanations.

Open the hub dashboard at http://127.0.0.1:8080/ and keep it on screen for the whole demo. With
`scripts/demo.py --keep`, click *Start run* to replay federated learning live.

## Story

1. **Hub and spoke.** On the dashboard, b and cn are green and polling the hub, b through its proxy, and
   neither accepts connections. In b, show that the gateway can't reach the hub or the internet directly:

   ```bash
   docker exec zone-b-gateway-1 python -c "import urllib.request as u; u.urlopen('http://hub:9090/health', timeout=3)"
   ```

   Then show the squid log, where all of b's hub traffic appears:

   ```bash
   docker exec zone-b-proxy-1 tail /var/log/squid/access.log
   ```

2. **Federated learning across three networks.** The aggregator lives in b, behind the proxy. Clients in az, b
   and cn each hold a third of MNIST. On the dashboard, the weights travel b → proxy → hub → az / cn and back
   each round, and the accuracy chart climbs (about 91% after one round, about 95% after five). The traffic log
   shows the size of every update: only weights move, never data. In the logs:

   ```bash
   scripts/zone.sh b logs -f mnist-aggregator
   ```

   Start a new run from cn, a zone that no one can connect to:

   ```bash
   curl -s localhost:8082/send -H 'content-type: application/json' \
     -d '{"target_zone":"b","target_agent":"mnist-aggregator","payload":{"type":"fl.start","rounds":5}}'
   ```

3. **Non-IID data.** Restart cn's client with only some digits, and watch rounds still converge:

   ```bash
   CN_LABELS=0,1,2,3 scripts/zone.sh cn up -d mnist-client
   ```

4. **Proxy Doctor.** b hits `407 Proxy Authentication Required`, fixes it and sends the verified fix out. Then
   az, which has hardly ever seen that problem, consults the other zones and reuses b's fix:

   ```bash
   python -m agentzones.cli --gateway http://127.0.0.1:8081 ask proxy-doctor "pip: 407" --simulate proxy_auth_required
   ```

   ```bash
   python -m agentzones.cli --gateway http://127.0.0.1:8080 ask proxy-doctor "pip: 407" --simulate proxy_auth_required
   ```

   The fix is only applied on a simulated machine if the zone runs with `PD_AUTO_APPLY=1`. `scripts/demo.py`
   does that; the Docker zones only propose fixes by default.

5. **Security.** Tokens never cross the proxy. A spoke with a wrong token is rejected and appears in red under
   *Rejected requests* and *Events*. A proxy that edits a message breaks its signature
   (`pytest tests/test_zones_e2e.py -k tampering`).

## Talking points

* Fits real networks: one public endpoint, and every other site makes outbound calls only, through its proxy if
  it has to.
* Standard parts: MCP for tools (and an MCP server per zone for Claude Desktop or an IDE), an
  OpenAI-compatible API for models, HTTP and JSON between zones, one Docker image.
* FL is generic: two small plugin classes per model, parameters as numpy arrays, any framework.
* Safe by default: per-zone tokens that never travel, signed answers, localhost-only ports, verify before share.

## Simulated failure classes

`--simulate`: `untrusted_corp_ca`, `tool_ca_bundle`, `proxy_auth_required`, `missing_proxy_env`,
`proxy_unreachable`, `no_proxy_internal`, `dns_failure`, `clock_skew`, `healthy`.
