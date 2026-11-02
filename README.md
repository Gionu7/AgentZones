# Agent Zones

**Peer-to-peer, self-hosted AI agents.** An *Agent Zone* is one Docker container (or one process)
that hosts AI agents, MCP servers and a local language model on a machine that is already there:
a workstation, a server, an Azure VM or a Kubernetes cluster. Zones find each other through known peers,
with **no central registry**. Agents can ask agents in other zones and use the tools those zones expose.
Zones can also **train a shared model with federated learning** without exchanging raw data.

The demo agent is **Proxy Doctor**. It diagnoses and fixes corporate proxy and certificate problems on
the machine it runs on. When it is unsure it asks peer zones for help, and if they are unsure too, the local
LLM investigates. It verifies a fix before sharing it, and signs it so other zones know who verified it.

Every zone is also an **MCP server**, so an IDE assistant (Claude Code, Claude Desktop, VS Code) can plug into
the whole zone network.

```
     site A (dev laptops)            site B (build VMs)            site C (Kubernetes)
  +------------------------+     +------------------------+     +------------------------+
  | zone-a                 |     | zone-b                 |     | zone-c                 |
  |  agents: proxy-doctor  |<--->|  agents: proxy-doctor  |<--->|  agents: proxy-doctor  |
  |          assistant     |     |          assistant     |     |          assistant     |
  |  MCP:    proxy tools   |     |  MCP:    proxy tools   |     |  MCP:    proxy tools   |
  |  model:  Ollama (local)|     |  model:  Ollama (local)|     |  model:  Ollama (local)|
  |  fixes + FL student    |     |  fixes + FL student    |     |  fixes + FL student    |
  +------------------------+     +------------------------+     +------------------------+
        ^  gossip /p2p/announce, signed asks, exposed tools, verified fixes, FL weights  ^
        +--------------------------------------------------------------------------------+
```

## Quick start

**Demo with three zones on one machine** (Python 3.10+, no Docker or GPU needed):

```bash
pip install -e '.[dev]'
```

```bash
python scripts/demo.py
```

The script starts `zone-a`, `zone-b` and `zone-c`, then walks through discovery, asking peers, sharing a
verified fix and federated learning. Add `--keep` to leave the zones running so you can open the dashboards at
http://127.0.0.1:7701, :7702 and :7703. To get LLM-written explanations, run Ollama and set `MODEL_PROVIDER=openai`.

**One zone with a local model** (Docker):

```bash
cp .env.example .env
```

Edit `.env` (zone name, URL, seed peer and shared secret), then:

```bash
docker compose up -d
```

This starts the zone, Ollama, and a one-time download of `qwen2.5:1.5b`. The dashboard is at http://localhost:7700.
Running `./scripts/up.sh` does the same, and falls back to a local virtualenv when Docker is not available.

**Other targets:** an Azure VM with `deploy/azure/deploy-vm.sh`, Kubernetes with `deploy/k8s/`, or three zones
in Docker with `deploy/docker-compose.demo.yml`.

## One small config file

```yaml
name: zone-a
advertise_url: http://10.1.2.3:7700     # how peers reach this zone
model: {base_url: http://localhost:11434/v1, model: qwen2.5:1.5b}   # any OpenAI-compatible server
discovery: {peers: ["http://10.1.2.4:7700"]}                       # any zone already in the network
security: {network_secret: ${ZONE_SECRET}}                         # shared by all zones, HMAC-signs peer calls
mcp_servers:
  - {name: proxy, command: python, args: [-m, proxy_doctor.mcp_server], expose: [list_problems]}
agents:
  - {name: proxy-doctor, kind: python, impl: proxy_doctor.agent:ProxyDoctor}
  - {name: assistant, kind: llm, tools: ["proxy.*"]}               # generic tool-calling agent
federated: {enabled: true, task: proxy_doctor.student:ProxyStudentTask}
```

`agentzone init` writes a commented starter file. Any value can be written as `${ENV_VAR:-default}`.

## Use the zone network from your IDE (MCP)

Every zone serves MCP at `http://<zone>:7700/mcp`. Its tools are `list_zones`, `ask_agent` (this zone or any
peer), `ask_all_zones`, `list_tools`, `call_tool` (local tools, or the tools a peer exposes), `search_fixes`
and `diagnose_proxy`.

Claude Code or VS Code (HTTP):

```bash
claude mcp add --transport http agentzones http://127.0.0.1:7700/mcp
```

Claude Desktop (stdio bridge), in `claude_desktop_config.json`:

```json
{"mcpServers": {"agentzones": {"command": "agentzone", "args": ["--zone", "http://127.0.0.1:7700", "mcp"]}}}
```

The assistant acts as the local user. The same rules apply as for `/api/*`: it must connect from loopback or
send the `api_token`. Browser pages from other origins are refused.

## CLI

```
agentzone up -c zone.yaml [--peer URL]     run a zone
agentzone info | peers | tools | fixes     inspect the zone   (--zone URL for another one)
agentzone ask proxy-doctor "pip fails with CERTIFICATE_VERIFY_FAILED" [--simulate tool_ca_bundle]
agentzone call proxy.diagnose '{"target": "https://pypi.org"}'
agentzone fl status | train | aggregate | round
agentzone mcp                              stdio MCP server for IDE assistants
```

## What is in the box

| Part | Where | What it does |
|---|---|---|
| Zone runtime | `agentzones/zone.py`, `server.py` | FastAPI app: local API, dashboard, peer API under `/p2p/*` |
| Discovery | `agentzones/peers.py` | Gossip from seed peers, peer cards, TTL expiry, no registry |
| Security | `agentzones/security.py` | HMAC-SHA256 request signing, replay window, CIDR allow-list, API token |
| Identity | `agentzones/identity.py` | Ed25519 key per zone, key pinning, signed attestations of verified fixes |
| MCP gateway | `agentzones/zone_mcp.py` | The zone as an MCP server (`/mcp` and `agentzone mcp`) for IDE assistants |
| MCP | `agentzones/mcp_lite.py` | Dependency-free MCP client (stdio and streamable HTTP) and server |
| Agents | `agentzones/agent.py` | LLM tool-calling loop. Custom Python agents. `ask_peers`, `call_peer_tool` |
| Knowledge | `agentzones/knowledge.py` | Verified fixes. `verified_by` is computed only from valid signatures |
| Federated learning | `agentzones/federated.py` | FedAvg with no coordinator (train, then aggregate peer snapshots), optional DP noise |
| Proxy Doctor | `proxy_doctor/` | Real diagnostics, simulator, MCP server, student model, LLM investigator, agent |
| Model check | `scripts/llm_check.py` | Measures tool-calling and diagnosis-JSON reliability of local models |
| Distillation | `proxy_doctor/distill/` | Teacher labels, then student; SFT data, then LoRA, then GGUF, then Ollama |
| Deploy | `docker-compose.yml`, `deploy/` | Docker, GPU override, 3-zone demo, Kubernetes, Azure VM, Azure ML job |

Docs: [architecture and protocol](docs/ARCHITECTURE.md), [security and allowed traffic](docs/SECURITY.md),
[distillation and federated learning](docs/DISTILLATION.md), [hackathon demo script](docs/DEMO.md).

## Tests

```bash
pytest -q
```

The tests cover:

- request signing, gossip, the MCP stdio round-trip and FedAvg
- three live zones over HTTP: signed and unsigned calls, exposed and hidden tools, fix sharing and a federated round
- forged attestations, malicious fix steps and key pinning
- the LLM agent loop and the LLM investigator, against a fake OpenAI endpoint
- the zone MCP gateway over HTTP and over the stdio bridge
