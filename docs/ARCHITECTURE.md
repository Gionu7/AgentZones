# Architecture

## A zone

```
             ┌──────────────────────────── zone (one Compose project) ─────────────────────────────┐
 operator ──►│ 127.0.0.1:808x  gateway  local API  /directory  /send  /agents/register  /status    │
 MCP client ►│ 127.0.0.1:800x  mcp ──────┘   ▲          │                                           │
             │                               │ register │ POST /message                             │
             │                agents ────────┘          ▼                                           │
             │                (proxy-doctor, fl-client, fl-aggregator, assistant, ...)              │
             │                  │ MCP (stdio)            │ OpenAI-compatible API                     │
             │                  ▼                        ▼                                           │
             │                MCP servers               ollama (COMPOSE_PROFILES=llm)               │
             └──────────────────────────────────────────────────────────────────────────────────────┘
                 hub zone only: gateway hub API on :9090  ◄── spokes (signed, long-poll)
```

* **Gateway** ([agentzones/gateway.py](../agentzones/gateway.py)): one per zone, configured by environment
  variables (`ZONE`, `ROLE`, `HUB_URL`, `HUB_TOKEN`, `HUB_PROXY`, `SPOKE_TOKENS`). It keeps the list of local
  agents (30 s TTL) and routes messages: to a local agent, to the hub (spoke), or into a mailbox (hub).
* **Agents** ([agentzones/agent.py](../agentzones/agent.py)): one container each, all from the same image.
  `AgentHost` serves `POST /message` and re-registers with the gateway every 10 s. Agents see the gateway
  through `GatewayClient.directory()`, `send()` and `send_all()` (directory + send to each match), nothing
  else. Directory entries carry `zone`, `agent`, `capabilities`, `description`, `version` and `accepts`
  (message type → payload description). Every call carries the agent's `X-Agent-Token`, see
  [SECURITY.md](SECURITY.md#agents-inside-a-zone).
* **MCP server** ([agentzones/mcp_gateway.py](../agentzones/mcp_gateway.py)): `list_agents` and
  `send_message` over streamable HTTP (`POST /mcp`) or stdio, for Claude Desktop, IDEs and other MCP clients.
* **LLM agent** ([agentzones/llm_agent.py](../agentzones/llm_agent.py)): a tool-calling loop on the local
  model with MCP tools plus `zone.list_agents` and `zone.send_message`. Agent-to-agent chains stop after two
  hops.

## Messages

```json
{"id": "…", "kind": "request", "from_zone": "b", "from_agent": "mnist-aggregator",
 "to_zone": "cn", "to_agent": "mnist-client", "payload": {"type": "fl.fit", "...": "..."}, "expires": 1767225600.0}
{"id": "…", "kind": "reply", "reply_to": "<request id>", "from_zone": "cn", "to_zone": "b", "payload": {"...": "..."}}
```

`POST /send` creates a request, waits for the reply (up to `timeout`, default 60 s) and returns its
payload. `expires` travels with the message; the hub drops messages that are past it, and the receiving
gateway gives the agent only the time that is left.

## Hub protocol (spoke → hub, HTTP + JSON, every call signed)

| Call | Purpose |
|---|---|
| `POST /hub/register {agents}` | this spoke's agents for the directory (every 10 s; entries expire after 60 s) |
| `GET /hub/directory` | agents of all zones |
| `GET /hub/poll?wait=20` | long poll: returns this spoke's messages (requests for its agents, replies to its requests) |
| `POST /hub/ack {ids}` | acknowledge received messages |
| `POST /hub/send {message}` | a request from one of this spoke's agents, or a reply from one |

The spoke runs two loops: register, and poll → ack → deliver. It acknowledges a batch as soon as it receives
it and remembers message ids for 15 minutes, so a redelivered message is not handled twice. A message the hub
handed out but did not see acknowledged within `ACK_TIMEOUT_S` (30 s) goes back to the mailbox; the long poll
wakes up for it. Mailboxes hold at most 1000 messages per spoke.

### Routing rules on the hub

* A request's `from_zone` is always the authenticated sender, whatever the message says.
* A request for an unknown zone is refused (HTTP 403, `/send` returns 404).
* When a request is routed, the hub remembers `(from_zone, to_zone)` for it. A reply is accepted only from
  `to_zone`, only once, and delivered only to `from_zone`. A spoke therefore cannot answer requests meant for
  another zone, and cannot inject replies into other zones' conversations.
* Messages for the hub's own zone are delivered to its local agents directly.

## Federated learning

```
aggregator (b)                         hub (az)                 clients (az, b, cn)
  fl.start  ◄──────── from any zone
  directory("fl-client:mnist") ─────► agents of all zones
  round r:  fl.fit {parameters, round, config} ──────────────► fit() on local data
            ◄─────── {status: ok, parameters, num_samples, metrics}
            check shapes, aggregate_fit(), evaluate(), save /models/mnist-latest.npz
  done:     fl.model {parameters} ───────────────────────────► on_global(), FL_MODEL_OUT
```

Parameters travel as a base64 `.npz` (float32) inside the message. The aggregator re-reads the directory every
round. A client that is busy (a redelivered `fl.fit`), times out or returns bad parameters counts as a failure
for that round; the round still counts with `FL_MIN_FIT_CLIENTS` results.

## Proxy Doctor flow

```
ask (own zone only) → diagnose (MCP) → student.classify → confident and a verified fix known? ─yes─► apply → re-diagnose → verified? → store + pd.fix to other zones
                                                             │ no
                                                             ▼
                                         pd.consult (observation only) to the other zones' proxy-doctor
                                         → vote (unsure) or collect their verified fixes (confident)
```

* The **observation** holds only diagnostics: env flags with redacted proxy credentials, HTTP status codes and
  error strings, the TLS issuer, CA store checks and clock skew. Files and secrets are never included.
* The **student** is a softmax classifier over 27 structured features plus 256 hashed error-text features,
  about 2.5k parameters. It is distilled from a teacher and improved with federated learning
  (`tasks/proxy_student.py`). With `PD_MODEL_PATH` the doctor reloads the federated model whenever the file
  changes. The local LLM writes the explanation, but the decision does not depend on it.
* **Verification** means the doctor re-runs the checks after applying the fix and the external (and internal)
  request now succeeds. Only verified fixes are shared.
* **Other zones** can only consult (their observation, classified by our model, plus our verified fixes) and
  push fixes. A pushed fix is stored as verified by the *sending* zone only, and is not applied until the local
  doctor verifies it.
