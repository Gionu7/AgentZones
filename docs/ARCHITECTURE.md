# Architecture

## A zone

```
            +--------------------------- zone (one container / process) ---------------------------+
 user/CLI ->| /api/*  dashboard  ----> agents ----------> tool registry ----> MCP clients ----> MCP servers (stdio / HTTP)
            |                            |   \                 |  zone.ask_peers / call_peer_tool / search_fixes
            |                            |    \--> LLM client --------------------------------> local model (Ollama, vLLM, llama.cpp)
            |                            v
 peers   -->| /p2p/*  (HMAC)  -> peer table (gossip) | fix store | federated node (task + snapshot)
            +------------------------------------------------------------------------------------------------+
```

* **Model:** any OpenAI-compatible `/v1/chat/completions` endpoint. The zone never calls a cloud API
  unless you point `model.base_url` at one.
* **MCP servers:** started as child processes (stdio) or reached over streamable HTTP. Their tools are named
  `<server>.<tool>`. A tool is visible to peers only if it matches the server's `expose` patterns.
* **Agents:**
  * `kind: llm` is a generic tool-calling loop. It gets the selected tools plus `zone.ask_peers`,
    `zone.list_peers` and `zone.call_peer_tool`.
  * `kind: python` is any class `module:Class` that subclasses `agentzones.agent.Agent`.
    `ProxyDoctor` is one.
* **Fix store:** verified fixes as JSON in `data_dir`. Fixes received from peers are kept together with the
  list of zones that verified them. Peers' fixes are never auto-applied without re-verification.
* **Federated node:** wraps a *task* (`get_weights`, `set_weights`, `train`, `evaluate`, `n_samples`).

## Peer protocol (HTTP + JSON, all requests HMAC-signed)

| Call | Purpose |
|---|---|
| `POST /p2p/announce` (card) → `{self: card, peers: [...]}` | Gossip. A card holds name, url, public agents, exposed tools, FL task and model name |
| `POST /p2p/ask` `{agent, question, context, ctx}` | Ask a public agent. `ctx` carries origin, hops, trace and request_id |
| `POST /p2p/tools/call` `{tool, arguments}` | Call an exposed tool |
| `POST /p2p/fixes` (fix) | Push a verified fix |
| `GET /p2p/fl/weights` | Fetch the last *trained* snapshot `{task, round, n_samples, weights}` |

### Discovery without a registry

A zone starts with zero or more seed URLs. Every `gossip_interval_s` it announces itself to every peer it knows.
Each reply returns that peer's card and the peers *it* knows. New URLs are contacted in the next round, so
membership spreads like an epidemic. In the demo `zone-c` only knows `zone-b` and `zone-b` only knows `zone-a`,
yet the network is fully connected within one or two gossip intervals. Peers that stay silent for longer than
`peer_ttl_s` are dropped. Seeds are kept and retried. If a zone comes back under a new URL, entries are merged
by zone name.

### Asking across zones and stopping loops

`ask_peers` sends the question to one public agent in each live peer (by default the agent with the same name).
The `ctx.trace` lists the zones already visited, and `hops` is limited to `MAX_HOPS = 2`. Each zone also drops a
`request_id` it has already seen, so questions cannot bounce around forever.

### Federated learning without a coordinator

1. `train`: each zone trains on local data only and freezes a **snapshot**. Peers only ever see the snapshot.
2. `aggregate`: each zone pulls the snapshots of live peers that run the same task and averages them by
   sample count (FedAvg).

If every zone trains before any zone aggregates, all of them end the round with the same global model.
`scripts/demo.py` does this, and so does `agentzone fl train` followed by `aggregate` on each zone. If the
zones run out of step, for example with `federated.auto_rounds`, the process turns into gossip averaging,
which still converges. You can add `dp_noise_std` to perturb the shared weights. Secure aggregation is on the
roadmap.

## Proxy Doctor flow

```
diagnose (MCP) -> student.classify -> confident and a verified fix known? --yes--> apply -> re-diagnose -> verified? -> store + share
                                         | no
                                         v
                       ask_peers(observation only) -> vote (unsure) / collect fixes (confident)
```

* The **observation** holds only diagnostics: env flags with redacted proxy credentials, HTTP status codes and
  error strings, the TLS issuer, CA store checks and clock skew. Files and secrets are never included.
* The **student** is a softmax classifier over 27 structured features plus 256 hashed error-text features,
  about 2.5k parameters in total. It is distilled from a teacher and improved with FL. The local LLM writes the
  explanation, but the decision does not depend on it.
* **Verification** means the doctor re-runs the checks after applying the fix and the external (and internal)
  request now succeeds. Only verified fixes are shared.
* **Peers answering:** a peer classifies *your* observation with *its* model and returns *its* verified fixes.
  It never runs diagnostics on its own machine for your problem.
