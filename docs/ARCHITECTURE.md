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
* **Identity:** each zone has an Ed25519 key (`data_dir/zone_key.pem`) and publishes the public key in its card.
  Peers pin the first key they see for a zone name, or use pre-shared `security.trusted_keys`. A card with a
  different key for a pinned name is refused.
* **Fix store:** verified fixes as JSON in `data_dir`. Each fix carries *attestations*: Ed25519 signatures by
  the zones that applied it and saw their checks pass. `verified_by` is recomputed from the attestations that
  verify against pinned keys. Whatever the sender claims is ignored, and a fix with no valid attestation is
  refused. Agents can add domain validators (`zone.fix_validators`). Peers' fixes are never auto-applied.
* **MCP gateway:** `POST /mcp` turns the zone into an MCP server for IDE assistants (`agentzones/zone_mcp.py`).
  `agentzone mcp` is a stdio bridge to it.
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
diagnose (MCP) -> student.classify -> confident and a verified fix known? --yes--> apply -> re-diagnose -> verified? -> sign, store, share
                                         | no                                         ^
                                         v                                            |
                       ask_peers(observation only) -> vote (unsure) / collect fixes   |
                                         | still < investigate_below (0.5)            |
                                         v                                            |
                       local LLM investigates -> votes with a catalog class ----------+
                                              -> "unknown": stop, show hypothesis + next checks, change nothing
```

* The **observation** holds only diagnostics: env flags with redacted proxy credentials, HTTP status codes and
  error strings, the TLS issuer, CA store checks and clock skew. Files and secrets are never included.
* The **student** is a softmax classifier over 27 structured features plus 256 hashed error-text features,
  about 2.5k parameters in total. It is distilled from a teacher and improved with FL. The local LLM writes the
  explanation, but the decision does not depend on it.
* **LLM investigation** runs when the combined confidence stays below `investigate_below`, or when
  `context.investigate` is set. The local model gets the observation and the opinions so far. On a real
  machine it can also run read-only checks: `check_proxy_env`, `check_https`, `inspect_tls` and `dns_lookup`.
  It must answer with JSON naming a catalog class, or `unknown`.
  * A catalog class is one more vote.
  * `unknown`, or any answer outside the catalog, stops the flow. Nothing is applied, and the user gets the
    model's reasoning and the next checks to run.

  The LLM can therefore *decide* but never *act* outside the catalog.
* **Verification** means the doctor re-runs the checks after applying the fix and the external (and internal)
  request now succeeds. Only verified fixes are signed and shared.
* **Fixes travel as catalog templates** (`{proxy}`, `{ca_path}` unrendered). Each zone renders them with its
  own observation. `catalog.validate_fix` refuses any step that is not a catalog template for that problem.
  So no free text from a peer is ever shown to a user or executed, and the same fix has the same ID on every
  zone, which lets attestations from different zones accumulate.
* **Peers answering:** a peer classifies *your* observation with *its* model and returns *its* verified fixes.
  It never runs diagnostics on its own machine for your problem.
