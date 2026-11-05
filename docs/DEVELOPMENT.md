# Development setup

How to get a working checkout, run zones while you change code, and check your work the way CI does. What
the parts are and how they fit together is in the [README](../README.md) and
[ARCHITECTURE.md](ARCHITECTURE.md).

## Requirements

| What | Needed for |
|---|---|
| Python 3.10 or newer (CI and the Docker image use 3.12) | everything |
| Git | everything |
| Docker with Compose 2.24 or newer (the zone files use `!reset`) | Docker zones only |
| [Ollama](https://ollama.com), or another OpenAI-compatible server | LLM agents only; see [MODELS.md](MODELS.md) |
| An NVIDIA GPU | distillation and fine-tuning only; see [DISTILLATION.md](DISTILLATION.md) |

The tests and the demo need neither Docker, a model, nor the internet: they run every zone in one process on
`127.0.0.1`.

## Install

```bash
git clone https://github.com/Gionu7/AgentZones.git
```

```bash
cd AgentZones
```

```bash
python -m venv .venv
```

```bash
source .venv/bin/activate
```

On Windows: `.venv\Scripts\activate`. Then install the package in editable mode, with the test tools:

```bash
pip install -e '.[dev]'
```

This also installs the `agentzone` command. Code changes take effect the next time a process starts; there
is no build step, not even for the dashboard. The `distill` extra (`pip install -e '.[dev,distill]'`) adds
torch and transformers; you only need it on the GPU box.

Behind a corporate proxy, export `HTTPS_PROXY` before `pip install`. If the proxy inspects TLS, also point
`PIP_CERT` (pip) and `SSL_CERT_FILE` (httpx, model downloads) at a CA bundle that includes the corporate root
CA. For Docker builds, export `BUILD_HTTPS_PROXY` and `BUILD_HTTP_PROXY` (`zone/base.yml` passes them to the
build), put the root CA as a `*.crt` file in `deploy/certs/`, and set `APT_PACKAGES=` if apt cannot reach a
Debian mirror; the comments at the top of [deploy/Dockerfile](../deploy/Dockerfile) have the details.

## Check the setup

```bash
pytest -q
```

About a minute; every test should pass.

```bash
python scripts/demo.py
```

Runs zones az, cn and b with b behind a small HTTP proxy, and walks through messaging, Proxy Doctor,
federated learning, an approval request and a rejected token. It ends with `demo passed`.

## Run zones while you work

| Way | Start it with | Good for |
|---|---|---|
| Demo | `python scripts/demo.py --keep` | the whole network: hub, two spokes, a proxy, FL, the dashboards |
| Laptop zone | `agentzone laptop --no-llm` | one zone: gateway, dashboard, Proxy Doctor, MCP server |
| One part at a time | `agentzone gateway`, `agentzone agent`, ... | working on one component, or your own agent |
| Docker zones | `scripts/zone.sh az` (then `cn`, `b`) | the real images, networks and proxy rules |

### Demo

```bash
python scripts/demo.py --keep
```

After the story it keeps the three zones running on the same ports as the Docker zones: the hub dashboard at
http://127.0.0.1:8080/, b at 8081, cn at 8082. If a port is taken it picks a free one and prints the URLs.
`--llm` adds the `assistant` and `code-assistant` agents in az, on `MODEL_URL` (default: Ollama on this
machine); `--rounds N` changes the FL run. Each run starts with fresh state in a temporary folder. Ctrl+C
stops it.

### Laptop zone

```bash
agentzone laptop --no-llm
```

One standalone zone in one process: dashboard on http://127.0.0.1:8080/, MCP server on
http://127.0.0.1:8000/mcp, data in `~/.agentzones/<zone>/`. With Ollama running, leave out `--no-llm` to get
the `assistant` agent, and add `--code .` for a `code-assistant` that reads this repository. `--fl` joins
federated training, and `--hub URL --token T` makes the zone a spoke of a hub. `agentzone laptop --help`
lists the rest, including `--port` and `--mcp-port`.

### One part at a time

Every part reads its settings from environment variables, so you can start a zone piece by piece. A gateway
and an echo agent, each in its own terminal:

```bash
ZONE=dev GATEWAY_HOST=127.0.0.1 agentzone gateway
```

```bash
AGENT_NAME=echo AGENT_IMPL=agentzones.agent:EchoAgent GATEWAY_URL=http://127.0.0.1:8080 AGENT_PORT=9001 AGENT_URL=http://127.0.0.1:9001 agentzone agent
```

Then talk to it from a third:

```bash
agentzone send dev echo '{"hello": "zone"}'
```

Replace `AGENT_IMPL` with `your.module:YourAgent` to run your own agent. Settings are documented where they
are read: `GatewaySettings.from_env` in [agentzones/gateway.py](../agentzones/gateway.py), the docstrings of
[agentzones/agent.py](../agentzones/agent.py), [agentzones/fl/aggregator.py](../agentzones/fl/aggregator.py)
and [proxy_doctor/agent.py](../proxy_doctor/agent.py), and the Docker defaults in
[zone/base.yml](../zone/base.yml) and [.env.example](../.env.example).

### Docker zones

The commands are in the README under
[Run locally with Docker](../README.md#run-locally-with-docker-all-three-zones-on-one-host). While developing:

- `scripts/zone.sh <zone>` runs `docker compose up -d --build`, so running it again rebuilds the image with
  your changes and restarts what changed.
- `scripts/zone.sh <zone> logs -f <service>` follows a service's log.
- `scripts/zone.sh <zone> down` stops a zone; add `-v` to also delete its volumes (MNIST data, Ollama
  models, saved FL models, Proxy Doctor's fixes).

### Ports

| Port on 127.0.0.1 | What |
|---|---|
| 8080, 8081, 8082 | local API and dashboard of az, b, cn (a laptop zone uses 8080) |
| 9090 | hub API of az |
| 8000, 8001, 8002 | MCP servers of az, b, cn (a laptop zone uses 8000) |

The demo with `--keep`, a laptop zone and the Docker zones all use some of these ports, so run one of them
at a time, or move the laptop zone with `--port` and `--mcp-port`.

## Tests

```bash
pytest tests/test_hub.py -q
```

```bash
pytest -q -k federated
```

The tests are plain `async def` functions: pytest-asyncio runs in auto mode, so they need no marker.

| File | Covers |
|---|---|
| `test_security.py`, `test_hub.py` | request signing and replay protection; hub routing, redelivery, offline spokes |
| `test_identity.py` | per-agent identity on the gateway, directory entries, `send_all` |
| `test_fl.py`, `test_fl_control.py` | FedAvg and parameter checks; approval, limits, audit log, update screening |
| `test_llm_agent.py`, `test_chat.py`, `test_mcp.py` | the LLM agent's tool loop against a fake model, remote tool limits, the MCP servers |
| `test_models.py` | `agentzone model setup` against a fake Ollama, and `agentzone laptop` |
| `test_dashboard.py`, `test_knowledge.py` | the dashboard data, the store of verified fixes |
| `test_zones_e2e.py` | three live zones over HTTP, b through a proxy (and a tampering one), Proxy Doctor and a full FL run |

[tests/conftest.py](../tests/conftest.py) has the helpers for live zones. Inside a
`contextlib.AsyncExitStack`:

- `run_gateway(stack, zone=..., role=..., ...)` starts a gateway with short timeouts and returns
  `(gateway, local_url, hub_url)`. It is an async context manager: `await stack.enter_async_context(...)`.
- `run_agent(stack, gateway_url, name, "module:Class", **env)` starts an agent host.
- `wait_for(predicate)` polls until a condition holds (registration, a state change).
- `RunningServer(app, port)` and `free_port()` serve any ASGI app, such as a fake model server;
  `ForwardProxy(tamper=...)` is the HTTP proxy that stands in for zone b's.

[tests/test_identity.py](../tests/test_identity.py) is a short example that uses most of them.

## Before you push

CI ([.github/workflows/ci.yml](../.github/workflows/ci.yml)) runs on every push and pull request, with
Python 3.12:

```bash
pytest -q
```

```bash
python scripts/demo.py
```

and checks that every zone file is valid (this needs a running Docker daemon):

```bash
docker network create agent-mesh
```

```bash
for z in az cn b; do COMPOSE_PROFILES=sim scripts/zone.sh "$z" config -q; done
```

There is no linter in CI. `ruff check --select F .` (after `pip install ruff`) finds unused imports and
names; the tree is clean.

If you change an FL plugin or Proxy Doctor's model, regenerate the charts in the docs:

```bash
python scripts/make_doc_charts.py
```

The runs are deterministic, so if `git status` shows no change in `docs/images/`, your change did not move
the numbers. If it did on purpose, commit the new charts and update the numbers in the README,
[ARCHITECTURE.md](ARCHITECTURE.md) and [DISTILLATION.md](DISTILLATION.md); the script prints them as markdown
tables. Set `MNIST_PATH` to an
`mnist.npz` to also redo the MNIST chart.

## Where changes go

| Change | Where | Notes |
|---|---|---|
| A new agent | your own module, a subclass of `agentzones.agent.Agent` | implement `handle`; set `capabilities` and `accepts`. In a zone file, add a service with `extends: { file: ../../zone/base.yml, service: agent }`, `AGENT_NAME` and `AGENT_IMPL`. Test it with `run_agent` |
| A new FL model | `tasks/<name>.py`: a `ClientTask` and a `Strategy` | point `FL_TASK` / `FL_STRATEGY` at them. Try it in one process first, as the `__main__` block of `tasks/mnist.py` does (`MNIST_PATH=data/mnist.npz python -m tasks.mnist`) |
| An MCP tool | a function with `@srv.tool()` on a `MiniMCPServer` | type hints become the input schema, the docstring the description. Examples: [proxy_doctor/mcp_server.py](../proxy_doctor/mcp_server.py), [agentzones/code_mcp.py](../agentzones/code_mcp.py) |
| A setting | an environment variable, read where it is used | `env_flag` and `env_list` in [agentzones/config.py](../agentzones/config.py) parse booleans and lists. Document it in the module docstring, and for Docker zones in `zone/base.yml` or `.env.example` |
| The dashboard | [agentzones/static/dashboard.html](../agentzones/static/dashboard.html), one file with no build step | its data comes from `GET /dashboard/data` ([agentzones/dashboard.py](../agentzones/dashboard.py)). The gateway reads the page when it starts: restart it to see an edit |
| Hub protocol, signing, routing | `gateway.py`, `hub.py`, `security.py` in `agentzones/` | covered by `test_security.py`, `test_hub.py` and `test_zones_e2e.py`; update [ARCHITECTURE.md](ARCHITECTURE.md) and [SECURITY.md](SECURITY.md) too |
| Libraries for a plugin (torch, ...) | the `pip install` line in [deploy/Dockerfile](../deploy/Dockerfile) | keep them out of the core dependencies in `pyproject.toml` |

## Conventions

- Few dependencies: the core is FastAPI, uvicorn, httpx, pydantic, PyYAML and numpy. Heavy libraries go in
  the `distill` extra or the image.
- No MCP SDK: [agentzones/mcp_lite.py](../agentzones/mcp_lite.py) implements the protocol; its docstring
  says why.
- HTTP clients that talk to the gateway, to agents or to the hub are made with `trust_env=False`, so a proxy
  set in the environment never sees zone traffic. Do the same in new code. Hub traffic goes through a proxy
  only when `HUB_PROXY` says so. The clients for the model and for MCP servers over HTTP do honour the proxy
  variables (a model may be remote), which is why `zone/base.yml` lists `ollama` and `host.docker.internal`
  in `NO_PROXY`.
- Files that another process reads are written atomically (`agentzones.fl.save_npz`, `FixStore`).
- Code targets Python 3.10, with `from __future__ import annotations` for the newer type syntax. Lines are
  up to about 120 characters.

## Troubleshooting

| Problem | Fix |
|---|---|
| `address already in use` | another setup holds the port; see [Ports](#ports) |
| HTTP 401 `missing or bad bearer token` | the zone has `ZONE_API_TOKEN` set: open the dashboard as `http://127.0.0.1:8080/#token=<token>`, and export the same `ZONE_API_TOKEN` for `agentzone` commands |
| `network agent-mesh declared as external, but could not be found` | you ran `docker compose` directly; `docker network create agent-mesh`, or use `scripts/zone.sh` |
| FL rounds are very slow with several agents on one host | numpy is using every core in every agent; keep `AGENT_BLAS_THREADS` low (default 2) |
| `model setup failed` or the LLM agents answer with an error | no model server at `MODEL_URL`; see [MODELS.md](MODELS.md). Proxy Doctor still works without one |
