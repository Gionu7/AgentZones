# Local models

Agents never load a model themselves. They call an **OpenAI-compatible server** with two settings:

| Setting | Meaning | Default |
|---|---|---|
| `MODEL_URL` | the server | `http://ollama:11434/v1` in Docker zones, `http://127.0.0.1:11434/v1` on a laptop |
| `MODEL_NAME` | the model the agents ask for | `qwen2.5:1.5b` in Docker zones, `llama3.2:3b` on a laptop |

[Ollama](https://ollama.com) is the recommended server, because `agentzone model setup` can fill it for you.
LM Studio, llama.cpp's `llama-server` and vLLM work too, but you load the model in them yourself.

The model is optional. Proxy Doctor decides with its own small classifier and only uses the LLM to word its
explanation; with no model it explains from a template. Only the `assistant` agent needs an LLM.

## Which model

Sizes are for 4-bit (Q4) versions. Check each licence for your use.

| Machine | Model (`MODEL_NAME`) | Size | Licence | Notes |
|---|---|---|---|---|
| Any CPU, 8 GB RAM | `qwen2.5:1.5b` | 1 GB | Apache-2.0 | default in Docker zones; fine for explanations, weak at tool calling |
| | `llama3.2:1b` | 1.3 GB | Llama 3.2 Community | |
| Laptop, 16 GB RAM or Apple Silicon | **`llama3.2:3b`** | 2 GB | Llama 3.2 Community | default on laptops; tool calling works |
| | `phi4-mini` | 2.5 GB | MIT | good at function calling |
| | `qwen2.5:3b` | 1.9 GB | Qwen Research (non-commercial) | avoid on work machines |
| GPU with 8–12 GB, or a 32 GB Mac | `qwen2.5:7b` | 4.7 GB | Apache-2.0 | reliable tool calling; good for the `assistant` |
| | `llama3.1:8b` | 4.9 GB | Llama 3.1 Community | |
| GPU with 16 GB+ | `qwen2.5:14b` | 9 GB | Apache-2.0 | |
| Proxy Doctor fine-tune | `proxy-doctor` | 1–2 GB | as its base model | your own model, see [DISTILLATION.md](DISTILLATION.md) |

Rule of thumb: the `assistant` (which calls tools) wants 3B or bigger; anything works for Proxy Doctor's
explanations. Newer models work the same way: use their Ollama name.

## Where a model comes from: `MODEL_SOURCE`

`agentzone model setup` puts `MODEL_NAME` into Ollama from `MODEL_SOURCE`, which defaults to `MODEL_NAME`. It only
uses Ollama's HTTP API, so it works the same for the Ollama container of a zone and for Ollama on a laptop.

| `MODEL_SOURCE` | From |
|---|---|
| `llama3.2:3b` | the Ollama library (`registry.ollama.ai`) |
| `hf.co/bartowski/Qwen2.5-7B-Instruct-GGUF:Q4_K_M` | a GGUF repo on Hugging Face, pulled by Ollama itself |
| `https://mirror.corp/models/llama3.2-3b-q4.gguf` | any HTTPS file: an internal mirror, an Azure blob (SAS URL), a Hugging Face `resolve` URL. Downloaded once into a cache; add `MODEL_SHA256` to check it |
| `/local-models/proxy-doctor-1.5b.gguf` | a file. In Docker zones `/local-models` is this repo's `models/` folder; on a laptop use any path |

A model that is already there is left alone (`--force` replaces it). `MODEL_SYSTEM` sets a system prompt for a
model created from a file.

Behind a corporate proxy: `AGENT_HTTPS_PROXY` is used for the downloads, and `HOST_CA_BUNDLE` makes the proxy's
CA trusted. Where public registries are blocked (zone cn, often), use an internal mirror URL or a file.

### Models in the Azure ML registry

The zones do not talk to Azure ML. That would need the Azure CLI and a login on every machine, and cn may not
reach Azure at all. Instead, download the model once where you have access, and serve it as a file or a URL:

```bash
az ml model download --registry-name <registry> --name proxy-doctor --version 1 --download-path models/
```

Then point `MODEL_SOURCE` at the `.gguf` file under `models/proxy-doctor/`. Alternatively, give the zones a
read-only SAS URL of the model's blob as `MODEL_SOURCE`; it is only needed for the first download. Registering a
model is shown in [DISTILLATION.md](DISTILLATION.md).

## Docker zones

| | Command |
|---|---|
| Ollama in the zone, on the CPU | `COMPOSE_PROFILES=llm scripts/zone.sh az` |
| Ollama in the zone, on NVIDIA GPUs | `AGENTZONES_GPU=1 COMPOSE_PROFILES=llm scripts/zone.sh az` (needs the NVIDIA Container Toolkit) |
| the host's own model server | `MODEL_URL=http://host.docker.internal:11434/v1 COMPOSE_PROFILES=host-llm scripts/zone.sh az` |
| a different model | add `MODEL_NAME=qwen2.5:7b`, and `MODEL_SOURCE=...` if it does not come from the Ollama library |

The `model-setup` service runs once per start and exits. Watch it with `scripts/zone.sh az logs model-setup`.
On a Mac, run Ollama natively and use `host-llm`: Docker cannot use the Apple GPU.

The agent image stays small and the same everywhere: no model, no PyTorch, no CUDA. Models live in Ollama's
volume.

## Laptops and personal machines: `agentzone laptop`

On a laptop, run the zone **without Docker**. Proxy Doctor then checks the laptop itself (a container would
only see its own network and certificates), Ollama uses the laptop's GPU, and Docker Desktop is not needed.

1. Install [Ollama](https://ollama.com) (or LM Studio), and Python 3.10 or newer.
2. In a clone of this repository:

   ```bash
   pip install -e .
   ```

3. Run the zone. On its own:

   ```bash
   agentzone laptop
   ```

   Or as a spoke of the hub, with a token that the hub lists in `EXTRA_SPOKE_TOKENS`:

   ```bash
   agentzone laptop --zone laptop-alice --hub http://<az-host>:9090 --token <token>
   ```

The first start pulls `llama3.2:3b` into the local Ollama. Then:

- the dashboard is at http://127.0.0.1:8080/;
- the MCP server is at http://127.0.0.1:8000/mcp, for Claude Desktop or an IDE;
- Proxy Doctor checks this machine: `agentzone ask proxy-doctor "pip fails with SSL errors"`.

Options: `--proxy http://proxy:8080` if the hub is only reachable through a proxy; `--model` and
`--model-source` as above; `--model-url http://127.0.0.1:1234/v1` for LM Studio; `--no-llm` for no model at all;
`--fl` to join federated training of Proxy Doctor's classifier. Everything listens on 127.0.0.1 only, and data
is kept in `~/.agentzones/<zone>`. Proxy Doctor only proposes fixes on a real machine; set
`PROXY_DOCTOR_ALLOW_APPLY=1` to let it apply user-level ones.

Because a laptop is a spoke, it only makes outbound connections. It works behind NAT, on VPN or on hotel Wi-Fi,
and nothing can connect into it.
