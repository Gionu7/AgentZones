"""Agent Zones: peer-to-peer, self-hosted AI agents.

A zone = one process (usually one container) that hosts:
  * a local LLM endpoint (OpenAI-compatible: Ollama, llama.cpp, vLLM, ...)
  * MCP servers (stdio or streamable HTTP) whose tools agents can use
  * agents (LLM tool-calling agents or custom Python agents)
  * a peer table, kept fresh by gossip with known peers (no central registry)
  * a shared knowledge store (verified fixes) and a federated-learning endpoint
"""

__version__ = "0.1.0"
