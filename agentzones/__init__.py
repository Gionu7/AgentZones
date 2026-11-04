"""Agent Zones: self-hosted AI agents in a hub-and-spoke network.

A zone is a Docker Compose stack (zones/<zone>/zone.yml) with:
  * a gateway (gateway.py): the only thing agents talk to. The hub keeps a mailbox per spoke
    and the agent directory; spokes only open outbound connections (long-polling the hub).
  * agents (agent.py): small HTTP services that register with their gateway
  * an MCP server (mcp_gateway.py) with list_agents / send_message
  * optionally a local LLM (Ollama) used by LLM agents (llm_agent.py)
  * generic federated learning (fl/): model-specific code lives in plugins (tasks/)
"""

__version__ = "0.2.0"
