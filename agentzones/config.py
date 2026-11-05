"""Settings shared by agents: the local model and MCP servers.

Zones themselves are configured by environment variables in their compose file
(zones/<zone>/zone.yml); see gateway.GatewaySettings and agent.AgentHost.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class ModelConfig(BaseModel):
    """Local language model. Anything that speaks the OpenAI chat API works."""

    provider: Literal["openai", "none"] = "openai"
    base_url: str = "http://localhost:11434/v1"  # Ollama default
    model: str = "qwen2.5:1.5b"
    api_key: str = "local"
    temperature: float = 0.2
    timeout_s: float = 120.0


class MCPServerConfig(BaseModel):
    name: str
    # stdio transport: command + args. streamable HTTP transport: url.
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    url: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)
    # Tools other zones may call. "*" = all tools of this server. Default: none.
    expose: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _one_transport(self) -> "MCPServerConfig":
        if bool(self.command) == bool(self.url):
            raise ValueError(f"mcp server {self.name!r}: set exactly one of 'command' or 'url'")
        return self


def mcp_servers_from_env(value: str) -> list[MCPServerConfig]:
    """``MCP_SERVERS`` is a JSON list of MCPServerConfig objects."""
    if not value.strip():
        return []
    import json

    return [MCPServerConfig.model_validate(v) for v in json.loads(value)]


def env_bool(value: str | None, default: bool) -> bool:
    """An on/off setting: 1/true/yes/on; unset or empty gives ``default``."""
    return default if value is None or value == "" else value.strip().lower() in ("1", "true", "yes", "on")


def model_from_env(env: dict[str, str]) -> ModelConfig:
    return ModelConfig(provider=env.get("MODEL_PROVIDER", "openai"),  # type: ignore[arg-type]
                       base_url=env.get("MODEL_URL", "http://ollama:11434/v1"),
                       model=env.get("MODEL_NAME", "qwen2.5:1.5b"), api_key=env.get("MODEL_API_KEY", "local"),
                       timeout_s=float(env.get("MODEL_TIMEOUT_S", "120")))
