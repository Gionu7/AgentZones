"""Zone configuration: one small YAML file per zone.

Values of the form ``${VAR}`` or ``${VAR:-default}`` are expanded from the
environment, so the same file works locally, in Docker and on Kubernetes.
"""

from __future__ import annotations

import os
import re
import socket
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator

_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        return _ENV_RE.sub(lambda m: os.environ.get(m.group(1), m.group(2) or ""), value)
    if isinstance(value, list):
        return [_expand(v) for v in value]
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    return value


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


class AgentConfig(BaseModel):
    name: str
    description: str = ""
    # "llm": generic tool-calling loop. "python": custom class "module:Class".
    kind: Literal["llm", "python"] = "llm"
    impl: str | None = None
    system_prompt: str = "You are a helpful agent running inside an Agent Zone."
    # Tool names (qualified "server.tool") or "server.*". Empty = all local tools.
    tools: list[str] = Field(default_factory=list)
    # Let the agent ask agents in peer zones (adds the ask_peers tool).
    can_ask_peers: bool = True
    # Accept questions from peer zones.
    public: bool = True
    max_steps: int = 8
    options: dict[str, Any] = Field(default_factory=dict)


class SecurityConfig(BaseModel):
    # Shared secret of the zone network. Every peer request is HMAC-signed with it.
    # Empty = unsigned (only acceptable on a laptop).
    network_secret: str = ""
    # Source CIDRs allowed to call this zone. Empty = any.
    allowed_cidrs: list[str] = Field(default_factory=list)
    max_clock_skew_s: int = 120
    # Bearer token for the local /api/* (users, CLI, dashboard) from non-loopback
    # addresses. Empty = open (fine on a laptop, not on a shared network).
    api_token: str = ""
    # Pre-shared Ed25519 public keys of zones (name -> base64). These always win.
    trusted_keys: dict[str, str] = Field(default_factory=dict)
    # Trust-on-first-use: pin the first key seen for unknown zone names.
    # false = only zones listed in trusted_keys are accepted as peers.
    tofu: bool = True


class DiscoveryConfig(BaseModel):
    peers: list[str] = Field(default_factory=list)  # seed URLs
    gossip_interval_s: float = 15.0
    peer_ttl_s: float = 90.0
    max_peers: int = 64


class FederatedConfig(BaseModel):
    enabled: bool = False
    task: str = "proxy_doctor.student:ProxyStudentTask"
    data: dict[str, Any] = Field(default_factory=dict)  # passed to the task
    local_epochs: int = 5
    learning_rate: float = 0.5
    dp_noise_std: float = 0.0  # Gaussian noise added to shared weights
    auto_rounds: int = 0  # >0: run rounds in the background
    round_interval_s: float = 30.0


class ZoneConfig(BaseModel):
    name: str = Field(default_factory=socket.gethostname)
    host: str = "0.0.0.0"
    port: int = 7700
    # URL peers should use to reach us. Default: http://<hostname>:<port>
    advertise_url: str | None = None
    data_dir: str = "./.zone-data"
    tags: list[str] = Field(default_factory=list)
    model: ModelConfig = Field(default_factory=ModelConfig)
    mcp_servers: list[MCPServerConfig] = Field(default_factory=list)
    agents: list[AgentConfig] = Field(default_factory=list)
    discovery: DiscoveryConfig = Field(default_factory=DiscoveryConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    federated: FederatedConfig = Field(default_factory=FederatedConfig)

    @property
    def url(self) -> str:
        if self.advertise_url:
            return self.advertise_url.rstrip("/")
        return f"http://{socket.gethostname()}:{self.port}"


def load_config(path: str | os.PathLike | None = None, overrides: dict | None = None) -> ZoneConfig:
    raw: dict[str, Any] = {}
    if path:
        text = Path(path).read_text(encoding="utf-8")
        raw = yaml.safe_load(text) or {}
    raw = _expand(raw)
    for key, val in (overrides or {}).items():
        if val is not None:
            raw[key] = val
    return ZoneConfig.model_validate(raw)
