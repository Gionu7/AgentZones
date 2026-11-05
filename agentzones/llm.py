"""Client for the zone's local language model (OpenAI-compatible chat API).

Works with Ollama (``http://ollama:11434/v1``), llama.cpp ``llama-server``,
vLLM, LM Studio, LocalAI and - if a site allows it - Azure OpenAI.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

import httpx

from .config import ModelConfig

log = logging.getLogger(__name__)


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMReply:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)


class LLMUnavailable(RuntimeError):
    pass


class LLM:
    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        self.http = httpx.AsyncClient(timeout=cfg.timeout_s)

    @property
    def enabled(self) -> bool:
        return self.cfg.provider != "none"

    async def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> LLMReply:
        if not self.enabled:
            raise LLMUnavailable("no model configured for this zone")
        body: dict[str, Any] = {"model": self.cfg.model, "messages": messages, "temperature": self.cfg.temperature}
        if tools:
            body["tools"] = tools
        try:
            r = await self.http.post(f"{self.cfg.base_url.rstrip('/')}/chat/completions", json=body,
                                     headers={"Authorization": f"Bearer {self.cfg.api_key}"})
            r.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMUnavailable(f"model endpoint {self.cfg.base_url}: {exc}") from exc
        msg = r.json()["choices"][0]["message"]
        calls = []
        for i, tc in enumerate(msg.get("tool_calls") or []):
            fn = tc.get("function", {})
            args = fn.get("arguments") or "{}"
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"_raw": args}
            calls.append(ToolCall(tc.get("id") or f"call_{i}", fn.get("name", ""), args))
        return LLMReply(content=msg.get("content") or "", tool_calls=calls)

    async def close(self) -> None:
        await self.http.aclose()
