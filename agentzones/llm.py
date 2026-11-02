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
    raw: dict[str, Any] | None = None


class LLMUnavailable(RuntimeError):
    pass


class LLM:
    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        self.http = httpx.AsyncClient(timeout=cfg.timeout_s)

    @property
    def enabled(self) -> bool:
        return self.cfg.provider != "none"

    async def health(self) -> dict[str, Any]:
        if not self.enabled:
            return {"ok": False, "reason": "provider=none"}
        try:
            r = await self.http.get(f"{self.cfg.base_url.rstrip('/')}/models",
                                    headers={"Authorization": f"Bearer {self.cfg.api_key}"}, timeout=5)
            r.raise_for_status()
            models = [m.get("id") for m in r.json().get("data", [])]
            return {"ok": True, "model": self.cfg.model, "available": models}
        except Exception as exc:
            return {"ok": False, "reason": str(exc)}

    async def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
                   temperature: float | None = None, json_mode: bool = False) -> LLMReply:
        if not self.enabled:
            raise LLMUnavailable("no model configured for this zone")
        body: dict[str, Any] = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": self.cfg.temperature if temperature is None else temperature,
        }
        if tools:
            body["tools"] = tools
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        try:
            r = await self.http.post(f"{self.cfg.base_url.rstrip('/')}/chat/completions", json=body,
                                     headers={"Authorization": f"Bearer {self.cfg.api_key}"})
            r.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMUnavailable(f"model endpoint {self.cfg.base_url}: {exc}") from exc
        data = r.json()
        msg = data["choices"][0]["message"]
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
        return LLMReply(content=msg.get("content") or "", tool_calls=calls, raw=data)

    async def close(self) -> None:
        await self.http.aclose()
