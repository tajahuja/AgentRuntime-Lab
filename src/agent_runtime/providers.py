"""Generation providers: an offline deterministic provider and optional API client."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass

from .models import Message, RetrievalResult, ToolResult


@dataclass(frozen=True)
class Generation:
    text: str
    usage: dict[str, int]


class ModelProvider(ABC):
    """Provider interface. Implementations should return deterministic identity metadata."""

    @property
    @abstractmethod
    def cache_namespace(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def generate(self, prompt: str, context: list[Message], retrieved: list[RetrievalResult], tools: list[ToolResult]) -> Generation:
        raise NotImplementedError


class DeterministicProvider(ModelProvider):
    """Offline, reproducible fixture provider; its text is not a language-model quality test."""

    @property
    def cache_namespace(self) -> str:
        return "deterministic-v1"

    def generate(self, prompt, context, retrieved, tools) -> Generation:
        parts = [f"Query: {prompt.strip()}"]
        if retrieved:
            parts.append("Evidence: " + " | ".join(r.text for r in retrieved))
        if tools:
            parts.append("Tool results: " + " | ".join(f"{t.name}={t.output}" for t in tools))
        if context:
            parts.append(f"Context turns: {len(context)}")
        return Generation("\n".join(parts), {})


class OpenAICompatibleProvider(ModelProvider):
    """Minimal standard-library client for OpenAI-compatible chat-completions APIs."""

    def __init__(self, api_key: str | None = None, base_url: str | None = None, model: str | None = None, timeout: float = 60):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.model = model or os.getenv("OPENAI_MODEL", "")
        self.timeout = timeout
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAICompatibleProvider")
        if not self.model:
            raise ValueError("OPENAI_MODEL is required for OpenAICompatibleProvider")

    @property
    def cache_namespace(self) -> str:
        return f"openai-compatible:{self.base_url}:{self.model}"

    def generate(self, prompt, context, retrieved, tools) -> Generation:
        messages = [{"role": message.role, "content": message.content} for message in context]
        supporting = []
        if retrieved:
            supporting.append("Retrieved evidence (use only when relevant):\n" + "\n".join(r.text for r in retrieved))
        if tools:
            supporting.append("Deterministic tool results:\n" + "\n".join(f"{t.name}: {t.output}" for t in tools))
        user_content = prompt
        if supporting:
            user_content += "\n\n" + "\n\n".join(supporting)
        messages.append({"role": "user", "content": user_content})
        payload = json.dumps({"model": self.model, "messages": messages, "temperature": 0}).encode()
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=payload,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read())
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            raise RuntimeError(f"LLM provider request failed: {error}") from error
        try:
            text = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise RuntimeError("provider returned an unexpected chat-completions response") from error
        usage = {key: int(value) for key, value in body.get("usage", {}).items() if isinstance(value, (int, float))}
        return Generation(text, usage)
