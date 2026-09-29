"""Exact response cache. It deliberately does not implement semantic caching."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

from .models import Message, RetrievalResult


@dataclass
class ResponseCache:
    _store: dict[str, str] = field(default_factory=dict)

    @staticmethod
    def key(
        prompt: str,
        context: list[Message],
        retrieved: list[RetrievalResult],
        provider_namespace: str,
        tool_namespace: str,
    ) -> str:
        """Hash all generation inputs so a hit cannot cross context/provider changes."""
        payload = {
            "prompt": prompt,
            "context": [{"role": m.role, "content": m.content} for m in context],
            "retrieved": [{"id": r.doc_id, "text": r.text} for r in retrieved],
            "provider": provider_namespace,
            "tool_policy": tool_namespace,
        }
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def get(self, key: str) -> str | None:
        return self._store.get(key)

    def put(self, key: str, value: str) -> None:
        self._store[key] = value
