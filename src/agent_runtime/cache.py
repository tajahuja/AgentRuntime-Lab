import hashlib
import json
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ResponseCache:
    _store: dict[str, str] = field(default_factory=dict)

    @staticmethod
    def key(prompt: str, context: list[dict[str, str]], retrieved_ids: list[str]) -> str:
        """Key a stateless response cache by request and retrieved evidence.

        Conversation-state caching is intentionally a separate experiment because
        changing history can invalidate an otherwise identical response.
        """
        payload = json.dumps({
            "prompt": prompt,
            "retrieved_ids": retrieved_ids,
        }, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()

    def get(self, key: str) -> Optional[str]:
        return self._store.get(key)

    def put(self, key: str, value: str) -> None:
        self._store[key] = value
