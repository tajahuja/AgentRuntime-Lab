from abc import ABC, abstractmethod
from .models import RetrievalResult, ToolResult

class ModelProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str, context: list[dict[str, str]],
                 retrieved: list[RetrievalResult], tools: list[ToolResult]) -> str:
        raise NotImplementedError

class DeterministicProvider(ModelProvider):
    """Offline provider for reproducible runtime experiments."""

    def generate(self, prompt, context, retrieved, tools):
        parts = [f"Query: {prompt.strip()}"]
        if retrieved:
            parts.append("Evidence: " + " | ".join(r.text for r in retrieved[:2]))
        if tools:
            parts.append("Tool results: " + " | ".join(f"{t.name}={t.output}" for t in tools))
        if context:
            parts.append(f"Context turns: {len(context)}")
        return "\n".join(parts)
