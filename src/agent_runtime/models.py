from dataclasses import dataclass, field
from typing import Any

@dataclass(frozen=True)
class Document:
    doc_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class RetrievalResult:
    doc_id: str
    text: str
    score: float
    metadata: dict[str, Any]

@dataclass(frozen=True)
class ToolResult:
    name: str
    output: str

@dataclass
class RuntimeState:
    history: list[dict[str, str]] = field(default_factory=list)
    last_retrieval: list[RetrievalResult] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)

@dataclass(frozen=True)
class AgentResponse:
    answer: str
    cache_hit: bool
    retrieval_calls: int
    tool_calls: int
    provider_calls: int
    latency_ms: float
