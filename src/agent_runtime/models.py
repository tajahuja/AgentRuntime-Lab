"""Typed records shared by the agent runtime and its evaluations."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Message:
    role: str
    content: str


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
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolResult:
    name: str
    output: str


@dataclass
class AgentState:
    history: list[Message] = field(default_factory=list)
    last_retrieval: list[RetrievalResult] = field(default_factory=list)
    last_tool_results: list[ToolResult] = field(default_factory=list)


@dataclass(frozen=True)
class ExecutionTrace:
    request_id: int
    cache_hit: bool
    retrieval_calls: int
    tool_calls: int
    provider_calls: int
    context_characters: int
    retrieved_document_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AgentResponse:
    answer: str
    cache_hit: bool
    retrieval_calls: int
    tool_calls: int
    provider_calls: int
    latency_ms: float
    context_characters: int
    trace: ExecutionTrace
    usage: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationResult:
    exact_match: bool
    task_completed: bool
    retrieved_gold: bool | None = None
