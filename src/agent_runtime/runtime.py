"""Orchestration for context, retrieval, deterministic tools, generation, and cache."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass

from .cache import ResponseCache
from .models import AgentResponse, AgentState, ExecutionTrace, Message
from .providers import ModelProvider
from .retriever import TfidfRetriever
from .tools import KnowledgeBaseTool, ToolRouter


@dataclass(frozen=True)
class RuntimeConfig:
    top_k: int = 2
    max_history_messages: int | None = 6
    enable_cache: bool = True
    enable_retrieval: bool = True
    enable_tool_routing: bool = True

    def __post_init__(self):
        if self.top_k < 0:
            raise ValueError("top_k must be non-negative")
        if self.max_history_messages is not None and self.max_history_messages < 0:
            raise ValueError("max_history_messages must be non-negative or None")


class AgentRuntime:
    """A single-conversation runtime with inspectable state and per-call traces."""

    def __init__(self, provider: ModelProvider, retriever: TfidfRetriever | None,
                 knowledge_base: KnowledgeBaseTool, config: RuntimeConfig | None = None):
        self.provider = provider
        self.retriever = retriever
        self.knowledge_base = knowledge_base
        self.config = config or RuntimeConfig()
        self.state = AgentState()
        self.cache = ResponseCache()
        self._request_id = 0
        self.tool_namespace = "router-v1"

    def _context(self) -> list[Message]:
        limit = self.config.max_history_messages
        return list(self.state.history if limit is None else self.state.history[-limit:] if limit else [])

    def clear_history(self) -> None:
        """Start a fresh conversation while retaining exact-response cache entries."""
        self.state = AgentState()

    def run(self, query: str) -> AgentResponse:
        if not query.strip():
            raise ValueError("query must not be empty")
        start = time.perf_counter()
        self._request_id += 1
        context = self._context()
        retrieved = self.retriever.search(query, self.config.top_k) if self.config.enable_retrieval and self.retriever else []
        facts_digest = hashlib.sha256(json.dumps(self.knowledge_base.facts, sort_keys=True).encode()).hexdigest()
        tool_namespace = self.tool_namespace + ":" + facts_digest + (":enabled" if self.config.enable_tool_routing else ":disabled")
        key = ResponseCache.key(query, context, retrieved, self.provider.cache_namespace, tool_namespace)
        cached = self.cache.get(key) if self.config.enable_cache else None
        hit = cached is not None
        if hit:
            answer, usage = cached, {}
            tools = []
            retrieval_calls, tool_calls, provider_calls = int(bool(self.retriever and self.config.enable_retrieval)), 0, 0
        else:
            tools = ToolRouter(self.knowledge_base).route(query) if self.config.enable_tool_routing else []
            generation = self.provider.generate(query, context, retrieved, tools)
            answer, usage = generation.text, generation.usage
            retrieval_calls = int(bool(self.retriever and self.config.enable_retrieval))
            tool_calls, provider_calls = len(tools), 1
            if self.config.enable_cache:
                self.cache.put(key, answer)

        self.state.last_retrieval = retrieved
        self.state.last_tool_results = tools if not hit else []
        self.state.history.extend([Message("user", query), Message("assistant", answer)])
        if self.config.max_history_messages is not None:
            self.state.history = self.state.history[-self.config.max_history_messages:] if self.config.max_history_messages else []

        context_characters = sum(len(message.content) for message in context)
        elapsed = (time.perf_counter() - start) * 1000
        trace = ExecutionTrace(
            request_id=self._request_id, cache_hit=hit,
            retrieval_calls=retrieval_calls, tool_calls=tool_calls,
            provider_calls=provider_calls, context_characters=context_characters,
            retrieved_document_ids=tuple(result.doc_id for result in retrieved),
        )
        return AgentResponse(answer, hit, retrieval_calls, tool_calls, provider_calls, elapsed, context_characters, trace, usage)
