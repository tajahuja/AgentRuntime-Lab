import re
import time
from dataclasses import dataclass
from .cache import ResponseCache
from .models import AgentResponse, RuntimeState
from .providers import ModelProvider
from .retriever import TfidfRetriever
from .tools import KnowledgeBaseTool, calculate

@dataclass(frozen=True)
class RuntimeConfig:
    top_k: int = 2
    max_history: int = 6
    enable_cache: bool = True

class AgentRuntime:
    def __init__(self, provider: ModelProvider, retriever: TfidfRetriever,
                 knowledge_base: KnowledgeBaseTool, config: RuntimeConfig | None = None):
        self.provider = provider
        self.retriever = retriever
        self.knowledge_base = knowledge_base
        self.config = config or RuntimeConfig()
        self.state = RuntimeState()
        self.cache = ResponseCache()

    def _select_tools(self, query: str):
        tools = []
        m = re.search(r"(?:calculate|compute)\s+(.+?)(?:\?|$)", query, flags=re.I)
        if m:
            try:
                tools.append(calculate(m.group(1).strip()))
            except ValueError:
                pass
        kb = re.search(r"(?:what is|define)\s+([a-zA-Z0-9 _-]+)\??$", query, flags=re.I)
        if kb:
            tools.append(self.knowledge_base.lookup(kb.group(1)))
        return tools

    def run(self, query: str) -> AgentResponse:
        start = time.perf_counter()
        retrieved = self.retriever.search(query, self.config.top_k)
        retrieved_ids = [r.doc_id for r in retrieved]
        context = self.state.history[-self.config.max_history:]
        key = self.cache.key(query, context, retrieved_ids)
        if self.config.enable_cache:
            cached = self.cache.get(key)
            if cached is not None:
                elapsed = (time.perf_counter() - start) * 1000
                return AgentResponse(cached, True, 1, 0, 0, elapsed)

        tools = self._select_tools(query)
        answer = self.provider.generate(query, context, retrieved, tools)
        self.state.last_retrieval = retrieved
        self.state.tool_results = tools
        self.state.history.append({"role": "user", "content": query})
        self.state.history.append({"role": "assistant", "content": answer})
        self.state.history = self.state.history[-self.config.max_history:]
        if self.config.enable_cache:
            self.cache.put(key, answer)
        elapsed = (time.perf_counter() - start) * 1000
        return AgentResponse(answer, False, 1, len(tools), 1, elapsed)
