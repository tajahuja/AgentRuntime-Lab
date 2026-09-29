"""Run a small offline example without an API key."""

from agent_runtime import AgentRuntime, DeterministicProvider, Document, RuntimeConfig
from agent_runtime.retriever import TfidfRetriever
from agent_runtime.tools import KnowledgeBaseTool

documents = [
    Document("agents", "AI agents retrieve information, select tools, maintain state, and execute multi-step tasks."),
    Document("rag", "RAG combines information retrieval with generation so a model can use external knowledge."),
    Document("efficiency", "Caching can reduce repeated computation when a request and its relevant context are unchanged."),
]

runtime = AgentRuntime(
    DeterministicProvider(), TfidfRetriever(documents),
    KnowledgeBaseTool({"rag": "Retrieval-Augmented Generation combines retrieval and generation."}),
    RuntimeConfig(enable_cache=True),
)

for query in ["How can AI agents use tools?", "What is RAG?", "calculate 24 * 19", "How can AI agents use tools?"]:
    response = runtime.run(query)
    print(f"\nQUERY: {query}\nCACHE HIT: {response.cache_hit}\nLATENCY (ms): {response.latency_ms:.3f}\n{response.answer}")
