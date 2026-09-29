import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent_runtime.models import Document
from src.agent_runtime.providers import DeterministicProvider
from src.agent_runtime.retriever import TfidfRetriever
from src.agent_runtime.runtime import AgentRuntime
from src.agent_runtime.tools import KnowledgeBaseTool

docs = [
    Document("agents", "AI agents can retrieve information, select tools, maintain state, and execute multi-step tasks."),
    Document("rag", "RAG combines information retrieval with generation so a model can use external knowledge."),
    Document("efficiency", "Caching can reduce repeated computation when a request and its relevant context are unchanged."),
]

runtime = AgentRuntime(
    provider=DeterministicProvider(),
    retriever=TfidfRetriever(docs),
    knowledge_base=KnowledgeBaseTool({"rag": "Retrieval-Augmented Generation combines retrieval and generation."}),
)

for query in ["How can AI agents use tools?", "What is RAG?", "calculate 24 * 19", "How can AI agents use tools?"]:
    response = runtime.run(query)
    print("\nQUERY:", query)
    print("CACHE HIT:", response.cache_hit)
    print("LATENCY (ms):", round(response.latency_ms, 3))
    print(response.answer)
