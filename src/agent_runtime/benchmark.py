import statistics
from .models import Document
from .providers import DeterministicProvider
from .retriever import TfidfRetriever
from .runtime import AgentRuntime, RuntimeConfig
from .tools import KnowledgeBaseTool

DOCS = [
    Document("genai", "Generative AI systems can combine language models with retrieval and tools to answer questions and perform tasks."),
    Document("agents", "AI agents maintain state, select tools, retrieve information, and execute multi-step workflows."),
    Document("cache", "Caching repeated requests can reduce redundant computation and improve latency when context is unchanged."),
    Document("tabular", "Tabular machine learning often faces distribution shift, feature engineering challenges, and changing data distributions."),
]

QUERIES = [
    "How can agents use tools?",
    "Why is caching useful for inference?",
    "What challenges occur in tabular machine learning?",
    "calculate 17 * 23",
    "How can agents use tools?",
    "Why is caching useful for inference?",
]

def run(enable_cache: bool):
    runtime = AgentRuntime(
        DeterministicProvider(),
        TfidfRetriever(DOCS),
        KnowledgeBaseTool({"rag": "Retrieval-Augmented Generation combines retrieval with generation.", "agent": "An agent is a system that can select actions and tools to accomplish a task."}),
        RuntimeConfig(enable_cache=enable_cache),
    )
    responses = [runtime.run(q) for q in QUERIES]
    latencies = [r.latency_ms for r in responses]
    return {
        "cache_enabled": enable_cache,
        "mean_latency_ms": round(statistics.mean(latencies), 3),
        "p50_latency_ms": round(statistics.median(latencies), 3),
        "cache_hits": sum(r.cache_hit for r in responses),
        "provider_calls": sum(r.provider_calls for r in responses),
        "tool_calls": sum(r.tool_calls for r in responses),
    }

if __name__ == "__main__":
    print("Baseline:", run(False))
    print("Cached:", run(True))
