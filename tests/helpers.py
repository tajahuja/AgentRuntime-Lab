from agent_runtime.models import Document
from agent_runtime.providers import DeterministicProvider
from agent_runtime.retriever import TfidfRetriever
from agent_runtime.runtime import AgentRuntime, RuntimeConfig
from agent_runtime.tools import KnowledgeBaseTool


def make_runtime(**config):
    docs = [
        Document("agents", "Agents can use tools and maintain state."),
        Document("tabular", "Tabular models can experience distribution shift."),
    ]
    return AgentRuntime(
        DeterministicProvider(), TfidfRetriever(docs),
        KnowledgeBaseTool({"rag": "retrieval and generation", "agent": "A tool-using system"}),
        RuntimeConfig(**config),
    )
