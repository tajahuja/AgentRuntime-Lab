"""Optional API-backed example; requires OPENAI_API_KEY and OPENAI_MODEL."""

from agent_runtime import AgentRuntime, Document, OpenAICompatibleProvider, RuntimeConfig
from agent_runtime.retriever import TfidfRetriever
from agent_runtime.tools import KnowledgeBaseTool

documents = [Document("runtime", "An agent runtime manages state, tools, retrieval, and model calls.")]
runtime = AgentRuntime(
    OpenAICompatibleProvider(), TfidfRetriever(documents), KnowledgeBaseTool({"runtime": documents[0].text}),
    RuntimeConfig(enable_cache=False),
)
response = runtime.run("What does an agent runtime manage?")
print(response.answer)
print("Usage reported by provider:", response.usage or "not supplied")
