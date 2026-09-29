from src.agent_runtime.models import Document
from src.agent_runtime.providers import DeterministicProvider
from src.agent_runtime.retriever import TfidfRetriever
from src.agent_runtime.runtime import AgentRuntime
from src.agent_runtime.tools import KnowledgeBaseTool, calculate


def make_runtime():
    docs = [
        Document("a", "Agents can use tools and maintain state."),
        Document("b", "Tabular models can experience distribution shift."),
    ]
    return AgentRuntime(DeterministicProvider(), TfidfRetriever(docs), KnowledgeBaseTool({"rag": "retrieval and generation"}))


def test_retrieval_returns_relevant_document():
    result = make_runtime().retriever.search("distribution shift", top_k=1)
    assert result[0].doc_id == "b"


def test_calculator_is_safe_and_correct():
    assert calculate("12 * (3 + 2)").output == "60"


def test_calculator_rejects_unsafe_expression():
    try:
        calculate("__import__('os').system('echo bad')")
    except ValueError:
        assert True
    else:
        assert False


def test_runtime_cache_hits_repeated_request():
    runtime = make_runtime()
    first = runtime.run("How can agents use tools?")
    second = runtime.run("How can agents use tools?")
    assert first.cache_hit is False
    assert second.cache_hit is True
    assert second.provider_calls == 0


def test_tool_routing():
    response = make_runtime().run("calculate 7 * 8")
    assert response.tool_calls == 1
    assert "56" in response.answer
