import unittest
from unittest.mock import patch

from agent_runtime.models import Message
from tests.helpers import make_runtime


class RuntimeTests(unittest.TestCase):
    def test_empty_query_rejected(self):
        with self.assertRaises(ValueError):
            make_runtime().run("  ")

    def test_cache_reuses_identical_generation(self):
        runtime = make_runtime()
        first = runtime.run("How can agents use tools?")
        runtime.clear_history()
        second = runtime.run("How can agents use tools?")
        self.assertFalse(first.cache_hit)
        self.assertTrue(second.cache_hit)
        self.assertEqual(second.provider_calls, 0)

    def test_cache_hit_updates_conversation_history(self):
        runtime = make_runtime()
        runtime.run("How can agents use tools?")
        runtime.clear_history()
        runtime.run("How can agents use tools?")
        self.assertEqual([m.role for m in runtime.state.history], ["user", "assistant"])

    def test_cache_hit_skips_tool_router(self):
        runtime = make_runtime()
        runtime.run("calculate 7 * 8")
        runtime.clear_history()
        with patch("agent_runtime.runtime.ToolRouter.route", side_effect=AssertionError("router should be skipped")):
            response = runtime.run("calculate 7 * 8")
        self.assertTrue(response.cache_hit)
        self.assertEqual(response.tool_calls, 0)

    def test_history_invalidation_prevents_stale_cached_response(self):
        runtime = make_runtime()
        runtime.run("How can agents use tools?")
        second = runtime.run("How can agents use tools?")
        self.assertFalse(second.cache_hit)
        self.assertIn("Context turns: 2", second.answer)

    def test_knowledge_base_change_invalidates_cached_response(self):
        runtime = make_runtime()
        first = runtime.run("What is RAG?")
        runtime.clear_history()
        runtime.knowledge_base.facts["rag"] = "Updated definition"
        second = runtime.run("What is RAG?")
        self.assertFalse(second.cache_hit)
        self.assertIn("Updated definition", second.answer)

    def test_history_is_bounded_by_message_count(self):
        runtime = make_runtime(max_history_messages=3, enable_cache=False)
        for query in ["first", "second", "third"]:
            runtime.run(query)
        self.assertEqual(len(runtime.state.history), 3)

    def test_zero_history_disables_context(self):
        runtime = make_runtime(max_history_messages=0, enable_cache=False)
        response = runtime.run("hello")
        self.assertEqual(response.context_characters, 0)
        self.assertEqual(runtime.state.history, [])

    def test_full_history_option(self):
        runtime = make_runtime(max_history_messages=None, enable_cache=False)
        runtime.run("first")
        response = runtime.run("second")
        self.assertEqual(response.context_characters, len("first") + len("Query: first"))

    def test_runtime_trace_has_monotonic_request_id(self):
        runtime = make_runtime()
        self.assertEqual(runtime.run("first").trace.request_id, 1)
        self.assertEqual(runtime.run("second").trace.request_id, 2)

    def test_tool_routing_disabled(self):
        response = make_runtime(enable_tool_routing=False).run("calculate 7 * 8")
        self.assertEqual(response.tool_calls, 0)

    def test_retrieval_disabled(self):
        response = make_runtime(enable_retrieval=False).run("distribution shift")
        self.assertEqual(response.retrieval_calls, 0)
        self.assertEqual(response.trace.retrieved_document_ids, ())

    def test_invalid_config_is_rejected(self):
        with self.assertRaises(ValueError):
            make_runtime(top_k=-1)


if __name__ == "__main__":
    unittest.main()
