import unittest

from agent_runtime.cache import ResponseCache
from agent_runtime.models import Message, RetrievalResult


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.context = [Message("user", "hello")]
        self.retrieved = [RetrievalResult("doc", "some text", 0.5)]

    def key(self, **changes):
        values = dict(prompt="query", context=self.context, retrieved=self.retrieved,
                      provider_namespace="provider-v1", tool_namespace="router-v1")
        values.update(changes)
        return ResponseCache.key(**values)

    def test_identical_inputs_have_identical_key(self):
        self.assertEqual(self.key(), self.key())

    def test_context_is_part_of_key(self):
        self.assertNotEqual(self.key(), self.key(context=[Message("user", "different")]))

    def test_retrieved_text_is_part_of_key(self):
        self.assertNotEqual(self.key(), self.key(retrieved=[RetrievalResult("doc", "changed", 0.5)]))

    def test_tool_policy_identity_is_part_of_key(self):
        self.assertNotEqual(self.key(), self.key(tool_namespace="router-v2"))

    def test_provider_identity_is_part_of_key(self):
        self.assertNotEqual(self.key(), self.key(provider_namespace="provider-v2"))

    def test_cache_get_and_put(self):
        cache = ResponseCache()
        key = self.key()
        self.assertIsNone(cache.get(key))
        cache.put(key, "answer")
        self.assertEqual(cache.get(key), "answer")


if __name__ == "__main__":
    unittest.main()
