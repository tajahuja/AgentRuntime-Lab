import unittest

from agent_runtime.models import Document
from agent_runtime.retriever import TfidfRetriever


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.retriever = TfidfRetriever([
            Document("a", "Agents can use tools and maintain state."),
            Document("b", "Tabular models can experience distribution shift."),
        ])

    def test_retrieval_returns_relevant_document(self):
        self.assertEqual(self.retriever.search("distribution shift", 1)[0].doc_id, "b")

    def test_query_without_overlap_returns_no_results(self):
        self.assertEqual(self.retriever.search("ocean astronomy"), [])

    def test_blank_query_returns_no_results(self):
        self.assertEqual(self.retriever.search(" "), [])

    def test_top_k_limits_results(self):
        self.assertEqual(len(self.retriever.search("agents tools distribution", 1)), 1)

    def test_zero_top_k_returns_empty(self):
        self.assertEqual(self.retriever.search("agents", 0), [])

    def test_negative_top_k_rejected(self):
        with self.assertRaises(ValueError):
            self.retriever.search("agents", -1)

    def test_empty_corpus_rejected(self):
        with self.assertRaises(ValueError):
            TfidfRetriever([])

    def test_duplicate_document_ids_rejected(self):
        with self.assertRaises(ValueError):
            TfidfRetriever([Document("same", "one"), Document("same", "two")])


if __name__ == "__main__":
    unittest.main()
