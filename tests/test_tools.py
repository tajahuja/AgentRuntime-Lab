import unittest

from agent_runtime.tools import KnowledgeBaseTool, ToolRouter, calculate


class ToolTests(unittest.TestCase):
    def test_calculator_nested_expression(self):
        self.assertEqual(calculate("12 * (3 + 2)").output, "60")

    def test_calculator_division(self):
        self.assertEqual(calculate("8 / 2").output, "4.0")

    def test_calculator_power(self):
        self.assertEqual(calculate("2 ** 4").output, "16")

    def test_calculator_rejects_code_execution(self):
        with self.assertRaises(ValueError):
            calculate("__import__('os').system('echo bad')")

    def test_calculator_rejects_names(self):
        with self.assertRaises(ValueError):
            calculate("x + 1")

    def test_calculator_rejects_zero_division(self):
        with self.assertRaises(ValueError):
            calculate("1 / 0")

    def test_calculator_limits_exponent(self):
        with self.assertRaises(ValueError):
            calculate("2 ** 100")

    def test_calculator_limits_expression_length(self):
        with self.assertRaises(ValueError):
            calculate("1+" * 80 + "1")

    def test_knowledge_base_lookup_is_case_insensitive(self):
        self.assertEqual(KnowledgeBaseTool({"rag": "fact"}).lookup(" RAG ").output, "fact")

    def test_missing_fact_has_explicit_response(self):
        self.assertEqual(KnowledgeBaseTool({}).lookup("unknown").output, "No fact found.")

    def test_router_detects_calculator(self):
        result = ToolRouter(KnowledgeBaseTool({})).route("calculate 7 * 8")
        self.assertEqual(result[0].output, "56")

    def test_router_detects_definition_lookup(self):
        result = ToolRouter(KnowledgeBaseTool({"rag": "Retrieval and generation"})).route("What is RAG?")
        self.assertEqual(result[0].output, "Retrieval and generation")

    def test_router_does_not_infer_tool_for_unmatched_text(self):
        self.assertEqual(ToolRouter(KnowledgeBaseTool({})).route("Tell me about arithmetic"), [])


if __name__ == "__main__":
    unittest.main()
