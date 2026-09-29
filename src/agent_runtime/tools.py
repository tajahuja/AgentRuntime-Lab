import ast
import operator as op
from dataclasses import dataclass
from .models import ToolResult

_ALLOWED = {
    ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
    ast.Div: op.truediv, ast.Pow: op.pow, ast.Mod: op.mod,
    ast.USub: op.neg,
}


def _safe_eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED:
        return _ALLOWED[type(node.op)](_safe_eval(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED:
        return _ALLOWED[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    raise ValueError("Unsupported expression")


def calculate(expression: str) -> ToolResult:
    tree = ast.parse(expression, mode="eval")
    value = _safe_eval(tree.body)
    return ToolResult("calculator", str(value))

@dataclass
class KnowledgeBaseTool:
    facts: dict[str, str]

    def lookup(self, key: str) -> ToolResult:
        value = self.facts.get(key.lower().strip(), "No fact found.")
        return ToolResult("knowledge_base", value)
