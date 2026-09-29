"""Deterministic tool implementations and a deliberately small routing policy."""

from __future__ import annotations

import ast
import operator as op
import re
from dataclasses import dataclass

from .models import ToolResult

_ALLOWED = {
    ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
    ast.Div: op.truediv, ast.Pow: op.pow, ast.Mod: op.mod,
    ast.USub: op.neg, ast.UAdd: op.pos,
}


def _safe_eval(node: ast.AST) -> int | float:
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        if abs(node.value) > 10**12:
            raise ValueError("number exceeds calculator limit")
        return node.value
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED:
        return _ALLOWED[type(node.op)](_safe_eval(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED:
        left, right = _safe_eval(node.left), _safe_eval(node.right)
        if isinstance(node.op, ast.Pow) and (abs(right) > 12 or abs(left) > 10**6):
            raise ValueError("exponent exceeds calculator limit")
        value = _ALLOWED[type(node.op)](left, right)
        if isinstance(value, (int, float)) and abs(value) > 10**15:
            raise ValueError("result exceeds calculator limit")
        return value
    raise ValueError("unsupported calculator expression")


def calculate(expression: str) -> ToolResult:
    if len(expression) > 120:
        raise ValueError("expression is too long")
    try:
        tree = ast.parse(expression, mode="eval")
        value = _safe_eval(tree.body)
    except (SyntaxError, TypeError, ZeroDivisionError, OverflowError) as error:
        raise ValueError(f"invalid calculator expression: {error}") from error
    return ToolResult("calculator", str(value))


@dataclass
class KnowledgeBaseTool:
    facts: dict[str, str]

    def lookup(self, key: str) -> ToolResult:
        value = self.facts.get(key.lower().strip(), "No fact found.")
        return ToolResult("knowledge_base", value)


class ToolRouter:
    """Route only explicit calculator and exact knowledge-base query patterns."""

    def __init__(self, knowledge_base: KnowledgeBaseTool):
        self.knowledge_base = knowledge_base

    def route(self, query: str) -> list[ToolResult]:
        results = []
        match = re.search(r"\b(?:calculate|compute)\s+(.+?)(?:\?|$)", query, flags=re.I)
        if match:
            results.append(calculate(match.group(1).strip()))
        match = re.search(r"\b(?:what is|define)\s+([a-zA-Z0-9 _-]+?)\??$", query, flags=re.I)
        if match:
            results.append(self.knowledge_base.lookup(match.group(1)))
        return results
