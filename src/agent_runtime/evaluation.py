"""Small deterministic evaluation metrics; these are not model-judge scores."""

from __future__ import annotations

from .models import AgentResponse, EvaluationResult


def evaluate_response(response: AgentResponse, expected: str, gold_document_ids: list[str] | None = None) -> EvaluationResult:
    exact = response.answer.strip().casefold() == expected.strip().casefold()
    completed = expected.casefold() in response.answer.casefold()
    grounded = None if gold_document_ids is None else bool(set(gold_document_ids) & set(response.trace.retrieved_document_ids))
    return EvaluationResult(exact, completed, grounded)
