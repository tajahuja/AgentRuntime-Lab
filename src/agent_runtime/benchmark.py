"""Reproducible deterministic runtime experiments for the included task set."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .evaluation import evaluate_response
from .models import Document
from .providers import DeterministicProvider
from .retriever import TfidfRetriever
from .runtime import AgentRuntime, RuntimeConfig
from .tools import KnowledgeBaseTool

ROOT = Path(__file__).resolve().parents[2]
TASKS_PATH = ROOT / "benchmarks" / "tasks.json"
RESULTS_PATH = ROOT / "results" / "benchmark_results.json"
DOCUMENTS = [
    Document("agents", "AI agents maintain state, select tools, retrieve information, and execute multi-step workflows."),
    Document("rag", "Retrieval-Augmented Generation combines retrieval of external documents with language model generation."),
    Document("cache", "Caching repeated requests can reduce redundant computation and improve latency when context is unchanged."),
    Document("tabular", "Tabular machine learning often faces distribution shift, feature engineering challenges, and changing data distributions."),
]
FACTS = {
    "rag": "Retrieval-Augmented Generation combines retrieval and generation.",
    "agent": "An agent can select actions and tools to accomplish a task.",
}


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = max(0, min(len(ordered) - 1, int((percentile / 100) * len(ordered) + 0.999999) - 1))
    return ordered[index]


def _runtime(*, cache: bool, retrieval: bool = True, tools: bool = True, history: int | None = 6) -> AgentRuntime:
    return AgentRuntime(
        DeterministicProvider(), TfidfRetriever(DOCUMENTS), KnowledgeBaseTool(FACTS),
        RuntimeConfig(enable_cache=cache, enable_retrieval=retrieval,
                      enable_tool_routing=tools, max_history_messages=history),
    )


def _run_workloads(tasks: list[dict[str, Any]], repetitions: int, **options: Any) -> dict[str, Any]:
    runtime = _runtime(**options)
    latencies: list[float] = []
    context_sizes: list[int] = []
    responses = []
    for _ in range(repetitions):
        runtime.clear_history()
        for task in tasks:
            response = runtime.run(task["query"])
            responses.append((task, response))
            latencies.append(response.latency_ms)
            context_sizes.append(response.context_characters)
    evaluations = [evaluate_response(response, task["expected_contains"], task.get("gold_document_ids"))
                   for task, response in responses]
    n = max(1, len(responses))
    grounding_checks = [result.retrieved_gold for result in evaluations if result.retrieved_gold is not None]
    return {
        "requests": len(responses),
        "mean_latency_ms": round(statistics.fmean(latencies), 6) if latencies else 0,
        "p50_latency_ms": round(statistics.median(latencies), 6) if latencies else 0,
        "p95_latency_ms": round(_percentile(latencies, 95), 6),
        "mean_context_characters": round(statistics.fmean(context_sizes), 3) if context_sizes else 0,
        "cache_hits": sum(response.cache_hit for _, response in responses),
        "cache_hit_rate": round(sum(response.cache_hit for _, response in responses) / n, 6),
        "provider_calls": sum(response.provider_calls for _, response in responses),
        "retrieval_calls": sum(response.retrieval_calls for _, response in responses),
        "tool_calls": sum(response.tool_calls for _, response in responses),
        "task_completion_rate": round(sum(result.task_completed for result in evaluations) / n, 6),
        "exact_match_rate": round(sum(result.exact_match for result in evaluations) / n, 6),
        "retrieval_gold_hit_rate": round(sum(bool(value) for value in grounding_checks) / len(grounding_checks), 6) if grounding_checks else None,
        "answers_identical_within_workflows": _answers_repeat_consistently(responses, len(tasks)),
    }


def _answers_repeat_consistently(responses, workflow_size: int) -> bool:
    if workflow_size == 0 or len(responses) <= workflow_size:
        return True
    reference = [response.answer for _, response in responses[:workflow_size]]
    return all([response.answer for _, response in responses[start:start + workflow_size]] == reference
               for start in range(workflow_size, len(responses), workflow_size))


def run_benchmarks(tasks_path: Path = TASKS_PATH, repetitions: int = 5) -> dict[str, Any]:
    if repetitions < 1:
        raise ValueError("repetitions must be at least 1")
    tasks = json.loads(tasks_path.read_text(encoding="utf-8"))
    workload = tasks["tasks"]
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "method": {
            "provider": "DeterministicProvider",
            "repetitions": repetitions,
            "workflow": "The same ordered task set is repeated; conversation history resets between workflows while the response cache persists.",
            "timing_note": "Wall-clock measurements include local retrieval, routing, and Python overhead. They are environment-specific and not LLM latency estimates.",
            "percentile_method": "nearest-rank p95",
        },
        "task_set": {"path": str(tasks_path.relative_to(ROOT)) if tasks_path.is_relative_to(ROOT) else str(tasks_path), "count": len(workload), "ids": [task["id"] for task in workload]},
        "experiments": {
            "A_cache_off": _run_workloads(workload, repetitions, cache=False),
            "A_cache_on": _run_workloads(workload, repetitions, cache=True),
            "B_full_history": _run_workloads(workload, repetitions, cache=False, history=None),
            "B_bounded_history": _run_workloads(workload, repetitions, cache=False, history=4),
            "C_retrieval_off": _run_workloads(workload, repetitions, cache=False, retrieval=False),
            "C_retrieval_on": _run_workloads(workload, repetitions, cache=False, retrieval=True),
            "D_tools_off": _run_workloads(workload, repetitions, cache=False, tools=False),
            "D_deterministic_routing": _run_workloads(workload, repetitions, cache=False, tools=True),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, default=TASKS_PATH)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--output", type=Path, default=RESULTS_PATH)
    args = parser.parse_args()
    report = run_benchmarks(args.tasks, args.repetitions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
