# Benchmark results

The machine-readable run is in [`benchmark_results.json`](benchmark_results.json). It was generated on 2026-09-29 with Python 3.12.14 on Linux using the deterministic provider and five replays of the five-task workload (25 calls per configuration).

| Comparison | Main observed measurements |
|---|---|
| Cache off / on | Provider calls: 25 / 5; exact cache hits: 0 / 20 (80%); tool calls: 10 / 2; task completion: 100% / 100%. Mean latency: 0.0428 / 0.0324 ms; p95: 0.0637 / 0.0414 ms. |
| Full / bounded history | Mean selected-history size: 419.4 / 263.2 characters. Task completion: 100% in both. |
| Retrieval off / on | Task completion: 40% / 100%; gold-document hit rate: 0% / 75%. |
| Tool routing off / on | Task completion: 60% / 100%; tool calls: 0 / 10. |

These outcomes describe only the small deterministic fixture workload. The cache reduces provider calls because identical workflows are replayed with history reset while the cache remains warm. Retrieval/tool gains correspond to simple expected-substring checks, not a general answer-quality measure. Latencies are sub-millisecond local Python measurements, fluctuate with system conditions, and exclude LLM inference. Treat them as a reproducibility record, not a performance claim.

Reproduce with:

```bash
python -m agent_runtime.benchmark --repetitions 5
```

This overwrites the raw JSON with a new timestamp and environment record.
