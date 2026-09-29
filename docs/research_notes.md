# Research Notes

## Hypothesis
Explicit runtime state and response caching can reduce repeated provider work and improve latency for repeated or context-equivalent agent requests.

## Variables
- Cache: enabled / disabled
- Context window: bounded / full
- Retrieval: enabled / disabled
- Provider: deterministic baseline / external LLM

## Metrics
- Mean latency
- p50 / p95 latency
- Provider calls
- Tool calls
- Cache hit rate
- Retrieval calls
- Answer quality (to be added with an LLM-based evaluation set)

## Important limitation
The current deterministic provider is deliberately simple. It makes the runtime reproducible, but it does not establish claims about the quality or efficiency of large language models. A future experiment should connect the same runtime to an actual LLM and add a labeled evaluation set.
