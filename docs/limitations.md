# Limitations and next steps

## Current limitations

- The included provider is deterministic fixture code, not an LLM. It cannot establish that caching preserves answer quality for generated responses.
- The default retriever is unigram TF-IDF. It is a transparent lexical baseline, not embedding retrieval or a modern RAG system.
- The task set has five hand-authored examples. It is too small to support broad performance, factuality, or generalization claims.
- Timing covers local Python runtime work and excludes model inference, tokenization, network, queueing, and GPU costs.
- Context size is character count rather than token count. The optional provider returns token usage when available, but the deterministic benchmark does not measure tokens.
- Exact caching only reuses identical requests under identical selected context/evidence/tool/provider identity. Semantic caching and cache eviction are absent.
- The cache is in-memory, single-process, and unbounded. It is not a production cache implementation.
- The tool router uses explicit regular-expression patterns and the knowledge-base tool does exact-key matching. It is intentionally limited and may not route paraphrases.
- Experiment D compares rule-based routing with no routing. It does not compare deterministic routing with model-selected function calls.
- **Not executed in the current environment:** an API-backed benchmark, because no API-backed evaluation configuration was set for the recorded run. The optional provider can be smoke-tested by setting `OPENAI_API_KEY` and `OPENAI_MODEL` (and optionally `OPENAI_BASE_URL`) and running `python examples/llm_demo.py`. A comparative LLM quality experiment still needs a reviewed labeled dataset and is not implemented here. Endpoint compatibility varies; keys, quotas, model availability, and network access are user responsibilities.
- This project does not implement model KV caching, continuous batching, asynchronous execution, multi-agent coordination, or custom inference kernels.

## Future experiments

1. Add a larger, versioned evaluation set with multiple phrasings, expected evidence IDs, and carefully reviewed answer keys.
2. Run paired API-backed trials across a named model, report token usage and configuration, and inspect errors and quality regressions rather than relying only on a judge score.
3. Compare deterministic routing with actual provider tool calling using a documented OpenAI-compatible schema.
4. Add token-aware context budgets, then compare bounded history strategies while controlling the prompts sent to the same provider.
5. Test cache invalidation and bounded eviction under document corpus updates and concurrent requests.
6. Measure latency distributions over more repetitions, randomize configuration order, preserve raw per-request samples, and disclose machine/load conditions.
7. Only after a local or hosted inference engine is integrated, study KV cache behavior separately from application response caching.
