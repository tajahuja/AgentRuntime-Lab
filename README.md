# AgentRuntime-Lab

A research-oriented, provider-agnostic experimental framework for studying **efficient agentic AI inference**.

## Research question

> Can explicit context management, deterministic tool routing, retrieval, and response caching reduce repeated computation and latency in multi-step AI agents without degrading answer quality?

## What this repository demonstrates

- A small agent runtime with explicit execution state
- Local TF-IDF retrieval over a document collection
- Deterministic calculator and knowledge-base tools
- Context/state management with bounded history
- Content-addressed response caching
- A provider interface for plugging in an LLM later
- A deterministic local provider so the project is runnable without an API key
- A benchmark harness measuring latency, cache hits, tool calls, and retrieval behavior
- Tests for retrieval, tools, cache correctness, and runtime behavior

## Architecture

```text
                    +----------------+
                    |     Query      |
                    +-------+--------+
                            |
                            v
                  +---------+----------+
                  |    AgentRuntime    |
                  +---------+----------+
                            |
              +-------------+-------------+
              |             |             |
              v             v             v
        +-----------+ +-----------+ +-----------+
        | Retriever | | ToolRouter| |  Context  |
        +-----------+ +-----------+ +-----------+
              |             |             |
              v             v             v
        Local corpus     Calculator     State/cache
              \             |             /
               +------------+------------+
                            |
                            v
                    +---------------+
                    | ModelProvider |
                    +---------------+
                            |
                            v
                         Answer
```

## Quick start

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python examples/demo.py
python -m pytest -q
```

No API key is required for the included deterministic provider.

## Optional LLM integration

The runtime exposes a small `ModelProvider` interface. An external OpenAI-compatible endpoint can be integrated without changing retrieval, tools, state, or benchmarking. API credentials should be supplied through environment variables and never committed to the repository.

## Experiments

Run the benchmark:

```bash
python -m src.agent_runtime.benchmark
```

The benchmark reports:

- mean latency
- p50/p95 latency
- cache hit rate
- retrieval calls
- tool calls
- provider calls
- answer length

The intended research workflow is to compare configurations such as:

1. no response cache vs response cache
2. full history vs bounded context
3. retrieval enabled vs disabled
4. deterministic tool routing vs provider-only answering

## Limitations

This is an experimental research prototype, not a production agent framework. The included provider is deterministic and therefore does not measure the capabilities of a large language model. Results from the benchmark should be interpreted as runtime measurements of the architecture, not as claims about general LLM performance.

## Research direction

The next stage is to connect the runtime to an actual LLM provider and evaluate whether explicit state management and caching reduce repeated work while preserving answer quality. Future experiments can also investigate asynchronous execution, multi-agent concurrency, multimodal state, and KV-cache-aware scheduling.
