# Research notes

## Why runtime architecture matters

An agent request can involve several pieces of work: assembling conversation state, retrieving evidence, selecting and invoking tools, calling a model, and recording the result. Making those steps explicit makes it possible to inspect behavior and measure where repeated work occurs. A runtime also defines which state is visible to a generation call, which affects both correctness and cacheability.

Repeated work can appear when workflows revisit the same request, re-run tool selection, or send a long conversation prefix to a provider on every turn. Response caching can skip generation for an exact repeated input. Bounded history limits the amount of earlier dialogue selected for the next call. Retrieval can supply external evidence that is absent from the prompt. These mechanisms solve different problems and may affect answer quality differently.

Latency matters for interactive systems, but a local deterministic-provider benchmark is not a substitute for measuring a real model-serving stack. The local measurements include Python orchestration and retrieval overhead and do not include tokenization, GPU execution, network round trips, model queueing, or generated-token time.

## Cache terminology

- **Application-level exact response caching** stores a completed answer under a key made from the request and all generation inputs. This repository uses this form. It can avoid a provider call for exact repeated inputs, but retrieval and deterministic routing still run to construct the key inputs.
- **Semantic caching** attempts to reuse an answer for a similar, non-identical query. That introduces similarity thresholds and a risk of returning an answer for the wrong intent. It is not implemented here.
- **Transformer KV caching** stores attention keys and values created during model inference so a model can reuse prior token computation. It is an inference-engine/model-state mechanism, not a completed-response cache. This prototype does not access or manage a model KV cache.
- **Model inference optimization** includes batching, kernels, quantization, speculative decoding, and scheduling. None is implemented or measured here.

## Connection to the Yandex Research article

The Yandex Research article [“The KV cache as an agent runtime”](https://research.yandex.com/blog/the-kv-cache-as-an-agent-runtime) discusses treating model KV state as active execution state. It describes shared cache blocks, different attention views, and runtime scheduling that can support concurrent reasoning, communication, observations, and tool interactions. Its scope is model-serving and inference-runtime mechanisms, including custom kernels and SGLang work.

AgentRuntime-Lab is related at a much smaller and higher application layer: it makes conversation history, retrieval, tool routing, generation, and completed-response caching explicit and inspectable. It does not implement shared KV blocks, attention views, asynchronous streams, kernel changes, or any of the article’s reported results. This is an independent educational research prototype, not a reproduction of that work.

## Evaluation boundary

The included deterministic provider provides predictable output for tests. The evaluation checks fixed expected substrings, exact string matches, and whether a designated document ID appeared in retrieval output. These checks assess plumbing on a small controlled task set. They do not assess open-ended reasoning, factuality in general, robustness, or parity with a real model. The optional provider can report token usage only when the endpoint returns it; API-backed quality evaluation is not included in the checked-in measurements.

## Reading

- Yakushev, G. (2026). [The KV cache as an agent runtime](https://research.yandex.com/blog/the-kv-cache-as-an-agent-runtime). Yandex Research.
- Lewis et al. (2020). [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401). NeurIPS.
- Yao et al. (2023). [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629). ICLR.
