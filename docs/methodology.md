# Experimental methodology

## Research question and hypothesis

**Question:** Can explicit context management, deterministic tool routing, retrieval, and exact response caching reduce repeated computation and latency in multi-step agents without degrading answer quality?

**Hypothesis:** Exact caching will reduce provider calls on repeated identical workflows. Bounded history will reduce the selected prior-message character count relative to full history. Retrieval and deterministic tools will improve completion on tasks whose expected answer is present in the corpus or calculator result.

These are hypotheses. Outcomes must be read from `results/benchmark_results.json`; no effect should be generalized beyond this deterministic task set.

## Task set and controls

`benchmarks/tasks.json` contains five fixed requests spanning retrieval, a knowledge-base lookup, arithmetic, caching facts, and tabular ML. The same ordered task list is replayed for each repetition. Conversation state is reset between workflows while the response cache persists. Each configuration uses the same provider, corpus, tool facts, and task order. A response can be a cache hit only when the exact query, selected history, retrieved evidence, provider identity, and versioned tool-policy identity match. The exact tool output is not part of the key because routing is skipped on a hit; tool behavior and knowledge-base facts are covered by the tool-policy identity.

The deterministic provider is intentionally simple. The score `task_completion_rate` means only that a required substring appears in its output. `retrieval_gold_hit_rate` is document-ID recall over tasks with a specified gold ID. `exact_match_rate` is included but is not a useful measure for the current free-form deterministic outputs; it is expected to be low. No LLM-as-judge is used.

## Experiments

| Experiment | Configurations | Primary measurements |
|---|---|---|
| A: exact response cache | Off / on | p50, p95, mean latency; provider calls; cache hit rate; task completion |
| B: conversation history | Full / last four messages | selected history characters; latency; task completion |
| C: retrieval | Off / on | retrieval calls; gold-document hit rate; task completion |
| D: deterministic routing | Off / on | tool calls; task completion |
| E: repeated workflow | Same five requests repeated five times by default | repeated answers, calls, latency distribution, hit rate |

For A, E is also the cache workload. The no-cache and cached configurations run the same replay. The cache remains warm across repetitions, but each workflow begins from an empty conversation. For B, this is a context-size experiment using a deterministic fixture; it is not an LLM token-cost or answer-quality study. For D, “routing off” means the provider receives no tool results. There is no provider-driven tool selection baseline in this version.

## Metrics and interpretation

- Latencies use `time.perf_counter()` around each runtime call and are reported in milliseconds. p95 uses nearest rank.
- Provider, retrieval, and tool call counts are instrumented from runtime execution traces.
- Context size is the character count of selected earlier messages; it is not tokenizer token count and excludes the current query and retrieved evidence.
- Cache hits skip deterministic tool execution and generation after retrieval has produced cache-key inputs. Retrieval costs remain in the measured path; tool behavior is scoped through a versioned policy identity.
- Task completion, exact match, and retrieval hit checks are deterministic and tied to the fields in the checked-in JSON task set.

Wall-clock timing can vary with CPU load, interpreter, and hardware. The benchmark reports raw environment metadata and results so others can rerun it. Do not describe timing differences as statistically significant from this small, unrandomized run.

## Reproduction

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
python -m pytest -q
python -m agent_runtime.benchmark --repetitions 5
```

The runtime and benchmark use only the Python standard library. Pytest is a development dependency.
