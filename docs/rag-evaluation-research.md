# RAG Evaluation Framework Research

> **Status:** Research complete. **Integrated as Story 2.8** in [Epic 2](epic-2-core-rag-query-system.md).
>
> **Recommendation:** Langfuse (tracing/prompts) + Arize Phoenix (evaluation) - both with FREE UI dashboards

## Requirements

- Evaluate retrieval quality (precision, recall, relevancy)
- Evaluate LLM response quality (faithfulness, answer relevancy)
- Trace and log pipeline steps
- Compare different prompts/approaches
- **Must be free or have generous free tier with UI dashboard**

---

## Framework Comparison Summary

| Framework | License | Free UI | Best For |
|-----------|---------|---------|----------|
| **Langfuse** | MIT | Self-host: unlimited; Cloud: 1M spans/mo | Tracing, prompt management |
| **Arize Phoenix** | ELv2 | Self-host: unlimited; Cloud: free tier | Evaluation with visualization |
| DeepEval | MIT | CLI only (Confident AI dashboard is paid) | pytest-native evaluation |
| RAGAS | Apache 2.0 | No UI | Lightweight RAG metrics |
| LangSmith | Proprietary | 5k traces/mo only | Not recommended (limited) |

---

## Recommended Stack

```
┌─────────────────────────────────────────────────────┐
│            TRACING & PROMPT MANAGEMENT              │
│  Langfuse (self-hosted - full UI free)              │
│  - LLM call tracing with costs                      │
│  - Prompt versioning & A/B comparison               │
│  - Session tracking                                 │
│  - Customizable dashboards                          │
└─────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│            EVALUATION & EXPERIMENTS                 │
│  Arize Phoenix (self-hosted - full UI free)         │
│  - RAG evaluation with UI visualization             │
│  - Hallucination detection                          │
│  - Context relevancy scoring                        │
│  - Experiment tracking & comparison                 │
└─────────────────────────────────────────────────────┘
```

### Why This Combination?

1. **Langfuse** - Best-in-class prompt management and tracing with customizable dashboards
2. **Phoenix** - Built-in evaluation UI with pre-built templates (no paid tier needed)
3. Both have full UI dashboards when self-hosted (no cost)
4. Both support OpenTelemetry (can share trace data)
5. Separation of concerns: Langfuse for ops/prompts, Phoenix for quality evaluation

---

## Implementation Plan

### Dependencies

```toml
# pyproject.toml
[tool.poetry.dependencies]
langfuse = "^2.36"              # Tracing & prompt management
arize-phoenix = "^4.0"          # Evaluation with UI dashboard
openinference-instrumentation-openai = "^1.0"  # OpenTelemetry for OpenAI
```

### New Modules

```
src/observability/           # Langfuse integration
├── __init__.py
├── tracer.py               # Singleton tracer (graceful degradation)
└── decorators.py           # @trace_span decorator

src/evaluation/             # Phoenix integration
├── __init__.py
├── phoenix_client.py       # Phoenix client singleton
├── evaluators.py           # RAG evaluation functions
└── experiments.py          # A/B testing utilities
```

### Configuration (.env.example)

```bash
# Langfuse (self-host or cloud free tier)
LANGFUSE_ENABLED=false
LANGFUSE_PUBLIC_KEY=pk-lf-xxx
LANGFUSE_SECRET_KEY=sk-lf-xxx
LANGFUSE_HOST=http://localhost:3000

# Phoenix (self-hosted, runs locally)
PHOENIX_ENABLED=false
PHOENIX_PORT=6006
PHOENIX_COLLECTOR_ENDPOINT=http://localhost:6006

# Evaluation thresholds
EVAL_HALLUCINATION_THRESHOLD=0.3
EVAL_RELEVANCE_THRESHOLD=0.7
```

---

## Code Examples

### Langfuse Tracing

```python
# src/observability/tracer.py
from langfuse.decorators import observe, langfuse_context

@observe()
async def query_pipeline(query: str) -> str:
    with langfuse_context.update_current_observation(
        metadata={"query_type": "lore_question"}
    ):
        chunks = await retrieve(query)
        response = await generate(query, chunks)
    return response
```

### Phoenix Evaluation

```python
# src/evaluation/evaluators.py
import phoenix as px
from phoenix.evals import HallucinationEvaluator, RelevanceEvaluator

class RAGEvaluator:
    def __init__(self, model: str = "gpt-3.5-turbo"):
        self.hallucination_eval = HallucinationEvaluator(model=model)
        self.relevance_eval = RelevanceEvaluator(model=model)

    def evaluate_response(self, query: str, response: str, context: list[str]) -> dict:
        return {
            "hallucination": self.hallucination_eval(query, response, context),
            "relevance": self.relevance_eval(query, response),
        }
```

---

## Files to Create

| File | Purpose |
|------|---------|
| `src/observability/__init__.py` | Module exports |
| `src/observability/tracer.py` | Langfuse singleton tracer |
| `src/observability/decorators.py` | @trace_span decorator |
| `src/evaluation/__init__.py` | Evaluation module exports |
| `src/evaluation/phoenix_client.py` | Phoenix client singleton |
| `src/evaluation/evaluators.py` | RAG evaluation functions |
| `src/utils/eval_config.py` | Config dataclass |
| `tests/evaluation/conftest.py` | Evaluation fixtures |
| `tests/evaluation/test_retrieval_quality.py` | Retrieval tests |
| `tests/evaluation/test_generation_quality.py` | Generation tests |
| `tests/evaluation/datasets/sample_questions.json` | Golden Q&A pairs |

---

## Running Evaluation

```bash
# Launch Phoenix UI locally
poetry run python -c "import phoenix as px; px.launch_app()"
# Visit http://localhost:6006

# Run evaluation tests
poetry run pytest -m evaluation
```

---

## Sources

- [Langfuse GitHub](https://github.com/langfuse/langfuse) | [Docs](https://langfuse.com/docs)
- [Arize Phoenix GitHub](https://github.com/Arize-ai/phoenix) | [Docs](https://arize.com/docs/phoenix)
- [DeepEval](https://github.com/confident-ai/deepeval)
- [RAGAS](https://docs.ragas.io/en/stable/)
