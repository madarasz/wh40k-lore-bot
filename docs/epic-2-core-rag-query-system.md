# Epic 2: Core RAG Query System

**Epic Goal:** Build the RAG query engine with hybrid retrieval (vector + BM25), metadata filtering, context expansion, and LLM-powered response generation with **structured JSON output (Pydantic validation)**, personality modes, and smalltalk handling, supporting both OpenAI and Anthropic providers, delivering a complete CLI-testable lore query system.

**Value Proposition:** Delivers the core lore query functionality with engaging personality and intelligent smalltalk detection that can be tested via CLI, validating the RAG pipeline before Discord integration complexity. This epic transforms the data from Epic 1 into an intelligent question-answering system with production-ready structured output validation.

**Key Features:**
- Structured LLM output: `answer`, `personality_reply`, `sources` (wiki URLs), `smalltalk` flag
- Server-side Pydantic validation (OpenAI `beta.chat.completions.parse`, Anthropic `beta.messages.parse`)
- Client-side fallback validation for compatibility
- Configurable personality modes (default, grimdark_narrator)
- Natural conversation flow with smalltalk detection
- Multi-LLM provider support (OpenAI, Anthropic)

**Dependencies:**
- Epic 1 must be complete (vector database populated, BM25 index built, markdown archives ready)
- Existing repositories: `VectorRepository`, `BM25Repository`
- Existing infrastructure: SQLite, Alembic, structlog
- **NEW:** Pydantic ^2.0 for structured output validation

**Estimated Effort:** 20.5 hours (8 stories, increased from 17.5 hours due to structured output implementation)

---

## Story 2.1: Hybrid Retrieval Service Implementation

**Story:** As a developer, I want a hybrid retrieval service that combines vector similarity search and BM25 keyword matching so that I can retrieve the most relevant chunks for both conceptual and exact-match queries.

**Acceptance Criteria:**
1. `HybridRetrievalService` class created in `src/rag/hybrid_retrieval.py`
2. Vector search integration with `VectorRepository`:
   - Accepts query embedding (1536-dim vector)
   - Returns top-k chunks with similarity scores
   - Supports async operations
3. BM25 search integration with `BM25Repository`:
   - Accepts query text string
   - Returns top-k chunks with BM25 scores
   - Supports async operations
4. Parallel retrieval execution:
   - Vector and BM25 searches run concurrently using `asyncio.gather()`
   - Performance: Both complete within combined time (not sequential)
5. Reciprocal Rank Fusion (RRF) algorithm implementation:
   - Combines vector and BM25 results into unified ranking
   - Formula: `score = Σ 1/(k + rank_i)` where k=60 (standard RRF constant)
   - Returns final top-k chunks with fused scores
6. Configuration via environment:
   - `RETRIEVAL_TOP_K` (default: 20)
   - `RETRIEVAL_VECTOR_WEIGHT` (default: 0.5)
   - `RETRIEVAL_BM25_WEIGHT` (default: 0.5)
7. Structured logging with timing metrics:
   - Vector search latency
   - BM25 search latency
   - Fusion processing time
   - Total retrieval time
8. Unit tests covering:
   - Happy path: Both retrievals return results
   - Edge case: Vector returns results, BM25 returns none (and vice versa)
   - Edge case: Both return empty results
   - RRF algorithm correctness with known inputs
   - Async execution timing validation

**Technical Notes:**
- RRF is proven to improve retrieval quality by 15-30% for mixed query types
- Parallel execution critical for latency (target: <100ms for retrieval)
- Proper noun handling (character names, factions) benefits from BM25
- Vector search handles conceptual/semantic queries

**Estimated Effort:** 3 hours

---

## Story 2.2: Context Expander Implementation

**Story:** As a developer, I want a context expander that follows cross-references in chunk metadata to enrich context so that the LLM has access to related information beyond the initial retrieval results.

**Acceptance Criteria:**
1. `ContextExpander` class created in `src/rag/context_expander.py`
2. `expand_context()` method implementation:
   - Accepts List[WikiChunk] (initial retrieved chunks)
   - Accepts `expansion_depth: int` (default: 1, max: 2)
   - Returns expanded List[WikiChunk] with additional related chunks
3. Expansion algorithm (depth=1):
   - Extract all `metadata.links` (article titles) from initial chunks
   - Query `VectorRepository` to fetch chunks from linked articles
   - Limit: Max 2 additional chunks per link
   - Deduplication: Skip chunks already in results (by chunk_id)
   - Ranking: Append to end of list (preserve initial ranking)
4. Expansion algorithm (depth=2):
   - Perform depth=1 expansion
   - Extract links from depth=1 expanded chunks
   - Fetch chunks from secondary links (max 1 chunk per link)
   - Total expansion limit: Max 30 total chunks after expansion
5. Configuration via environment:
   - `CONTEXT_EXPANSION_ENABLED` (default: True)
   - `CONTEXT_EXPANSION_DEPTH` (default: 1)
   - `CONTEXT_EXPANSION_MAX_CHUNKS` (default: 30)
6. Structured logging:
   - Initial chunk count
   - Links extracted count
   - Chunks added from expansion
   - Final chunk count
   - Expansion depth used
7. Unit tests covering:
   - No expansion (depth=0 or disabled)
   - Depth=1 expansion with multiple links
   - Depth=2 expansion
   - Deduplication (linked chunk already in results)
   - Max chunk limit enforcement
   - Edge case: No links in metadata (no expansion)
   - Edge case: Linked articles not found in vector DB

**Technical Notes:**
- Cross-reference expansion improves contextual understanding (e.g., "Roboute Guilliman" → "Ultramarines", "Primarch")
- Depth=1 sufficient for most queries; depth=2 for complex multi-entity queries
- Total context limit prevents token budget overflow for LLM
- Preserves initial ranking: expanded chunks are supplementary, not primary

**Estimated Effort:** 2 hours

---

## Story 2.3: Multi-LLM Router, OpenAI & Anthropic Providers with Structured Output

**Story:** As a developer, I want a multi-LLM router with a pluggable provider system, OpenAI and Anthropic implementations with structured JSON output using Pydantic validation, so that I can generate validated responses with personality and source attribution.

**Acceptance Criteria:**
1. `LLMStructuredResponse` Pydantic model created in `src/llm/structured_output.py`:
   - `answer: str | None` (required when smalltalk=false)
   - `personality_reply: str` (always required)
   - `sources: list[HttpUrl] | None` (required when smalltalk=false, full wiki URLs)
   - `smalltalk: bool` (always required)
   - Field validators enforce rules based on smalltalk flag
   - JSON schema generation for server-side structured output
2. `LLMProvider` abstract base class created in `src/llm/base_provider.py`:
   - `async def generate(prompt: str, options: GenerationOptions) -> LLMResponse`
   - `async def generate_structured(prompt: str, options: GenerationOptions, response_schema: type[BaseModel]) -> LLMStructuredResponse`
   - `async def estimate_cost(prompt_tokens: int, completion_tokens: int) -> float`
   - `def get_provider_name() -> str`
   - `def supports_language(language: str) -> bool`
3. `GenerationOptions` data class defined:
   - `model`: str (e.g., "gpt-4o-mini", "gpt-4", "claude-3-5-sonnet-20241022")
   - `temperature`: float (default: 0.7)
   - `max_tokens`: int (default: 800, increased for structured output)
   - `system_prompt`: Optional[str]
   - `response_language`: str (default: "hu" for Hungarian)
   - `use_structured_output`: bool (default: False)
4. `LLMResponse` data class defined (for backward compatibility with text-only generation):
   - `text`: str (generated response)
   - `provider`: str (provider name)
   - `model`: str (model used)
   - `tokens_prompt`: int
   - `tokens_completion`: int
   - `cost_usd`: float
   - `latency_ms`: int
5. `OpenAIProvider` implementation in `src/llm/providers/openai_provider.py`:
   - Uses `openai` SDK v2.14+ with structured output support
   - Implements all base class methods
   - API key from environment: `OPENAI_API_KEY`
   - Default model from environment: `OPENAI_QUERY_MODEL` (default: "gpt-4o-mini")
   - **Structured Output Strategy:**
     - Primary: Use `beta.chat.completions.parse(response_format=PydanticModel)`
     - Fallback: Use `response_format={"type": "json_schema"}` + client-side Pydantic validation
     - Graceful degradation if beta API unavailable
6. `AnthropicProvider` implementation in `src/llm/providers/anthropic_provider.py`:
   - Uses `anthropic` SDK v0.75+ with structured output support
   - Implements all base class methods
   - API key from environment: `ANTHROPIC_API_KEY`
   - Default model from environment: `ANTHROPIC_QUERY_MODEL` (default: "claude-3-5-sonnet-20241022")
   - **Structured Output Strategy:**
     - Primary: Use `beta.messages.parse(response_model=PydanticModel)` with header `anthropic-beta: structured-outputs-2025-11-13`
     - Uses constrained decoding (grammar-based, cannot produce invalid JSON)
     - Fallback: Use tool use pattern + client-side Pydantic validation
     - 100-300ms overhead for grammar compilation (cached 24 hours)
7. `generate_structured()` implementation with retry logic:
   - Exponential backoff: wait times [2s, 4s, 8s] for 3 retries
   - Retry on: `RateLimitError`, `APIConnectionError`, `APITimeoutError`
   - No retry on: `AuthenticationError`, `InvalidRequestError`, `ValidationError`
   - Fail-fast on Pydantic ValidationError (LLM returned invalid schema)
   - Log each retry attempt with context
8. Hungarian/English language support:
   - System prompt includes: "Respond in Hungarian" or "Respond in English" based on `response_language`
   - Language detection: If query contains English words >50%, respond in English
9. Cost estimation implementation:
   - GPT-4o-mini: $0.00015 per 1K input tokens, $0.0006 per 1K output tokens
   - GPT-4: $0.03 per 1K input tokens, $0.06 per 1K output tokens
   - Claude 3.5 Sonnet: $0.003 per 1K input tokens, $0.015 per 1K output tokens
10. `MultiLLMRouter` class in `src/llm/llm_router.py`:
    - Provider registry: Maps provider name → LLMProvider instance
    - `async def generate(prompt: str, provider: str, options: GenerationOptions) -> LLMResponse`
    - `async def generate_structured(prompt: str, provider: str, options: GenerationOptions, response_schema: type[BaseModel]) -> LLMStructuredResponse`
    - Provider selection: Use specified provider or default from `LLM_DEFAULT_PROVIDER` env
    - No fallback logic: Fail fast if provider fails
11. Structured logging:
    - Provider name, model, tokens, cost, latency
    - Structured output validation success/failure
    - Server-side vs client-side validation path taken
    - Retry attempts with errors
    - Final success/failure status
12. Unit tests covering:
    - Pydantic model validation (smalltalk=true/false cases)
    - OpenAI provider: Successful structured generation (beta.chat.completions.parse)
    - OpenAI provider: Client-side fallback (response_format + manual validation)
    - Anthropic provider: Successful structured generation (beta.messages.parse)
    - Anthropic provider: Client-side fallback (tool use pattern)
    - Retry logic: Retryable errors (3 retries, then fail)
    - Retry logic: Non-retryable errors (fail immediately, including ValidationError)
    - Cost estimation accuracy
    - Language support (Hungarian and English prompts)
    - Multi-LLM router: Provider selection and routing
    - Multi-LLM router: Unknown provider error
    - Graceful degradation from server-side to client-side validation

**Technical Notes:**
- Strategy pattern for future provider extensibility (Gemini, Grok, Mistral)
- Pydantic ONLY for LLM responses; existing dataclasses remain unchanged
- Server-side structured output reduces latency and improves reliability
- Anthropic's constrained decoding is most reliable (grammar guarantees)
- OpenAI's beta.chat.completions.parse functional but has occasional validation errors
- Client-side validation fallback ensures compatibility with older models
- Fail-fast approach: No automatic fallbacks to prevent unexpected behavior
- Hungarian language support critical for MVP
- Retry logic essential for production reliability (handles transient API failures)

**Dependencies:**
- Add to `pyproject.toml`: `pydantic = "^2.0"`

**Estimated Effort:** 4 hours (increased from 3 hours due to structured output implementation)

---

## Story 2.4: Response Formatter with Structured Output & Personality

**Story:** As a developer, I want a response formatter that accepts structured LLM outputs and formats them with personality additions, source citations, and smalltalk handling so that users receive engaging, verifiable responses.

**Acceptance Criteria:**
1. `ResponseFormatter` class created in `src/llm/response_formatter.py`
2. `format_cli_response()` method implementation:
   - Accepts `llm_response: LLMStructuredResponse` (Pydantic model from LLM)
   - Accepts `language: str` (default: "hu")
   - Returns formatted string for CLI output
3. CLI response format for lore questions (smalltalk=false):
   ```
   [Answer text from LLM]

   [Personality reply in italic/dim style]

   📚 Források:
   - https://warhammer40k.fandom.com/wiki/Roboute_Guilliman
   - https://warhammer40k.fandom.com/wiki/Ultramarines
   ```
4. CLI response format for smalltalk (smalltalk=true):
   ```
   [Personality reply only]
   ```
5. Source URL formatting:
   - Display full wiki URLs (LLM provides URLs, not article titles)
   - No deduplication needed (LLM handles this)
   - Limit to top 5 URLs in output (prevent clutter)
6. Personality reply formatting:
   - Always display personality_reply
   - Style: italic and dim for visual distinction
   - Positioned after answer, before sources
7. `format_discord_response()` method (stub for Epic 3):
   - Returns basic text format (Discord Embed formatting in Epic 3)
   - Placeholder implementation for future
8. Hungarian/English language support:
   - "📚 Források:" for Hungarian responses
   - "📚 Sources:" for English responses
   - Language parameter determines header (not detection from text)
9. Unit tests covering:
   - Lore question formatting (smalltalk=false, all fields present)
   - Smalltalk formatting (smalltalk=true, only personality_reply)
   - Source URL display (full URLs)
   - Source limit enforcement (>5 sources)
   - Hungarian language formatting
   - English language formatting
   - Empty sources list (smalltalk case)

**Technical Notes:**
- Structured output simplifies formatting (no manual parsing)
- Sources as URLs (not article titles) for direct verification
- Personality reply adds engagement and character to responses
- Smalltalk handling enables natural conversation flow
- CLI format simple and readable; Discord format will use Embeds (Epic 3)

**Estimated Effort:** 1.5 hours

---

## Story 2.5: Query Orchestrator with Structured Output & System Prompts

**Story:** As a developer, I want a central query orchestrator that coordinates the entire RAG pipeline with structured LLM output, personality modes, and smalltalk detection so that all entry points use a consistent, engaging query flow.

**Acceptance Criteria:**
1. `QueryOrchestrator` class created in `src/orchestration/query_orchestrator.py`
2. `QueryRequest` data class defined:
   - `query_text`: str (user question)
   - `user_id`: Optional[str] (for logging)
   - `server_id`: Optional[str] (for logging)
   - `language_preference`: Optional[str] (default: "hu")
3. `QueryResponse` data class defined (updated for structured output):
   - `answer`: str (LLM answer, empty if smalltalk)
   - `personality_reply`: str (always present)
   - `sources`: List[str] (wiki URLs as strings)
   - `smalltalk`: bool (true if smalltalk, false if lore question)
   - `metadata`: dict (latency, tokens, cost, provider)
   - `error`: Optional[str] (error message if failed)
4. `async def process(request: QueryRequest) -> QueryResponse` method:
   - **Step 1: Validation**
     - Query text length: 5-2000 characters
     - UTF-8 encoding check
     - Raise `InvalidQueryError` if validation fails
   - **Step 2: Embedding Generation**
     - Call `EmbeddingService.embed(query_text)`
     - Returns 1536-dim vector
   - **Step 3: Hybrid Retrieval (ALWAYS executes)**
     - Call `HybridRetrievalService.retrieve(query_embedding, query_text, top_k=20)`
     - Returns fused chunks with scores
     - Note: Retrieval happens even for potential smalltalk (LLM determines smalltalk from context)
   - **Step 4: Context Expansion**
     - Call `ContextExpander.expand_context(chunks, depth=1)`
     - Returns expanded chunks
   - **Step 5: LLM Structured Generation**
     - Build system prompt with personality mode and structured output instructions
     - Build context from expanded chunks (article titles and text)
     - Call `MultiLLMRouter.generate_structured(prompt, provider="openai", options, LLMStructuredResponse)`
     - System prompt instructs LLM on:
       - Smalltalk detection (greetings, off-topic vs lore questions)
       - Wiki URL format: `https://warhammer40k.fandom.com/wiki/{Article_Title}` (spaces→underscores)
       - Personality reply based on `BOT_PERSONALITY` env var
       - Language (Hungarian or English)
     - Returns validated `LLMStructuredResponse` (Pydantic model)
   - **Step 6: Response Formatting**
     - Call `ResponseFormatter.format_cli_response(llm_response, language=request.language_preference)`
     - Returns formatted string with answer, personality_reply, sources
   - **Step 7: Return QueryResponse**
     - Populate metadata: latency, tokens, cost, provider
     - Convert Pydantic sources (HttpUrl) to strings
5. System prompt templates:
   - **Default (Professional):**
     - Professional, informative tone
     - personality_reply: Brief thematic closing (e.g., "The Emperor protects.")
   - **Grimdark Narrator:**
     - Dramatic, atmospheric tone
     - personality_reply: Grim, evocative (e.g., "In the grim darkness, there is only war.")
   - Both include: smalltalk detection criteria, source URL format, language instructions
   - Selected via `BOT_PERSONALITY` env var (default: "default")
6. Error handling:
   - Catch specific exceptions: `InvalidQueryError`, `RetrievalError`, `LLMProviderError`, `ValidationError`
   - Log error with full context (query_id, user_id, server_id, error details)
   - Return `QueryResponse` with `error` field populated
   - User-friendly error messages (no stack traces)
7. Structured logging for each step:
   - Step start/end with timing
   - Correlation ID (query_id) across all logs
   - Log smalltalk flag from LLM response
   - Final summary log with total latency and success/failure
8. Performance tracking:
   - Total latency (target: <3 seconds p95)
   - Breakdown: embedding (50ms), retrieval (100ms), expansion (50ms), LLM (1500-1800ms with structured output), formatting (10ms)
9. Configuration via environment:
   - `QUERY_TIMEOUT_SECONDS` (default: 10)
   - `QUERY_MAX_LENGTH` (default: 2000)
   - `QUERY_MIN_LENGTH` (default: 5)
   - `BOT_PERSONALITY` (default: "default", options: "default", "grimdark_narrator")
   - `LLM_DEFAULT_PROVIDER` (default: "openai")
10. Unit tests covering:
    - Happy path: Full pipeline success with structured output
    - Lore question flow (smalltalk=false)
    - Smalltalk flow (smalltalk=true)
    - Validation errors: Too short, too long, invalid UTF-8
    - Retrieval failure: Empty results
    - LLM failure: API error
    - LLM validation failure: Invalid structured output
    - Timeout handling
    - Metadata population correctness
    - System prompt selection based on personality mode

**Technical Notes:**
- Central orchestrator ensures consistency across all entry points
- Dependency injection for testability (pass service instances)
- Correlation ID critical for debugging multi-step pipeline
- Retrieval ALWAYS executes (even for smalltalk) - LLM determines context from retrieved chunks
- System prompts are key to structured output quality
- Personality modes configurable via env (no code changes needed)
- Error handling user-friendly (Discord users see clean messages)

**Estimated Effort:** 3 hours (increased from 2.5 hours due to structured output and system prompts)

---

## Story 2.6: Query Logger with Structured Output Fields

**Story:** As a developer, I want a query logging system that asynchronously records all query attempts with structured output metadata (personality, smalltalk flag, sources) so that I can analyze usage patterns, smalltalk vs lore ratios, debug issues, and track API costs.

**Acceptance Criteria:**
1. `QueryLog` SQLAlchemy model created in `src/models/query_log.py`:
   - `id`: UUID (primary key)
   - `timestamp`: DateTime (indexed)
   - `user_id`: String (nullable, indexed)
   - `server_id`: String (nullable, indexed)
   - `query_text`: Text
   - `response_text`: Text (nullable, answer field from LLM)
   - **NEW:** `personality_reply`: Text (nullable, personality addition from LLM)
   - **NEW:** `smalltalk`: Boolean (default: False, indexed for analytics)
   - **NEW:** `source_urls`: Text (nullable, JSON array of wiki URLs)
   - `chunks_retrieved`: JSON (list of chunk IDs)
   - `retrieval_scores`: JSON (list of scores)
   - `llm_provider`: String
   - `llm_model`: String
   - `llm_tokens_prompt`: Integer
   - `llm_tokens_completion`: Integer
   - `llm_cost_usd`: Float
   - `latency_retrieval_ms`: Integer
   - `latency_llm_ms`: Integer
   - `latency_total_ms`: Integer
   - `error_occurred`: Boolean
   - `error_message`: Text (nullable)
2. Alembic migration created for `query_logs` table:
   - Indexes on: `timestamp DESC`, `user_id`, `server_id`, `error_occurred`, **`smalltalk`**
   - Migration adds 3 new columns: `personality_reply`, `smalltalk`, `source_urls`
   - Backward compatible (additive only)
   - Run migration as part of story
3. `QueryLogRepository` class in `src/repositories/query_log_repository.py`:
   - Repository pattern implementation
   - `async def create(log: QueryLog) -> QueryLog`
   - `async def get_recent(limit: int = 100) -> List[QueryLog]`
   - `async def get_by_user(user_id: str, limit: int = 50) -> List[QueryLog]`
   - `async def get_by_server(server_id: str, limit: int = 50) -> List[QueryLog]`
   - `async def get_error_logs(limit: int = 50) -> List[QueryLog]`
   - **NEW:** `async def get_smalltalk_queries(limit: int = 50) -> List[QueryLog]` - filter by smalltalk=True
   - **NEW:** `async def get_lore_queries(limit: int = 50) -> List[QueryLog]` - filter by smalltalk=False
4. `QueryLogger` service in `src/services/query_logger.py`:
   - `async def log_query(query_request: QueryRequest, query_response: QueryResponse) -> None`
   - Converts request/response into QueryLog model
   - Maps structured output fields: personality_reply, smalltalk, source_urls
   - Serializes source_urls list to JSON string
   - Saves via QueryLogRepository
   - Asynchronous execution (non-blocking via `asyncio.create_task()`)
   - Error handling: Log failures don't crash main query flow
5. Integration with `QueryOrchestrator`:
   - Add `QueryLogger` as dependency
   - Call `query_logger.log_query()` after query completes (success or failure)
   - Fire-and-forget pattern (don't await logging)
6. Structured logging for query logger itself:
   - Log when query log saved successfully
   - Include smalltalk flag in log message
   - Log errors if database write fails
7. Unit tests covering:
   - QueryLog model creation with new fields
   - Field validation (personality_reply, smalltalk, source_urls)
   - QueryLogRepository CRUD operations
   - QueryLogRepository: Query by smalltalk flag (get_smalltalk_queries, get_lore_queries)
   - QueryLogger service: Successful logging with structured fields
   - QueryLogger service: Database error handling (doesn't crash)
   - Async logging doesn't block query response
   - Alembic migration up/down

**Technical Notes:**
- Async logging ensures query latency not impacted by database writes
- Analytics enable cost tracking: daily/weekly LLM spend monitoring
- Smalltalk analytics: Track smalltalk vs lore query ratio for user behavior insights
- Error logs help identify patterns: Which queries fail? Which providers?
- Source URLs logged for debugging and validation
- Fire-and-forget acceptable: If log write fails, query still succeeds

**Estimated Effort:** 1.5 hours

---

## Story 2.7: CLI Adapter with Structured Output Display

**Story:** As a developer, I want a CLI command that displays structured LLM outputs (answer, personality, sources) with rich formatting so that I can test the RAG query system end-to-end and validate smalltalk handling before Discord integration.

**Acceptance Criteria:**
1. `CLIAdapter` class created in `src/adapters/cli_adapter.py`:
   - Uses `typer` library for CLI framework
   - Command: `query <question> [--language] [--verbose]`
   - Example: `poetry run python -m src.adapters.cli_adapter query "Who is Roboute Guilliman?"`
2. CLI command implementation:
   - `query` command:
     - Required argument: `question` (str)
     - Optional flags:
       - `--language` (choices: "hu", "en", default: "hu")
       - `--verbose` (flag, default: False)
   - Converts CLI args into `QueryRequest`
   - Calls `QueryOrchestrator.process(request)`
   - Displays structured output from `QueryResponse`
   - Prints metadata (latency, cost, smalltalk flag) if `--verbose` flag
3. Structured output display (using `rich` library):
   - **For lore questions (smalltalk=false):**
     - Answer text in default color
     - Personality reply in italic dim style (visual distinction)
     - Sources section header in bold blue ("📚 Források:" or "📚 Sources:")
     - Source URLs as clickable links (if terminal supports: `[link]URL[/link]`)
     - Limit to 5 sources displayed
   - **For smalltalk (smalltalk=true):**
     - Only display personality_reply
     - No answer or sources section
4. Error handling:
   - Catch all exceptions from orchestrator
   - Print user-friendly error messages in red
   - Exit with code 1 on error, 0 on success
5. Verbose mode metadata display:
   - Use `rich.Panel` for metadata box
   - Show: latency_ms, tokens (prompt + completion), cost_usd, provider
   - **NEW:** Show smalltalk flag (True/False)
   - Style: dim gray
6. Integration tests in `tests/integration/test_cli_query_flow.py`:
   - **Test 1: Lore question end-to-end (smalltalk=false)**
     - Query: "Who is Roboute Guilliman?"
     - Assert: Response.answer contains relevant text
     - Assert: Response.personality_reply present
     - Assert: Response.sources not empty
     - Assert: Response.smalltalk == False
     - Assert: No error
     - Assert: Total latency <5 seconds
   - **Test 2: Smalltalk end-to-end (smalltalk=true)**
     - Query: "Hello there!"
     - Assert: Response.personality_reply present
     - Assert: Response.smalltalk == True
     - Assert: No error
   - **Test 3: Source URL validation**
     - Query: "Tell me about the Ultramarines."
     - Assert: All sources are valid wiki URLs
     - Assert: URLs contain "https://warhammer40k.fandom.com/wiki/"
     - Assert: No spaces in URLs (underscores only)
   - **Test 4: LLM API failure**
     - Mock OpenAI API to return 503 error
     - Assert: User sees friendly error message
     - Assert: Query logged with `error_occurred=True`
   - **Test 5: Hungarian and English language**
     - Query in Hungarian: "Ki a Guilliman?"
     - Assert: Response in Hungarian
     - Query in English: "Who is Guilliman?"
     - Assert: Response in English
7. Performance benchmarking script in `scripts/benchmark_query.py`:
   - Run 10 test queries (mix of lore and smalltalk)
   - Record latency for each
   - Print p50, p95, p99 latencies
   - Print average cost per query
   - Print smalltalk vs lore ratio
8. Documentation updated:
   - `README.md` section: "Testing the RAG Query System"
   - Instructions: How to run CLI query command
   - Examples: 3-5 example queries (lore + smalltalk) with expected outputs

**Technical Notes:**
- CLI testing validates RAG pipeline before Discord complexity
- Integration tests use real vector DB (Chroma test collection) and mocked LLM
- Typer provides excellent CLI UX with help text and type validation
- Rich library makes output readable, professional, and engaging
- Smalltalk testing validates natural conversation flow
- Source URL validation ensures LLM generates proper wiki links

**Estimated Effort:** 2.5 hours

---

## Story 2.8: RAG Evaluation & Observability Framework

**Story:** As a developer, I want an evaluation and observability framework using Langfuse (tracing) and Arize Phoenix (evaluation) so that I can measure retrieval quality, detect hallucinations, and optimize the RAG pipeline with data-driven insights.

**Acceptance Criteria:**
1. Dependencies added to `pyproject.toml`:
   - `langfuse ^2.36` (tracing & prompt management)
   - `arize-phoenix ^4.0` (evaluation with UI dashboard)
   - `openinference-instrumentation-openai ^1.0` (OpenTelemetry for OpenAI)
2. `src/observability/` module created:
   - `tracer.py`: Langfuse singleton tracer with graceful degradation (disabled = no-op)
   - `decorators.py`: `@trace_span` decorator for pipeline step tracing
   - Tracer captures: LLM calls with costs, latency, token counts, session tracking
3. `src/evaluation/` module created:
   - `phoenix_client.py`: Phoenix client singleton (graceful degradation when disabled)
   - `evaluators.py`: `RAGEvaluator` class with hallucination and relevance evaluators
   - `experiments.py`: A/B testing utilities for prompt comparison
4. Configuration via environment (`.env.example` updated):
   ```
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
5. Langfuse tracing integration:
   - `@observe()` decorator on `QueryOrchestrator.process()`
   - Automatic LLM call tracing via OpenAI instrumentation
   - Metadata capture: query_type, user_id, server_id
   - Prompt versioning support for A/B comparisons
6. Phoenix evaluation implementation:
   - `HallucinationEvaluator`: Detects responses not grounded in context
   - `RelevanceEvaluator`: Scores query-response relevance
   - `RAGEvaluator.evaluate_response(query, response, context)` returns evaluation dict
   - Evaluation results logged to Phoenix for UI visualization
7. Graceful degradation:
   - When `LANGFUSE_ENABLED=false`: All tracing becomes no-op
   - When `PHOENIX_ENABLED=false`: All evaluation becomes no-op
   - No exceptions thrown when services disabled
   - Pipeline functions normally without observability dependencies
8. CLI commands for evaluation:
   - `poetry run python -c "import phoenix as px; px.launch_app()"` launches Phoenix UI
   - `poetry run pytest -m evaluation` runs evaluation test suite
9. Unit tests covering:
   - Tracer singleton initialization (enabled and disabled)
   - `@trace_span` decorator (traces when enabled, no-op when disabled)
   - Phoenix client initialization (enabled and disabled)
   - `RAGEvaluator` with mocked evaluators
   - Graceful degradation (no crashes when services unavailable)
10. Evaluation test fixtures created:
    - `tests/evaluation/conftest.py`: Evaluation fixtures
    - `tests/evaluation/test_retrieval_quality.py`: Retrieval metric tests
    - `tests/evaluation/test_generation_quality.py`: Generation metric tests
    - `tests/evaluation/datasets/sample_questions.json`: Golden Q&A pairs (5-10 examples)

**Technical Notes:**
- Langfuse: Best-in-class prompt management with customizable dashboards (MIT license, self-host free)
- Phoenix: Built-in evaluation UI with pre-built templates (ELv2 license, self-host free)
- Both support OpenTelemetry for shared trace data
- Separation of concerns: Langfuse for ops/prompts, Phoenix for quality evaluation
- Graceful degradation critical: Evaluation is optional, not required for query pipeline
- Self-hosted = unlimited usage with full UI (no paid tiers needed)

**Files to Create:**
| File | Purpose |
|------|---------|
| `src/observability/__init__.py` | Module exports |
| `src/observability/tracer.py` | Langfuse singleton tracer |
| `src/observability/decorators.py` | @trace_span decorator |
| `src/evaluation/__init__.py` | Evaluation module exports |
| `src/evaluation/phoenix_client.py` | Phoenix client singleton |
| `src/evaluation/evaluators.py` | RAG evaluation functions |
| `src/evaluation/experiments.py` | A/B testing utilities |
| `tests/evaluation/conftest.py` | Evaluation fixtures |
| `tests/evaluation/test_retrieval_quality.py` | Retrieval tests |
| `tests/evaluation/test_generation_quality.py` | Generation tests |
| `tests/evaluation/datasets/sample_questions.json` | Golden Q&A pairs |

**Estimated Effort:** 3 hours

---

## Epic Completion Criteria

**Functionality:**
- ✅ CLI command accepts natural language queries and returns formatted responses
- ✅ Hybrid retrieval (vector + BM25) successfully retrieves relevant chunks
- ✅ Context expansion follows cross-references correctly
- ✅ OpenAI LLM generates coherent Hungarian/English responses
- ✅ Response formatting includes proper source citations
- ✅ Query logging captures all attempts with metadata
- ✅ Integration tests pass (4 tests minimum)
- ✅ Observability framework operational (Langfuse tracing when enabled)
- ✅ Evaluation framework operational (Phoenix evaluators when enabled)

**Performance:**
- ✅ p95 query latency <5 seconds (target: 3 seconds)
- ✅ Retrieval latency <100ms
- ✅ LLM generation latency <2 seconds

**Quality:**
- ✅ Unit tests pass for all components (>70% coverage target)
- ✅ Integration tests pass for critical workflows
- ✅ No lint errors (`ruff check`)
- ✅ No type errors (`mypy`)
- ✅ Structured logging functional across all components

**Cost Validation:**
- ✅ Average cost per query <$0.002 (GPT-3.5-turbo)
- ✅ Cost tracking accurate (logged per query)

**Documentation:**
- ✅ README.md updated with CLI usage instructions
- ✅ Example queries documented
- ✅ Architecture diagram updated (optional)

---

## Compatibility Requirements

- ✅ Existing Epic 1 repositories (`VectorRepository`, `BM25Repository`) remain unchanged
- ✅ Database schema changes backward compatible (additive only via Alembic migrations)
- ✅ No changes to markdown archive structure
- ✅ SQLite database compatible with existing infrastructure

---

## Risk Mitigation

**Primary Risk:** LLM API failures or rate limits during testing/usage

**Mitigation:**
- Retry logic with exponential backoff (3 retries)
- Graceful error handling with user-friendly messages
- Query logging captures failures for debugging
- No fallback providers (fail-fast approach prevents confusion)

**Rollback Plan:**
- Alembic migration rollback: `alembic downgrade -1`
- Remove CLI adapter usage, continue using Epic 1 infrastructure
- No data loss: Query logs table can be dropped safely

---

## Handoff Notes

**To Story Manager (Epic 3):**

"Epic 2 delivers a fully functional CLI-testable RAG query system. The orchestrator pattern is designed for easy integration with Discord bot handlers in Epic 3. Key integration points:

- `QueryOrchestrator.process(QueryRequest)` is the single entry point
- Discord adapter will translate Discord messages → QueryRequest
- `ResponseFormatter` has stub `format_discord_response()` ready for Epic 3 Embed implementation
- Query logging captures user_id and server_id for Discord analytics
- Langfuse tracing provides LLM cost tracking and prompt versioning (optional, graceful degradation)
- Phoenix evaluation enables quality monitoring: hallucination detection, relevance scoring (optional, graceful degradation)

The system is production-ready for CLI usage and architected for seamless Discord integration. Evaluation framework enables continuous quality improvement without blocking core functionality."

---

**Epic 2 Complete**

Generated by John (PM Agent) using BMAD™ Core
