# Requirements

## Functional Requirements

**Data Ingestion & Processing:**

- **FR1:** System SHALL parse the WH40K Fandom Wiki XML export file (MediaWiki format) and extract main articles (namespace 0 only)
- **FR2:** System SHALL convert MediaWiki markup to Markdown format, preserving headings, links, lists, and formatting
- **FR3:** System SHALL chunk articles into semantically coherent pieces (200-500 tokens) based on section boundaries
- **FR4:** System SHALL extract metadata from chunks including: faction, era, character names, internal links, content type, and source books
- **FR5:** System SHALL generate vector embeddings for all chunks using OpenAI text-embedding-3-small (1536 dimensions)
- **FR6:** System SHALL store chunks, embeddings, and metadata in a vector database (Chroma for MVP, upgradeable to Qdrant)
- **FR7:** System SHALL build and maintain a BM25 index for keyword-based retrieval alongside vector search

**Query & Retrieval:**

- **FR8:** System SHALL accept natural language queries in Hungarian or English through Discord
- **FR9:** System SHALL perform hybrid retrieval combining vector similarity search and BM25 keyword matching using Reciprocal Rank Fusion
- **FR10:** System SHALL filter retrieved chunks based on server configuration (spoiler-free mode, faction filters, era filters)
- **FR11:** System SHALL optionally expand retrieved chunks by following cross-references in metadata links
- **FR12:** System SHALL rank and select top-k most relevant chunks for LLM context

**Response Generation:**

- **FR13:** System SHALL generate responses using configurable LLM providers (OpenAI, Gemini, Claude, Grok, Mistral)
- **FR14:** System SHALL generate responses in Hungarian by default, but respond in English when user queries in English (language detection via LLM prompt)
- **FR15:** System SHALL support configurable personality modes (e.g., "default", "grimdark_narrator") via .env with custom system prompts
- **FR16:** System SHALL cite sources by referencing wiki article titles in responses
- **FR17:** System SHALL handle follow-up questions with conversation context (last 3-5 exchanges)

**Discord Bot Integration:**

- **FR18:** System SHALL respond to Discord messages mentioning the bot or using a configurable command prefix
- **FR19:** System SHALL use Discord Guild roles to determine admin privileges for restricted commands
- **FR20:** System SHALL enforce rate limits per user and per server to control API costs (configured via .env)
- **FR21:** System SHALL provide admin-only commands including: trivia management (`/trivia new`, `/trivia end`), leaderboard reset (`/leaderboard reset`)
- **FR22:** System SHALL log all queries with metadata (user, server, query, response time, LLM tokens used) for analytics
- **FR23:** System SHALL read all configuration from .env file, with no runtime Discord-based configuration

**Trivia & Gamification:**

- **FR24:** System SHALL generate trivia questions on WH40K lore with configurable difficulty levels
- **FR25:** System SHALL present trivia questions as Discord Embeds with interactive button choices
- **FR26:** System SHALL validate user answers based on button interactions and track correct/incorrect attempts
- **FR27:** System SHALL maintain per-server leaderboards with scores, streaks, and achievements
- **FR28:** System SHALL support trivia commands: `/trivia new` (admin-only), `/trivia end` (admin-only), `/leaderboard`, `/leaderboard reset` (admin-only)

**Administration & Monitoring:**

- **FR29:** System SHALL provide CLI commands for data ingestion, index rebuilding, and database management
- **FR30:** System SHALL log errors, warnings, and info events to structured logs (JSON format)
- **FR31:** System SHALL track API usage and costs per LLM provider for monitoring

## Non-Functional Requirements

**Performance:**

- **NFR1:** System SHALL respond to Discord queries within 5 seconds on average (p95 < 10 seconds)
- **NFR2:** System SHALL support ingestion of full wiki XML (173MB, ~10,000 articles) in under 4 hours
- **NFR3:** System SHALL process XML with memory footprint under 500MB using streaming parser

**Scalability:**

- **NFR4:** System SHALL support up to 5 Discord servers concurrently in MVP phase
- **NFR5:** System SHALL handle up to 100 queries per hour across all servers without degradation

**Reliability:**

- **NFR6:** System SHALL implement retry logic with exponential backoff for LLM API calls (max 3 retries)
- **NFR7:** System SHALL handle rate limit errors gracefully with user-friendly error messages
- **NFR8:** System SHALL log all errors with sufficient context for debugging

**Security:**

- **NFR9:** System SHALL store API keys encrypted in environment variables, never in code or version control
- **NFR10:** System SHALL use defusedxml for XML parsing to prevent XML bomb attacks
- **NFR11:** System SHALL validate and sanitize all user inputs to prevent injection attacks

**Maintainability:**

- **NFR12:** System SHALL use Poetry for dependency management with pinned versions
- **NFR13:** System SHALL include comprehensive unit tests for core business logic (target: >70% coverage)
- **NFR14:** System SHALL include integration tests for critical workflows (ingestion pipeline, query flow)
- **NFR15:** System SHALL use Alembic for database schema migrations with version control
- **NFR16:** System SHALL follow repository pattern for data access abstraction
- **NFR17:** System SHALL use structured logging with contextual metadata (correlation IDs, user IDs, server IDs)

**Cost Efficiency:**

- **NFR18:** System SHALL aim for under $5/month in LLM API costs for 1000 queries (using GPT-3.5-turbo baseline)
- **NFR19:** System SHALL support switching between LLM providers to optimize cost vs. quality trade-offs
- **NFR20:** System SHALL use one-time embedding generation cost of ~$1 for full wiki (text-embedding-3-small)

**Deployment:**

- **NFR21:** System SHALL run on a single Debian Linux server (traditional deployment, no serverless)
- **NFR22:** System SHALL use SQLite for structured data persistence (logs, trivia, server config)
- **NFR23:** System SHALL persist vector database to disk (Chroma embedded mode for MVP)

---
