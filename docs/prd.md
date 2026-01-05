# WH40K Lore Discord Bot Product Requirements Document (PRD)

**Version:** 1.3
**Date:** 2026-01-05
**Status:** Draft
**Author:** PM Agent (John)

---

## Goals and Background Context

### Goals

- Create a Discord bot that answers Warhammer 40,000 lore questions using RAG (Retrieval-Augmented Generation) technology
- Provide accurate, contextual responses in Hungarian (default) or English (based on query language)
- Enable users to explore WH40K lore through natural conversation on Discord
- Support multiple Discord servers with .env-configured settings
- Include gamification through trivia challenges and leaderboards
- Support multiple LLM providers (OpenAI, Gemini, Claude, Grok, Mistral) for cost optimization and quality testing
- Maintain a private, self-hosted solution with full control over data and costs

### Background Context

The Warhammer 40,000 universe has extensive lore spanning decades of novels, codexes, and wiki content. Hungarian-speaking fans lack accessible tools to explore this lore in their native language. While the Fandom wiki (warhammer40k.fandom.com) provides comprehensive English content, navigating it requires significant time and effort, especially for newcomers.

This Discord bot solves this problem by combining a RAG architecture with intelligent language matching. Users can ask questions naturally in Discord (in Hungarian or English), and the bot retrieves relevant lore chunks from a vector database, then generates contextual responses in the same language as the query (defaulting to Hungarian). The hybrid retrieval system (vector + BM25) handles WH40K's extensive proper nouns effectively, while metadata-driven filtering enables rich features like spoiler-free mode and faction-specific queries.

The project uses the Fandom Wiki XML export (173MB, ~10,000-15,000 articles) as the primary data source, eliminating the need for web scraping and enabling offline processing.

### Change Log

| Date | Version | Description | Author |
|------|---------|-------------|--------|
| 2025-12-26 | 1.0 | Initial PRD draft | PM Agent (John) |
| 2025-12-26 | 1.1 | Updated configuration model: .env-only (no Discord config), fixed Hungarian language, role-based admin, trivia uses Discord Embed buttons | PM Agent (John) |
| 2025-12-26 | 1.2 | Corrected language handling: Hungarian default with automatic English response when user queries in English (via LLM prompt, not configuration) | PM Agent (John) |
| 2026-01-05 | 1.3 | Added Story 2.8: RAG Evaluation & Observability Framework (Langfuse + Arize Phoenix) to Epic 2; updated effort estimate to 13-17 hours | PM Agent (John) |

---

## Requirements

### Functional Requirements

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

### Non-Functional Requirements

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

## User Interface Design Goals

**Note:** This is a Discord bot, so the primary UI is Discord's native interface. These goals apply to the bot's interaction patterns and command design.

### Overall UX Vision

The bot should feel like a knowledgeable lore expert engaging in natural conversation, responding in Hungarian by default or matching the user's query language (English). Users should not need to learn complex syntax - simple mentions and conversational questions should work seamlessly. The bot's personality mode is configured via .env (e.g., neutral informative vs. grimdark narrator).

### Key Interaction Paradigms

- **Natural Language First:** Users ask questions naturally, no rigid command syntax required
- **Mention-Based Activation:** `@WH40K-Bot Who is Roboute Guilliman?` triggers a query
- **Slash Commands for Features:** `/trivia new` (admin), `/leaderboard` for structured features
- **Interactive Buttons:** Trivia questions use Discord Embed buttons for answer selection
- **Contextual Follow-Ups:** Bot remembers recent conversation context for follow-up questions
- **Inline Citations:** Responses include source article names for user verification

### Core Interaction Flows

1. **Lore Query Flow (Hungarian):**
   - User: `@WH40K-Bot Mesélj a Horus Eretnekségről`
   - Bot: `[Typing indicator]` → Response with citations in Hungarian
   - User: `Mikor történt?` (follow-up)
   - Bot: Responds using conversation context in Hungarian

1a. **Lore Query Flow (English):**
   - User: `@WH40K-Bot Tell me about the Horus Heresy`
   - Bot: `[Typing indicator]` → Response with citations in English
   - User: `When did it happen?` (follow-up)
   - Bot: Responds using conversation context in English

2. **Trivia Flow:**
   - Admin: `/trivia new medium` (starts new trivia question)
   - Bot: Posts Discord Embed with question text and interactive buttons (A, B, C, D)
   - User: Clicks button B
   - Bot: Updates embed with result (correct/incorrect), updates score, shows leaderboard position

3. **Admin Commands:**
   - Admin (with configured Discord role): `/trivia new {difficulty}`
   - Admin: `/trivia end` (ends current trivia)
   - Admin: `/leaderboard reset`
   - Non-admins receive error: "This command requires admin role"

### Accessibility

- Responses in Hungarian by default, English when user queries in English (automatic language detection)
- Clear error messages in appropriate language
- Simple, intuitive commands with help text

### Branding

- Bot name: "WH40K Lore Bot" (configured via .env)
- Avatar: Aquila symbol or WH40K themed icon
- Tone: Professional yet approachable, personality mode configured via .env
- Primary audience: Hungarian WH40K fans, with English support for international users

### Target Platforms

- Discord (desktop and mobile apps)
- Cross-platform (wherever Discord runs)

---

## Technical Assumptions

### Repository Structure: Monorepo

All components (RAG engine, Discord bot, CLI tools, tests) reside in a single repository for simplified development and deployment.

### Service Architecture

**Monolithic Service with Central Orchestrator Pattern:**
- Single Python application containing all components
- Central RAG Orchestrator exposes unified query interface
- Discord Bot Adapter is a thin client calling the orchestrator
- CLI Adapter provides offline tools (ingestion, index rebuild)
- Traditional server-based deployment (single Debian VM process)
- Simpler operations, lower latency, persistent in-memory caching

**Rationale:** For MVP with <5 servers and <1000 queries/month, a monolith is simpler to develop, deploy, and maintain than microservices. The central orchestrator pattern ensures the RAG pipeline is reusable across Discord, CLI, and tests.

### Testing Requirements

**Testing Pyramid Approach:**
- **Unit Tests:** Core business logic (chunking, metadata extraction, retrieval, ranking) with >70% coverage target
- **Integration Tests:** Critical workflows end-to-end:
  - XML parsing → markdown → chunking → embedding → vector DB storage
  - Query → retrieval → ranking → LLM generation → response
  - Discord command handling → orchestrator → response
- **Manual Testing:** Discord bot interaction testing (conversational flows, error handling)
- **Testing Tools:** pytest, pytest-cov, pytest-mock, pytest-asyncio

**Rationale:** Unit tests catch logic bugs early, integration tests validate end-to-end workflows, manual testing ensures Discord UX quality.

### Additional Technical Assumptions and Requests

**Programming Language:**
- **Python 3.11+** for all components (Discord.py, async/await support, modern type hints)

**Data Source:**
- **WH40K Fandom Wiki XML Export** (~173MB MediaWiki XML dump format)
- Downloaded manually from `https://warhammer40k.fandom.com`
- Legal: CC-BY-SA licensed, non-commercial use, attribution required
- **No web scraping required** - single offline XML file parsing

**Vector Database:**
- **MVP:** Chroma (embedded mode, persistent to disk, zero server setup)
- **Future:** Qdrant (if advanced metadata filtering or scaling needed)

**BM25 Search:**
- **rank-bm25** library (pure Python, no external dependencies)

**Embedding API:**
- **OpenAI text-embedding-3-small** (1536 dimensions, $0.02 per 1M tokens)
- Expected one-time cost: ~$1 for full wiki (~50K chunks)

**LLM Providers:**
- **MVP:** OpenAI API (GPT-3.5-turbo for cost efficiency, GPT-4 for quality)
- **Phase 2:** Add Gemini, Claude, Grok, Mistral via multi-provider strategy pattern
- Provider abstraction layer for easy switching

**XML Parsing:**
- **lxml** (fast C-based parser, XPath support)
- **defusedxml** (security protection against XML vulnerabilities)
- **mwparserfromhell** (MediaWiki markup → Markdown conversion)
- Streaming parser (iterparse) for memory efficiency (<500MB footprint)

**Database (Structured Data):**
- **SQLite** (embedded, zero-config, adequate for <5 servers)
- Stores: query logs, trivia attempts, server configs, leaderboards

**ORM & Migrations:**
- **SQLAlchemy 2.0+** (type-safe ORM, async support, repository pattern)
- **Alembic** (schema versioning and migrations)

**Discord Integration:**
- **Discord.py** (official Python library for Discord bots)

**Development Tools:**
- **Poetry** (dependency management, pinned versions)
- **ruff** (fast Python linter and formatter, replaces flake8+black+isort)
- **mypy** (static type checking)
- **structlog** (structured JSON logging with context injection)

**Deployment:**
- **Single Debian Linux VM** (traditional server-based deployment)
- **systemd** service for bot process management
- **Poetry** for environment management on server
- No containerization in MVP (can add later if needed)

**Cross-Cutting Concerns:**
- **Logging:** structlog with JSON formatting from start (integrated into all stories)
- **Error Handling:** Comprehensive error handling with retry logic for external APIs
- **Configuration:** All configuration via .env file (python-dotenv), including:
  - API keys (never in code/version control)
  - Bot settings (name, personality mode)
  - Rate limits (per-user, per-server)
  - Admin role names (for Discord role-based permissions)
  - LLM provider selection
  - No runtime configuration via Discord commands
- **Language Handling:** Automatic language detection and matching via LLM system prompt (Hungarian default, English when user queries in English)

---

## Epic List

The project is organized into 5 sequential epics, each delivering deployable, testable functionality:

### Epic 1: Foundation & Data Pipeline
**Goal:** Establish project infrastructure and implement a complete data ingestion pipeline that parses the WH40K Fandom Wiki XML export, converts it to markdown, chunks the content, generates embeddings, and stores everything in a vector database. Includes creation of a 100-page test bed for RAG fine-tuning.

**Value:** Enables the RAG system to have access to searchable WH40K lore data, which is the core requirement for all subsequent features.

**Estimated Effort:** 18-25 hours

---

### Epic 2: Core RAG Query System
**Goal:** Build the RAG query engine with hybrid retrieval (vector + BM25), metadata filtering, context expansion, LLM-powered response generation using OpenAI, and RAG evaluation framework for quality monitoring.

**Value:** Delivers the core lore query functionality that can be tested via CLI, validating the RAG pipeline before Discord integration complexity. Includes observability and evaluation tools for continuous quality improvement.

**Estimated Effort:** 13-17 hours

---

### Epic 3: Discord Bot Integration & User Interactions
**Goal:** Integrate the RAG engine with Discord, implement bot commands, role-based admin permissions, rate limiting, conversation context, and automatic language matching (Hungarian default, English when queried in English).

**Value:** Makes the lore bot accessible to users on Discord with intelligent language handling and admin-controlled features, completing the MVP user experience.

**Estimated Effort:** 10-14 hours

---

### Epic 4: Trivia System & Gamification
**Goal:** Implement trivia question generation, answer validation, leaderboards, and gamification features to enhance user engagement.

**Value:** Adds fun, interactive features that increase user retention and community engagement beyond passive lore queries.

**Estimated Effort:** 8-12 hours

---

### Epic 5: Multi-LLM Support & Provider Management
**Goal:** Add support for multiple LLM providers (Gemini, Claude, Grok, Mistral) with provider abstraction and .env-based provider selection.

**Value:** Enables cost optimization and quality testing across different LLM providers, giving administrators flexibility to choose based on performance and budget via .env configuration.

**Estimated Effort:** 6-10 hours

---

## Epic Details

### Epic 1: Foundation & Data Pipeline

**Detailed stories documented in:** [docs/epic-1-foundation-data-pipeline.md](epic-1-foundation-data-pipeline.md)

**Summary of 12 Stories:**
1. Project Setup & Core Infrastructure
2. Database Schema Design & Implementation (includes wiki_page_id)
3. Markdown Archive Setup
4. Wiki XML Parser Implementation (with page ID filtering support)
5. Test Bed Page Selection (100-page dataset from Blood Angels seed)
6. Intelligent Text Chunking Implementation
7. OpenAI Embeddings Integration
8. Chroma Vector Database Integration
9. Metadata Extraction from Content
10. Complete Ingestion Pipeline Orchestration (with page ID filtering)
11. Data Quality Validation & Monitoring
12. Ingestion Documentation & Usage Guide

**Epic Completion Criteria:**
- ✅ Test bed (100 pages) successfully created and ingested for RAG fine-tuning
- ✅ Full wiki XML (173MB) successfully ingested
- ✅ Vector database contains ~50,000 chunks with embeddings
- ✅ Markdown archive contains ~10,000-15,000 articles
- ✅ WikiChunk table includes wiki_page_id field for traceability
- ✅ Integration tests pass
- ✅ Processing time: 2-4 hours for full wiki, <5 minutes for test bed
- ✅ Cost estimate validated: ~$1.00 for embeddings

---

### Epic 2: Core RAG Query System

**Detailed stories documented in:** [docs/epic-2-core-rag-query-system.md](epic-2-core-rag-query-system.md)

**Summary of 8 Stories:**
1. Hybrid Retrieval Service (Vector + BM25 with RRF)
2. Context Expander (cross-reference following)
3. Multi-LLM Router & OpenAI Provider
4. Response Formatter & Source Attribution
5. Query Orchestrator Implementation
6. Query Logger & Analytics
7. CLI Adapter & Integration Tests
8. RAG Evaluation & Observability Framework (Langfuse + Arize Phoenix)

**Epic Completion Criteria:**
- ⬜ CLI command functional for natural language queries
- ⬜ Hybrid retrieval delivers relevant chunks (vector + BM25)
- ⬜ Context expansion follows cross-references
- ⬜ OpenAI LLM generates Hungarian/English responses
- ⬜ Source citations properly formatted
- ⬜ Query logging captures all attempts
- ⬜ Integration tests pass (4+ tests)
- ⬜ Observability framework operational (Langfuse tracing when enabled)
- ⬜ Evaluation framework operational (Phoenix evaluators when enabled)
- ⬜ p95 latency <5 seconds
- ⬜ Average cost <$0.002 per query

---

### Epic 3: Discord Bot Integration & User Interactions

*[To be detailed in next iteration]*

---

### Epic 4: Trivia System & Gamification

*[To be detailed in next iteration]*

---

### Epic 5: Multi-LLM Support & Server Configuration

*[To be detailed in next iteration]*

---

## Checklist Results Report

*[To be completed after all epic details are finalized]*

---

## Next Steps

### UX Expert Prompt

*[To be completed - Discord bot interaction design review]*

### Architect Prompt

"Please review this PRD and the existing architecture document (docs/architecture.md) to validate alignment. The architecture was created before this PRD and has already been updated for XML parsing and removal of canon classification. Identify any gaps or inconsistencies between the PRD requirements and the current architecture, and propose updates if needed."

---

**End of PRD v1.3**
