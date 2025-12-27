# Epic List

The project is organized into 5 sequential epics, each delivering deployable, testable functionality:

## Epic 1: Foundation & Data Pipeline
**Goal:** Establish project infrastructure and implement a complete data ingestion pipeline that parses the WH40K Fandom Wiki XML export, converts it to markdown, chunks the content, generates embeddings, and stores everything in a vector database. Includes creation of a 100-page test bed for RAG fine-tuning.

**Value:** Enables the RAG system to have access to searchable WH40K lore data, which is the core requirement for all subsequent features.

**Estimated Effort:** 18-25 hours

---

## Epic 2: Core RAG Query System
**Goal:** Build the RAG query engine with hybrid retrieval (vector + BM25), metadata filtering, context expansion, and LLM-powered response generation using OpenAI.

**Value:** Delivers the core lore query functionality that can be tested via CLI, validating the RAG pipeline before Discord integration complexity.

**Estimated Effort:** 12-16 hours

---

## Epic 3: Discord Bot Integration & User Interactions
**Goal:** Integrate the RAG engine with Discord, implement bot commands, role-based admin permissions, rate limiting, conversation context, and automatic language matching (Hungarian default, English when queried in English).

**Value:** Makes the lore bot accessible to users on Discord with intelligent language handling and admin-controlled features, completing the MVP user experience.

**Estimated Effort:** 10-14 hours

---

## Epic 4: Trivia System & Gamification
**Goal:** Implement trivia question generation, answer validation, leaderboards, and gamification features to enhance user engagement.

**Value:** Adds fun, interactive features that increase user retention and community engagement beyond passive lore queries.

**Estimated Effort:** 8-12 hours

---

## Epic 5: Multi-LLM Support & Provider Management
**Goal:** Add support for multiple LLM providers (Gemini, Claude, Grok, Mistral) with provider abstraction and .env-based provider selection.

**Value:** Enables cost optimization and quality testing across different LLM providers, giving administrators flexibility to choose based on performance and budget via .env configuration.

**Estimated Effort:** 6-10 hours

---
