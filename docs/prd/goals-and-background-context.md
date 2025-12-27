# Goals and Background Context

## Goals

- Create a Discord bot that answers Warhammer 40,000 lore questions using RAG (Retrieval-Augmented Generation) technology
- Provide accurate, contextual responses in Hungarian (default) or English (based on query language)
- Enable users to explore WH40K lore through natural conversation on Discord
- Support multiple Discord servers with .env-configured settings
- Include gamification through trivia challenges and leaderboards
- Support multiple LLM providers (OpenAI, Gemini, Claude, Grok, Mistral) for cost optimization and quality testing
- Maintain a private, self-hosted solution with full control over data and costs

## Background Context

The Warhammer 40,000 universe has extensive lore spanning decades of novels, codexes, and wiki content. Hungarian-speaking fans lack accessible tools to explore this lore in their native language. While the Fandom wiki (warhammer40k.fandom.com) provides comprehensive English content, navigating it requires significant time and effort, especially for newcomers.

This Discord bot solves this problem by combining a RAG architecture with intelligent language matching. Users can ask questions naturally in Discord (in Hungarian or English), and the bot retrieves relevant lore chunks from a vector database, then generates contextual responses in the same language as the query (defaulting to Hungarian). The hybrid retrieval system (vector + BM25) handles WH40K's extensive proper nouns effectively, while metadata-driven filtering enables rich features like spoiler-free mode and faction-specific queries.

The project uses the Fandom Wiki XML export (173MB, ~10,000-15,000 articles) as the primary data source, eliminating the need for web scraping and enabling offline processing.

## Change Log

| Date | Version | Description | Author |
|------|---------|-------------|--------|
| 2025-12-26 | 1.0 | Initial PRD draft | PM Agent (John) |
| 2025-12-26 | 1.1 | Updated configuration model: .env-only (no Discord config), fixed Hungarian language, role-based admin, trivia uses Discord Embed buttons | PM Agent (John) |
| 2025-12-26 | 1.2 | Corrected language handling: Hungarian default with automatic English response when user queries in English (via LLM prompt, not configuration) | PM Agent (John) |

---
