# Research Prompt: RAG Architectures for Wiki Data Retrieval

**Research Type:** Technology & Innovation Research
**Created:** 2025-12-17
**Project Context:** WH40k Lore Discord Bot

---

## Research Objective

**Primary Goal:** Comprehensively survey and evaluate RAG (Retrieval-Augmented Generation) architectures suitable for wiki-structured data, identifying optimal approaches for retrieving Warhammer 40k lore from scraped wiki sources with emphasis on accuracy, cost efficiency, and implementation feasibility.

**Scope:** Explore the full landscape of RAG architectures beyond basic vector search, including hierarchical retrieval (RAPTOR), graph-based approaches, hybrid methods, and emerging techniques specifically suited for interconnected knowledge bases like wikis.

---

## Background Context

### Project Overview
Building a Discord bot that answers Warhammer 40k lore questions using LLM+RAG technology for a 100-300 member Hungarian community.

### Technical Foundation
- **Existing infrastructure:** Python Discord bot, LLM+RAG pipeline operational
- **Current VectorDB:** Chroma (open to alternatives)
- **Data sources:** Warhammer 40k Fandom Wiki and/or Lexicanum
- **Multi-LLM support:** OpenAI, Gemini, Claude, Grok, Mistral
- **Cost model:** Few cents per query (cost efficiency matters)

### Wiki Data Characteristics
- **Structure:** Heavily cross-linked articles with sections, subsections, infoboxes, tables
- **Hyperlinks:** Dense network of wiki-internal references between topics
- **Content types:** Narrative lore, character bios, event timelines, faction descriptions, technical specs
- **Metadata needs:** Faction, character, era, source book, URL, canon/speculation flags, spoiler markers
- **Chunking strategy:** Section-based (no overlap preferred)

### Key Challenges
1. **Cross-reference handling:** Wiki hyperlinks create semantic relationships that pure vector search may miss
2. **Query types diversity:**
   - Specific facts ("Who is Roboute Guilliman?")
   - Broad topics ("Explain the Horus Heresy")
   - Relationships ("How are Blood Angels connected to Sanguinius?")
   - Comparisons ("Difference between Space Marines and Chaos Space Marines?")
3. **Canon vs speculation:** Need to retrieve and distinguish authoritative lore from fan theories
4. **Cost constraints:** Embedding and retrieval costs must be sustainable at community scale

### Prototype Approach
- **100-page testbed:** Small wiki subset for rapid architecture experimentation
- **Evaluation criteria:** Retrieval accuracy, cost, latency, implementation complexity
- **Decision timeline:** Architecture selection needed before full wiki scraping

---

## Research Questions

### Primary Questions (Must Answer)

1. **What RAG architectures exist beyond basic vector search?**
   - Survey of current state-of-the-art approaches
   - Hierarchical retrieval methods (RAPTOR, etc.)
   - Graph-based RAG approaches
   - Hybrid retrieval strategies (vector + keyword/BM25)
   - Multi-hop reasoning architectures
   - Late-interaction models (ColBERT, etc.)
   - Agentic RAG patterns

2. **How does RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval) work?**
   - Core methodology and workflow
   - Tree construction from document chunks
   - Abstraction/summarization strategy
   - Query-time traversal mechanisms
   - Implementation requirements and complexity
   - Proven use cases and performance benchmarks

3. **What graph-RAG architectures are suitable for wiki data?**
   - Knowledge graph construction from wiki hyperlinks
   - Entity relationship extraction methods
   - Graph embedding techniques
   - Query-time graph traversal strategies
   - Hybrid vector-graph retrieval patterns
   - Existing frameworks (LangGraph, Neo4j integration, etc.)

4. **How do hybrid retrieval methods (vector + BM25/keyword) compare?**
   - Dense vs sparse retrieval trade-offs
   - Fusion strategies (reciprocal rank fusion, etc.)
   - When keyword search outperforms semantic search
   - Implementation patterns in Python ecosystem
   - Cost implications of dual indexing

5. **Which architectures best handle wiki cross-references and relationships?**
   - Preserving hyperlink semantics during chunking
   - Metadata-based relationship encoding vs explicit graphs
   - Multi-hop retrieval for related topics
   - Context expansion strategies (retrieving linked chunks)
   - Comparative performance on relationship queries

6. **What are the cost/performance/complexity trade-offs?**
   - Embedding costs (one-time + updates)
   - Per-query retrieval costs (API calls, compute)
   - Implementation complexity ranking
   - Latency considerations for real-time Discord bot
   - Scalability from 100-page testbed to full wiki

7. **What retrieval evaluation metrics and benchmarks should be used?**
   - Relevance metrics (MRR, NDCG, Precision@K, Recall@K)
   - Faithfulness and groundedness measures
   - Answer quality evaluation frameworks
   - Wiki-specific benchmark datasets (if available)
   - How to construct evaluation dataset from WH40k lore

### Secondary Questions (Nice to Have)

8. **What emerging RAG techniques show promise for knowledge-intensive domains?**
   - Self-RAG (retrieval on demand)
   - CRAG (Corrective RAG)
   - Adaptive retrieval strategies
   - Query decomposition and planning
   - Iterative retrieval patterns

9. **How do different chunking strategies affect retrieval quality?**
   - Fixed-size vs semantic chunking
   - Section-based vs sliding window
   - Overlap strategies and their impact
   - Chunk size optimization for wiki content
   - Metadata extraction during chunking

10. **What vector databases are optimized for wiki-like workloads?**
    - Chroma vs Pinecone vs Weaviate vs Qdrant vs FAISS
    - Metadata filtering performance
    - Update/upsert efficiency for wiki refreshes
    - Python ecosystem integration
    - Self-hosted vs cloud cost comparisons

11. **Can LLMs be used to improve retrieval (query rewriting, expansion)?**
    - Query understanding and rewriting techniques
    - Multi-query generation strategies
    - Hypothetical document embeddings (HyDE)
    - Step-back prompting for broader context
    - Cost/benefit of LLM-in-the-loop retrieval

12. **What are best practices for handling speculation/canon separation in RAG?**
    - Metadata filtering during retrieval
    - Post-retrieval ranking by source authority
    - LLM-based confidence scoring
    - Presenting multiple perspectives in answers

---

## Research Methodology

### Information Sources

**Primary Academic Sources:**
- Recent papers on RAG architectures (2023-2025)
- NeurIPS, ACL, EMNLP conference proceedings
- arXiv preprints on retrieval and knowledge-intensive NLP
- RAPTOR original paper and follow-up work

**Technical Documentation & Frameworks:**
- LangChain documentation (RAG patterns)
- LlamaIndex architecture guides
- Haystack framework approaches
- Vector database vendor documentation
- Graph database integration guides (Neo4j, etc.)

**Industry Practice & Benchmarks:**
- Blog posts from AI labs (Anthropic, OpenAI, etc.)
- Production RAG case studies
- Open-source RAG implementations on GitHub
- Retrieval benchmark leaderboards (BEIR, MTEB)

**Community Knowledge:**
- Reddit r/LocalLLaMA, r/MachineLearning
- Hacker News discussions on RAG
- Discord communities (LangChain, AI engineering)
- Medium/Substack technical deep-dives

### Analysis Frameworks

**Comparison Matrix Dimensions:**
- Retrieval accuracy (qualitative and quantitative)
- Implementation complexity (LOC, dependencies, expertise needed)
- Cost structure (one-time setup, per-query, scaling)
- Latency (p50, p95, p99 response times)
- Suitability for wiki data (cross-references, structure)
- Python ecosystem maturity
- Maintenance burden

**Evaluation Criteria:**
- **Must-have:** Works with section-based chunking, supports rich metadata, Python-first
- **High priority:** Handles cross-references well, cost-efficient at scale, low latency (<2s p95)
- **Nice-to-have:** Active development community, cloud and self-hosted options, proven in production

**Architecture Classification:**
- **Naive RAG:** Simple vector search + top-k retrieval
- **Advanced RAG:** Query transformation, hybrid retrieval, re-ranking
- **Modular RAG:** Routing, multi-stage retrieval, fusion
- **Hierarchical RAG:** Tree/graph structures, multi-level abstraction
- **Agentic RAG:** LLM-driven retrieval decisions, iterative refinement

### Data Requirements

**Recency:** Prioritize 2023-2025 sources (RAG field evolving rapidly)
**Credibility:** Academic papers > technical documentation > vetted blog posts > forum discussions
**Specificity:** Prefer wiki/knowledge-base RAG case studies over generic document retrieval
**Actionability:** Focus on implemented approaches with code examples, not just theoretical proposals

---

## Expected Deliverables

### Executive Summary (1-2 pages)

**Key Findings:**
- Top 3-5 RAG architecture recommendations for wiki data
- RAPTOR suitability assessment (pros, cons, use cases)
- Graph-RAG feasibility verdict
- Hybrid retrieval (vector + BM25) trade-off summary
- Clear recommendation for 100-page testbed architecture

**Critical Implications:**
- Cost estimates for different architectures (setup + per-query)
- Implementation complexity ranking
- Latency expectations
- Scalability considerations

**Recommended Actions:**
- Specific architecture to prototype first
- Libraries/frameworks to use
- Evaluation metrics to track
- Decision criteria for production selection

### Detailed Analysis

#### Section 1: RAG Architecture Landscape
- Taxonomy of RAG approaches with definitions
- Evolution from naive to advanced/modular/agentic
- Current state-of-the-art (2025)
- Emerging trends and research directions

#### Section 2: RAPTOR Deep-Dive
- How RAPTOR works (methodology breakdown)
- Tree construction and abstraction process
- Query-time retrieval mechanism
- Implementation requirements and complexity
- Performance benchmarks and use cases
- Applicability to wiki data (strengths and weaknesses)
- Python implementation options

#### Section 3: Graph-RAG Approaches
- Knowledge graph construction methods
- Entity and relationship extraction
- Graph embedding techniques
- Hybrid vector-graph retrieval patterns
- Frameworks and tools (Neo4j, LangGraph, etc.)
- When graph-RAG outperforms pure vector search
- Cost and complexity considerations
- Simplified approaches (metadata "links" field vs full graph)

#### Section 4: Hybrid Retrieval (Vector + BM25)
- Dense vs sparse retrieval fundamentals
- Fusion strategies and algorithms
- When keyword search beats semantic search
- Implementation in Python (rank-bm25, Elasticsearch, etc.)
- Cost implications and trade-offs
- Recommended use cases for hybrid approach

#### Section 5: Wiki-Specific Considerations
- Handling hyperlinks and cross-references
- Section-based chunking best practices
- Metadata extraction and filtering
- Multi-hop retrieval for related topics
- Context expansion strategies
- Disambiguation techniques

#### Section 6: Cost-Performance-Complexity Matrix
- Detailed comparison table of architectures
- Embedding costs (per architecture)
- Per-query costs (retrieval + LLM)
- Implementation complexity ratings
- Latency benchmarks (if available)
- Scaling characteristics

#### Section 7: Evaluation Methodology
- Retrieval metrics explained (MRR, NDCG, Precision@K, Recall@K)
- Answer quality evaluation frameworks (DeepEval, RAGAS, etc.)
- How to construct WH40k lore evaluation dataset
- Benchmark design for 100-page testbed
- A/B testing strategies

#### Section 8: Implementation Guidance
- Recommended architecture for MVP/testbed
- Step-by-step implementation approach
- Python libraries and frameworks to use
- Common pitfalls and how to avoid them
- Iterative refinement strategy

### Supporting Materials

**Comparison Tables:**
- RAG architecture feature matrix
- Vector database comparison
- Cost breakdown by architecture
- Latency expectations table

**Code Examples:**
- Pseudo-code for key architectures
- Links to reference implementations
- Minimal working examples (where applicable)

**Source Documentation:**
- Annotated bibliography of key papers
- Framework documentation links
- GitHub repositories of interest
- Benchmark datasets

---

## Success Criteria

This research will be considered successful if it provides:

1. **Clear decision framework** for selecting RAG architecture based on project constraints
2. **Actionable recommendation** for which architecture to prototype in 100-page testbed
3. **Deep understanding** of RAPTOR and graph-RAG approaches (not just surface-level)
4. **Cost model** that predicts expenses for different architectures at scale
5. **Implementation roadmap** with specific tools, libraries, and steps
6. **Evaluation strategy** with metrics and benchmark design for testing
7. **Trade-off clarity** on accuracy vs cost vs complexity for each approach

---

## Timeline and Priority

**High Priority (Core to MVP):**
- Sections 1, 2, 3, 4, 6, 8 (architecture landscape, RAPTOR, graph-RAG, hybrid, cost matrix, implementation)
- Executive summary with clear recommendation

**Medium Priority (Informs testbed design):**
- Sections 5, 7 (wiki-specific considerations, evaluation methodology)

**Lower Priority (Nice to have):**
- Emerging techniques deep-dive
- Exhaustive vector DB comparison (can start with Chroma)

**Suggested Research Timeframe:** 4-8 hours of focused research + synthesis

---

## Next Steps After Research

### Execution Options

1. **Web Research with AI Assistant:**
   - Use this prompt with Claude, ChatGPT, or Perplexity for deep-dive research
   - Iterate on findings with follow-up questions
   - Synthesize into structured document

2. **Academic Literature Review:**
   - Search arXiv, Google Scholar for recent RAG papers
   - Focus on RAPTOR, graph-RAG, and wiki/knowledge-base retrieval
   - Extract implementation insights

3. **Hands-on Prototyping:**
   - After landscape understanding, build minimal examples
   - Test 2-3 promising architectures on small dataset
   - Measure actual performance vs theoretical claims

### Integration with Project

**Feeds into:**
- 100-page testbed implementation (Priority #1 from brainstorming session)
- Technical architecture decisions
- Cost modeling and LLM provider selection

**Validation Approach:**
- Implement top 2-3 architectures in testbed
- Run comparative evaluation with WH40k lore queries
- Measure retrieval quality, cost, latency
- Make data-driven final decision

**Documentation:**
- Update technical architecture docs with findings
- Record decision rationale for future reference
- Share learnings with community (if open-sourcing)

---

## Assumptions and Limitations

**Assumptions:**
- Python ecosystem preferred (not exploring Rust/Go alternatives)
- Focus on API-based LLMs (not local-only solutions)
- Wiki data structure relatively stable (sections, hyperlinks, infoboxes)
- 100-page testbed representative of full wiki characteristics

**Known Limitations:**
- RAG field evolving rapidly; findings may age quickly
- Benchmarks may not perfectly mirror WH40k lore query patterns
- Cost estimates are approximate until tested with real data
- Implementation complexity subjective (depends on developer expertise)

**Out of Scope:**
- Fine-tuning LLMs for domain-specific retrieval
- Building custom embedding models
- Non-English language retrieval optimization (separate from translation)
- Real-time wiki update detection and incremental indexing (deferred to post-MVP)

---

*Research prompt created using BMAD-METHOD™ framework*
