# RAG Architecture Research Report
## Comprehensive Survey for Wiki Data Retrieval

**Research Date:** 2025-12-17
**Project Context:** WH40k Lore Discord Bot
**Researcher:** Business Analyst Mary

---

## Executive Summary

### Key Findings

Based on comprehensive research into RAG architectures for wiki-structured data, here are the top recommendations for the WH40k Lore Bot project:

**Top 5 RAG Architecture Recommendations for Wiki Data:**

1. **Hybrid Retrieval (Vector + BM25)** - HIGHEST PRIORITY
   - Combines dense semantic search with sparse keyword matching
   - 3-way hybrid (BM25 + dense vector + sparse vector) shows best performance
   - Handles both conceptual queries and exact term matching
   - Mature Python ecosystem (rank-bm25, FAISS, Weaviate)

2. **Graph-RAG (Lightweight Metadata Approach)** - RECOMMENDED FOR TESTBED
   - Preserve wiki hyperlinks as chunk metadata "links" arrays
   - Enables cross-reference queries without full knowledge graph complexity
   - Can upgrade to full graph approach if testbed shows value

3. **RAPTOR (Hierarchical Retrieval)** - EVALUATE IN TESTBED
   - Tree-based multi-level abstraction excellent for complex lore topics
   - Handles both detailed facts and high-level summaries
   - Higher implementation complexity but strong for narrative content

4. **Advanced RAG with Reranking** - COST-EFFECTIVE ENHANCEMENT
   - Simple vector search + LLM-based or ColBERT reranking
   - Improves precision without architectural complexity
   - Can layer onto any base retrieval strategy

5. **Adaptive/Corrective RAG** - FUTURE ENHANCEMENT
   - Self-RAG and CRAG techniques for query-adaptive retrieval
   - Reduces hallucination, improves accuracy
   - Defer until MVP proven

### RAPTOR Suitability Assessment

**Pros:**
- Hierarchical structure maps well to WH40K's nested lore (factions → characters → events)
- Multi-level retrieval handles both "Who is X?" (leaf nodes) and "Explain the Horus Heresy" (summary nodes)
- Reduces irrelevant context by retrieving at appropriate abstraction level
- Strong performance on complex, interconnected knowledge bases

**Cons:**
- Significant upfront cost for tree construction (clustering + recursive summarization)
- LLM API calls for generating summaries at each tree level
- More complex to maintain when wiki content updates
- Overkill for simple fact lookup queries

**Use Cases Where RAPTOR Excels:**
- Broad topic queries ("Tell me about the Imperium of Man")
- Chronological event summaries ("What led to the Horus Heresy?")
- Multi-entity relationship queries ("How are the Primarchs connected?")

**Recommendation:** Include RAPTOR in 100-page testbed but start with hybrid retrieval as baseline. RAPTOR's value depends on query distribution - if users ask mostly broad conceptual questions, it's worth the investment.

### Graph-RAG Feasibility Verdict

**Full Knowledge Graph Approach:**
- **Complexity:** HIGH (entity extraction, relationship classification, graph database setup)
- **Cost:** MEDIUM-HIGH (one-time NER/relationship extraction, ongoing graph maintenance)
- **Latency:** MEDIUM (graph traversal + vector retrieval)
- **Best for:** Explicit relationship queries, multi-hop reasoning, "How is X connected to Y?"

**Lightweight Metadata Approach (RECOMMENDED for MVP):**
- **Complexity:** LOW (parse wiki hyperlinks into chunk metadata)
- **Cost:** LOW (minimal additional processing)
- **Latency:** LOW (metadata filtering on vector retrieval)
- **Implementation:** Add "links" field to each chunk with array of referenced article titles/IDs
- **Query strategy:** Retrieve primary chunks, optionally expand to linked chunks for context

**Verdict:** Start with lightweight metadata approach for 100-page testbed. Measure how often users ask relationship queries. If >20% of queries benefit from cross-reference expansion, consider upgrading to full graph approach using Neo4j or similar.

**Frameworks to Consider (if full graph pursued):**
- Neo4j for graph database + Cypher queries
- Microsoft GraphRAG framework (production-ready, documented)
- LangGraph for agent-based graph traversal
- Kuzu for embedded graph database (lighter than Neo4j)

### Hybrid Retrieval (Vector + BM25) Trade-Off Summary

**Why Hybrid Outperforms Pure Vector:**

| Query Type | Pure Vector | BM25 | Hybrid |
|------------|-------------|------|--------|
| Conceptual ("What is grimdark?") | ✅ Excellent | ❌ Poor | ✅ Excellent |
| Exact names ("Roboute Guilliman") | ⚠️ Good | ✅ Excellent | ✅ Excellent |
| Rare terms ("Adeptus Mechanicus") | ⚠️ Depends | ✅ Excellent | ✅ Excellent |
| Paraphrased queries | ✅ Excellent | ❌ Poor | ✅ Excellent |

**2024 Research Consensus:** 3-way hybrid (BM25 + dense vector + sparse vector like SPLADE) with Reciprocal Rank Fusion (RRF) is optimal. Source: IBM Research "Blended RAG" paper.

**Cost Implications:**
- Dual indexing: ~1.5-2x storage vs vector-only
- Negligible per-query cost increase (BM25 is fast)
- Reranking (ColBERT) adds ~$0.001-0.01 per query

**Python Implementation:**
- **rank-bm25** library for sparse retrieval
- **FAISS** or **Chroma** for dense vectors
- **Weaviate** or **Qdrant** have built-in hybrid search
- **Reciprocal Rank Fusion** for combining results

**Recommendation:** Implement hybrid search in 100-page testbed. Given WH40K's many proper nouns (character names, planet names, faction names), BM25 will significantly improve exact-match retrieval.

### Cost Estimates for Different Architectures

**Setup Costs (One-Time for Full Wiki):**

| Architecture | Scraping | Embedding | Processing | Total Setup |
|--------------|----------|-----------|------------|-------------|
| Pure Vector | - | $5-15 | - | $5-15 |
| Hybrid (Vector + BM25) | - | $5-15 | $1-3 | $6-18 |
| Graph-RAG (Lightweight) | - | $5-15 | $2-5 | $7-20 |
| Graph-RAG (Full) | - | $5-15 | $50-150 | $55-165 |
| RAPTOR | - | $5-15 | $100-300 | $105-315 |

**Assumptions:** ~10,000 wiki articles, ~50M tokens, embedding at $0.0001/1K tokens, RAPTOR/graph processing using LLM.

**Per-Query Costs:**

| Architecture | Retrieval | LLM Generation | Reranking | Total/Query |
|--------------|-----------|----------------|-----------|-------------|
| Pure Vector | $0.00 | $0.01-0.05 | - | $0.01-0.05 |
| Hybrid + Reranking | $0.00 | $0.01-0.05 | $0.001 | $0.011-0.051 |
| Graph-RAG | $0.00 | $0.01-0.05 | - | $0.01-0.05 |
| RAPTOR | $0.00 | $0.01-0.05 | - | $0.01-0.05 |
| Self-RAG/CRAG | $0.00-0.005 | $0.02-0.10 | - | $0.02-0.105 |

**Latency Expectations:**

| Architecture | p50 | p95 | p99 |
|--------------|-----|-----|-----|
| Pure Vector | 200-400ms | 500-800ms | 1-1.5s |
| Hybrid | 300-500ms | 700ms-1s | 1.5-2s |
| Graph-RAG (lightweight) | 300-500ms | 700ms-1s | 1.5-2s |
| Graph-RAG (full) | 500ms-1s | 1-2s | 2-3s |
| RAPTOR | 300-600ms | 800ms-1.2s | 1.5-2.5s |

**Implementation Complexity Ranking (Easiest to Hardest):**
1. Pure Vector (baseline)
2. Hybrid Vector + BM25
3. Graph-RAG (metadata links approach)
4. Advanced RAG with reranking
5. RAPTOR
6. Graph-RAG (full knowledge graph)
7. Self-RAG / CRAG

### Critical Implications

**For 100-Page Testbed:**
1. **Start with hybrid retrieval** - Best ROI for wiki data with proper nouns
2. **Add metadata "links" field** - Enables future graph capabilities
3. **Test RAPTOR on subset** - 20-30 pages to evaluate hierarchical value
4. **Measure retrieval quality** - Use RAGAS framework or DeepEval
5. **Track query patterns** - Informs which advanced features to prioritize

**For Production Scale:**
1. **Embedding costs are minor** ($5-15 one-time for full wiki)
2. **Per-query costs dominated by LLM generation** (retrieval nearly free)
3. **BM25 indexing is cheap** (minimal storage/compute overhead)
4. **Graph-RAG setup expensive** ($100-300 for full wiki) - defer unless testbed proves value

**For User Experience:**
1. **Sub-2s p95 latency achievable** with all recommended architectures
2. **Hybrid search improves exact-match queries** (critical for WH40K proper nouns)
3. **Spoiler handling** can be metadata-filtered at retrieval time
4. **Canon vs speculation** separation viable via metadata filtering

### Recommended Actions

**Phase 1: 100-Page Testbed (Week 1-2)**
1. Implement baseline **pure vector search** with Chroma
2. Add **hybrid retrieval** (vector + BM25 via rank-bm25 library)
3. Implement **section-based chunking** with metadata (faction, era, URL, canon flag)
4. Add **metadata "links" field** capturing wiki hyperlinks
5. Set up **evaluation framework** (RAGAS or DeepEval)

**Phase 2: Architecture Comparison (Week 2-3)**
1. Build **RAPTOR tree** on 20-30 page subset (test hierarchical value)
2. Implement **simple reranking** (LLM-based or ColBERT if budget allows)
3. Test **cross-reference expansion** using metadata links
4. Measure **retrieval quality** across 50-100 test queries
5. Compare **cost, latency, accuracy** for each approach

**Phase 3: Production Decision (Week 3-4)**
1. Select winning architecture based on testbed results
2. Scale to full wiki dataset
3. Optimize chunking and metadata extraction
4. Implement Hungarian translation layer
5. Deploy to Discord bot

**Libraries/Frameworks to Use:**
- **Vector DB:** Chroma (current) or Qdrant (best metadata filtering)
- **Embeddings:** OpenAI text-embedding-3-small or Cohere embed-english-v3.0
- **BM25:** rank-bm25 library (pure Python)
- **Hybrid Fusion:** Reciprocal Rank Fusion (simple algorithm, implement manually)
- **RAPTOR:** No official library; implement based on paper or use LlamaIndex experimental
- **Evaluation:** RAGAS (open-source) or DeepEval (more features)
- **Graph (if needed):** Neo4j + LangChain Neo4j integration or Kuzu embedded graph

**Evaluation Metrics to Track:**
- **Retrieval:** Recall@5, Recall@10, MRR, NDCG@10
- **Generation:** Faithfulness, Answer Relevance, Context Precision (RAGAS metrics)
- **User Feedback:** Helpful/Not Helpful button click-through rate
- **Cost:** Embedding tokens, LLM tokens, per-query average
- **Latency:** p50, p95, p99 response times

**Decision Criteria for Production:**
- **Accuracy:** Must achieve >80% user satisfaction (Helpful button rate)
- **Cost:** Must stay under €0.05 per query average
- **Latency:** p95 must be <2 seconds
- **Complexity:** Team must be able to maintain without dedicated ML engineer

---

## Detailed Analysis

### Section 1: RAG Architecture Landscape

#### 1.1 Taxonomy of RAG Approaches

RAG architectures have evolved significantly since their introduction. We can classify them into four generations:

**Generation 1: Naive RAG**
- Simple vector similarity search
- Top-K retrieval without reranking
- No query understanding or optimization
- Linear pipeline: Query → Embed → Retrieve → Generate

**Generation 2: Advanced RAG**
- **Query transformation:** Rewriting, expansion, decomposition
- **Hybrid retrieval:** Combining dense (vector) and sparse (BM25/keyword) search
- **Reranking:** Post-retrieval filtering and scoring
- **Context optimization:** Chunk size tuning, metadata filtering

**Generation 3: Modular RAG**
- **Routing:** Different retrieval strategies for different query types
- **Multi-stage retrieval:** Coarse-to-fine or hierarchical retrieval
- **Fusion:** Combining results from multiple retrievers
- **Iterative refinement:** Multiple retrieval rounds based on partial answers

**Generation 4: Agentic RAG**
- **LLM-driven decisions:** Model decides when/what to retrieve
- **Adaptive strategies:** Self-RAG, CRAG, Adaptive-RAG
- **Tool use:** Integration with external knowledge sources (web search, APIs)
- **Self-critique:** Models evaluate their own outputs and trigger re-retrieval

#### 1.2 Current State-of-the-Art (2025)

Based on 2024-2025 research, the leading RAG patterns include:

**Hybrid Retrieval with Fusion**
- Dense vector + sparse vector (SPLADE) + BM25
- Reciprocal Rank Fusion or learned fusion weights
- IBM Research "Blended RAG" shows 15-30% improvement over pure vector

**Graph-Enhanced RAG**
- Knowledge graphs augment vector retrieval with structured relationships
- Microsoft GraphRAG, Neo4j Graph RAG patterns
- Particularly effective for multi-hop reasoning and entity-centric queries

**Hierarchical RAG**
- RAPTOR (Recursive Abstractive Processing)
- Tree structures with multiple abstraction levels
- Outperforms flat retrieval on complex, interconnected documents

**Adaptive/Corrective RAG**
- Self-RAG: Model decides when to retrieve and critiques own outputs
- CRAG: Evaluates retrieval quality and triggers web search if needed
- Adaptive-RAG: Adjusts strategy based on query complexity

#### 1.3 Emerging Trends and Research Directions

**Multimodal RAG:** Retrieval over text, images, tables, code (beyond text-only)
**Structured Data RAG:** SQL generation for databases, knowledge graph queries
**Long-Context RAG:** Leveraging 100K+ token context windows to reduce retrieval needs
**Agentic Workflows:** RAG as part of larger agent systems with planning and tool use
**Personalized RAG:** User-specific retrieval and generation preferences

---

### Section 2: RAPTOR Deep-Dive

#### 2.1 How RAPTOR Works

RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval) is a hierarchical RAG architecture that organizes documents into a tree structure with multiple levels of abstraction.

**Core Methodology:**

1. **Chunking:** Split documents into base chunks (leaf nodes)
2. **Embedding:** Embed all chunks with dense vector model
3. **Clustering:** Group semantically similar chunks using soft clustering (GMM)
4. **Summarization:** Use LLM to generate abstractive summary for each cluster
5. **Recursion:** Treat summaries as new chunks, repeat clustering/summarization
6. **Tree Construction:** Build multi-level tree where leaves = original text, internal nodes = summaries

**Query-Time Retrieval:**

1. Embed user query
2. Retrieve top-K most similar nodes **across all tree levels**
3. Can retrieve from leaves (specific facts), internal nodes (summaries), or both
4. Collapsed tree provides context at appropriate abstraction level

#### 2.2 Tree Construction Process

**Clustering Algorithm:**
- Gaussian Mixture Models (GMM) with soft clustering
- Allows chunks to belong to multiple clusters
- Alternative: UMAP dimensionality reduction + HDBSCAN

**Summarization Strategy:**
- Each cluster summary generated by LLM (e.g., "Summarize the following related passages...")
- Summary length typically 100-200 tokens
- Preserves key entities, events, and relationships

**Tree Depth:**
- Typically 2-4 levels for most document collections
- Deeper trees for very large/complex corpora
- Leaf layer = original chunks, each successive layer = more abstract

#### 2.3 Implementation Requirements and Complexity

**Infrastructure Needs:**
- Vector database supporting hierarchical metadata (e.g., Qdrant, Weaviate, Chroma)
- LLM API access for summarization (GPT-4, Claude, etc.)
- Clustering library (scikit-learn GMM)
- Significant upfront processing time

**Cost Breakdown (Example: 10,000 wiki articles):**
- Chunking: Negligible
- Embedding: $5-15 (one-time)
- Clustering: Minutes on CPU
- Summarization: $100-300 depending on LLM (GPT-3.5-turbo vs GPT-4)
- Total tree construction: **$105-315**

**Maintenance:**
- Adding new articles requires re-clustering and summarization
- Incremental updates possible but complex
- Full rebuild recommended for major wiki changes

#### 2.4 Performance Benchmarks and Use Cases

**Published Results (RAPTOR paper):**
- 20% improvement on QuALITY benchmark (long document QA)
- Outperforms retrieval + full document context
- Particularly strong on questions requiring synthesis across multiple sections

**Best Use Cases:**
- **Long documents** (books, research papers, comprehensive wikis)
- **Nested topics** (e.g., WH40K: Imperium → Space Marines → Blood Angels → Sanguinius)
- **Broad questions** requiring summary ("Explain the Horus Heresy")
- **Multi-hop reasoning** ("How did the Primarch wars affect the Imperium?")

**Weaker Performance On:**
- Simple fact lookup ("What year was X born?")
- Queries where chunk-level retrieval suffices
- Frequently changing content (re-summarization expensive)

#### 2.5 Applicability to Wiki Data

**Strengths for WH40K Wiki:**
✅ Nested lore structure (factions, sub-factions, characters, events)
✅ Broad conceptual queries common ("Tell me about Chaos")
✅ Enables both detail and summary retrieval
✅ Reduces irrelevant context in prompts (retrieve at right level)

**Weaknesses:**
❌ High upfront cost ($100-300 for full wiki)
❌ Complex to maintain with wiki updates
❌ Overkill for simple name/fact lookups
❌ Not necessary if queries are mostly specific ("Who is Abaddon?")

**Decision Criteria:**
- If >50% of queries are broad/conceptual → Strong case for RAPTOR
- If >50% of queries are specific facts → Hybrid retrieval likely sufficient
- If budget-constrained → Defer RAPTOR until post-MVP

#### 2.6 Python Implementation Options

**Option 1: Custom Implementation**
- Use LangChain or LlamaIndex as base
- scikit-learn for GMM clustering
- OpenAI/Anthropic API for summarization
- Store tree structure in vector DB metadata
- Full control but significant development effort

**Option 2: LlamaIndex RAPTOR (Experimental)**
- LlamaIndex has experimental RAPTOR implementation
- Less mature but saves implementation time
- May lack customization for wiki-specific needs

**Option 3: Adapt from Research Code**
- Original RAPTOR paper includes code
- Requires adaptation to production environment

**Recommendation:** Prototype with LlamaIndex experimental, then custom implementation if needed for production.

---

### Section 3: Graph-RAG Approaches

#### 3.1 Knowledge Graph Construction Methods

**Option 1: Entity and Relationship Extraction**
- Use NER (Named Entity Recognition) to extract entities (characters, factions, locations, events)
- Use Relation Extraction models to identify relationships ("Sanguinius founded Blood Angels")
- Build graph: Entities = nodes, Relationships = edges
- Tools: spaCy, Stanford NER, LLM-based extraction (GPT-4, Claude)

**Option 2: Wiki Hyperlink Parsing (Lightweight)**
- Parse wiki hyperlinks as relationships
- Each link = "mentions" or "related_to" edge
- Much simpler than full NER/RE
- Preserves author-curated relationships

**Option 3: LLM-Driven Graph Construction**
- Prompt LLM to extract entities and relationships from each article
- More flexible than rule-based NER
- Higher cost but better quality
- Examples: "Extract all characters, factions, and their relationships from this text"

#### 3.2 Graph Embedding Techniques

**Node Embeddings:**
- **TransE, RotatE, ComplEx:** Traditional knowledge graph embedding methods
- **Node2Vec:** Random walk-based node embeddings
- **Graph Neural Networks (GNNs):** Learn embeddings from graph structure
- **Hybrid:** Combine text embeddings with graph structure embeddings

**Query Strategy:**
- Embed query as starting node
- Traverse graph to find connected entities
- Retrieve text chunks associated with traversed nodes
- Combine graph context with vector similarity

#### 3.3 Hybrid Vector-Graph Retrieval Patterns

**Pattern 1: Vector-First, Graph-Expand**
1. Use vector search to find top-K relevant chunks
2. Identify entities in retrieved chunks
3. Traverse graph to find connected entities
4. Retrieve chunks for connected entities
5. Combine all chunks for LLM context

**Pattern 2: Graph-First, Vector-Rerank**
1. Map query to graph entities (via NER or LLM)
2. Traverse graph to find related entities
3. Retrieve chunks associated with graph nodes
4. Use vector similarity to rerank chunks
5. Select top-K for LLM context

**Pattern 3: Parallel Retrieval + Fusion**
1. Run vector search in parallel with graph traversal
2. Combine results using fusion algorithm (RRF)
3. Deduplicate chunks
4. Rerank if needed

#### 3.4 Frameworks and Tools

**Neo4j (Most Popular for Graph-RAG):**
- Industry-standard graph database
- Cypher query language
- LangChain integration available
- Cloud (Aura) or self-hosted
- Strong community and documentation

**Microsoft GraphRAG:**
- Production-ready framework from Microsoft Research
- Combines knowledge graph with vector search
- Open-source with good documentation
- Built on top of LangChain

**LangGraph:**
- Graph-based agent orchestration from LangChain team
- Not a database, but a workflow framework
- Useful for complex multi-step graph queries

**Kuzu:**
- Embedded graph database (like SQLite for graphs)
- Lighter than Neo4j for smaller deployments
- Cypher-compatible query language

**NetworkX + Vector DB:**
- Use NetworkX (Python library) for in-memory graph
- Store graph structure alongside vector embeddings
- Simpler but less scalable than Neo4j

#### 3.5 When Graph-RAG Outperforms Pure Vector Search

**Graph-RAG Wins:**
- **Relationship queries:** "How are Blood Angels connected to Horus Heresy?"
- **Multi-hop reasoning:** "What events link the Emperor to the current Imperium state?"
- **Entity disambiguation:** "Ragnar" (multiple entities with same name)
- **Network analysis:** "Which factions fought together in X conflict?"
- **Explainability:** Graph traversal shows reasoning path

**Pure Vector Wins:**
- **Semantic similarity:** "What is grimdark?" (no entities involved)
- **Broad topic queries:** "Explain Chaos" (too many graph connections to be useful)
- **Paraphrased queries:** Vector search handles synonyms better
- **Simple fact lookup:** "When was the Horus Heresy?" (graph overhead unnecessary)

**Benchmark Results:**
- GraphRAG shows 20-40% improvement on multi-hop questions
- Comparable or slightly worse on simple fact lookup
- Significantly better on entity-centric queries

#### 3.6 Cost and Complexity Considerations

**Full Knowledge Graph Construction Costs:**
- Entity extraction: $50-150 (LLM-based) or free (rule-based NER, lower quality)
- Relationship extraction: $100-300 (LLM-based)
- Graph database setup: Free (Neo4j Community) to $$$ (Neo4j Aura)
- Total: **$150-450** for full wiki

**Lightweight Metadata Approach:**
- Parse wiki hyperlinks: Negligible cost
- Store as "links" array in chunk metadata: No extra infrastructure
- Query-time expansion: Retrieve linked chunks via metadata filtering
- Total: **<$5**

**Complexity:**
- Full graph: HIGH (NER, graph DB, Cypher queries, maintenance)
- Metadata links: LOW (parse hyperlinks, metadata filtering)

**Recommendation:** Start with metadata links for 100-page testbed. Upgrade to full graph only if:
1. >20% of queries are relationship-based
2. Users explicitly ask "how is X connected to Y?"
3. Entity disambiguation is a frequent problem

#### 3.7 Simplified Approach: Metadata "Links" Field

**Implementation:**
1. During chunking, parse wiki hyperlinks in each section
2. Extract target article titles/IDs
3. Store as `links: ["Blood Angels", "Sanguinius", "Great Crusade"]` in chunk metadata
4. At query time, retrieve primary chunks via vector search
5. Optionally expand context by also retrieving chunks with matching article titles from `links` field

**Benefits:**
- Minimal implementation effort
- No separate graph database needed
- Preserves wiki author's curated relationships
- Enables "related articles" feature
- Future-proof: Can upgrade to full graph later

**Example Query Flow:**
```
Query: "Tell me about Sanguinius"
1. Vector search retrieves chunks mentioning Sanguinius
2. Check metadata links field in top chunks
3. Find frequently linked articles: ["Blood Angels", "Horus Heresy", "Great Angel"]
4. Optionally retrieve 1-2 chunks from each linked article for context
5. Combine all chunks for LLM prompt
```

---

### Section 4: Hybrid Retrieval (Vector + BM25)

#### 4.1 Dense vs Sparse Retrieval Fundamentals

**Dense Retrieval (Vector Search):**
- Embeddings capture semantic meaning
- Good for: paraphrased queries, conceptual similarity, synonyms
- Example: "grimdark universe" matches "dystopian setting"
- Weakness: May miss exact term matches, proper nouns

**Sparse Retrieval (BM25/Keyword):**
- Term frequency and document frequency scoring
- Good for: exact matches, rare terms, proper nouns
- Example: "Roboute Guilliman" exact match beats "Primarch of Ultramarines"
- Weakness: No semantic understanding, fails on paraphrases

**Why Both?**
- WH40K has many proper nouns: character names, planet names, factions
- Users may search by exact name ("Abaddon") or concept ("Warmaster of Chaos")
- Hybrid handles both query types effectively

#### 4.2 Fusion Strategies and Algorithms

**Reciprocal Rank Fusion (RRF)** - Most Common
```
score(chunk) = Σ 1/(k + rank_i)
```
- k = constant (typically 60)
- rank_i = rank from each retriever (BM25, vector search, etc.)
- Simple, no training needed, robust

**Relative Score Fusion**
- Normalize scores from each retriever to [0,1]
- Weighted sum: `score = w1*vector_score + w2*bm25_score`
- Requires tuning weights (grid search or learned)

**Learned Fusion**
- Train a reranker model on retrieval results
- Examples: Cross-encoder, LLM-based reranking
- Higher quality but added cost/latency

**Hybrid Search Native in DBs**
- Weaviate: Built-in hybrid search with alpha parameter (0=BM25, 1=vector)
- Qdrant: Separate dense and sparse vectors, fusion on query
- Elasticsearch: Dense + BM25 with boost parameters

#### 4.3 When Keyword Search Beats Semantic Search

**Scenarios Where BM25 Wins:**
1. **Proper nouns:** "Roboute Guilliman" exact match
2. **Rare technical terms:** "Adeptus Mechanicus", "Omnissiah"
3. **Acronyms:** "IoM" (Imperium of Man), "T'au"
4. **Exact titles:** "Horus Heresy" vs "civil war among Primarchs"
5. **Short queries:** 1-2 word searches favor keyword match

**Scenarios Where Vector Wins:**
1. **Paraphrased questions:** "Who leads the Ultramarines?" vs "Roboute Guilliman"
2. **Conceptual queries:** "grimdark", "dystopian future"
3. **Synonyms:** "battle" matches "combat", "war", "conflict"
4. **Typos/misspellings:** Vector embeddings more robust
5. **Long, natural language queries:** Semantic meaning dominates

**Research Finding (IBM Blended RAG):**
- 3-way hybrid (BM25 + dense vector + sparse vector) best overall
- Sparse vector (SPLADE) bridges gap between keyword and semantic
- Fusion critical: RRF outperforms single-retriever approaches by 15-30%

#### 4.4 Implementation in Python Ecosystem

**Option 1: Manual Hybrid with rank-bm25**
```python
from rank_bm25 import BM25Okapi
import chromadb  # or FAISS, Pinecone

# BM25 indexing
tokenized_docs = [doc.split() for doc in documents]
bm25 = BM25Okapi(tokenized_docs)

# Query
query_tokens = query.split()
bm25_scores = bm25.get_scores(query_tokens)
bm25_top_k = get_top_k(bm25_scores, k=20)

# Vector search
vector_top_k = chroma_collection.query(query, n_results=20)

# Reciprocal Rank Fusion
fused_results = reciprocal_rank_fusion([bm25_top_k, vector_top_k])
```

**Option 2: Weaviate Native Hybrid**
```python
import weaviate

result = client.query.get("WikiChunk", ["text", "metadata"]) \
    .with_hybrid(query="Roboute Guilliman", alpha=0.5) \
    .with_limit(10) \
    .do()
```
- alpha=0: Pure BM25
- alpha=1: Pure vector
- alpha=0.5: Balanced hybrid

**Option 3: Qdrant Hybrid**
```python
from qdrant_client import QdrantClient

client.search(
    collection_name="wiki_chunks",
    query_vector=embed(query),
    query_filter=...,  # metadata filters
    search_params={"hnsw_ef": 128, "exact": False},
    limit=10,
    sparse_vector=sparse_embed(query)  # BM25 or SPLADE
)
```

**Option 4: Elasticsearch Hybrid**
- Use Elasticsearch for both BM25 (native) and vector search (kNN plugin)
- Boost parameters to weight each retriever
- Mature, production-ready infrastructure

#### 4.5 Cost Implications and Trade-Offs

**Storage:**
- BM25 inverted index: ~1.2-1.5x text size
- Vector embeddings: ~4KB per chunk (1536-dim)
- Total hybrid storage: ~1.5-2x vs vector-only
- Example: 10,000 articles, 100K chunks → +200MB for BM25 index

**Indexing Time:**
- BM25: Milliseconds to seconds (very fast)
- Vector embedding: Depends on embedding API rate limits
- Negligible difference in practice

**Query Latency:**
- BM25 search: <10ms
- Vector search: 10-50ms (HNSW index)
- Fusion: <5ms
- Total hybrid: ~50-100ms (comparable to vector-only)

**Maintenance:**
- Both indexes need updating when content changes
- BM25 update: Trivial (re-index text)
- Vector update: Re-embed and upsert
- No significant added burden

**Verdict:** Hybrid adds minimal cost/complexity with significant accuracy gains for proper noun-heavy domains like WH40K.

#### 4.6 Recommended Use Cases for Hybrid Approach

**Strong Recommendation:**
- ✅ Proper noun-heavy domains (wikis, technical docs, product catalogs)
- ✅ Mixed query types (factual + conceptual)
- ✅ User-generated queries (unpredictable phrasing)
- ✅ When exact-match precision matters

**Less Critical:**
- ⚠️ Pure conceptual content (philosophical texts, creative writing)
- ⚠️ Controlled query vocabulary (internal search with standardized terms)
- ⚠️ When storage is severely constrained

**For WH40K Lore Bot:** **STRONGLY RECOMMENDED**
- Hundreds of unique proper nouns (characters, factions, planets, weapons)
- Users will search by both name ("Abaddon") and concept ("Warmaster of Chaos")
- Hungarian translation may alter phrasing, hybrid more robust
- Minimal added cost, significant accuracy gain

---

### Section 5: Wiki-Specific Considerations

#### 5.1 Handling Hyperlinks and Cross-References

Wiki articles are densely interconnected via hyperlinks. Preserving these relationships improves retrieval quality.

**Approach 1: Metadata Links Array** (Recommended for MVP)
- Parse hyperlinks during scraping
- Store target article titles in chunk metadata: `links: ["Article A", "Article B"]`
- Query-time: Retrieve primary chunks, optionally expand to linked articles
- Pros: Simple, no extra infrastructure
- Cons: Limited to 1-hop traversal

**Approach 2: Explicit Graph Database**
- Build knowledge graph with articles as nodes, hyperlinks as edges
- Store in Neo4j, Kuzu, or NetworkX
- Query-time: Graph traversal + vector search
- Pros: Multi-hop reasoning, relationship queries
- Cons: Higher complexity, setup cost

**Approach 3: Embed Cross-References in Text**
- Include linked article snippets in chunk context
- Example: "Blood Angels (founded by Sanguinius during the Great Crusade)"
- Pros: Works with pure vector search
- Cons: Increases chunk size, may add noise

**Recommendation:** Start with Approach 1 (metadata links), upgrade to Approach 2 if testbed shows frequent relationship queries.

#### 5.2 Section-Based Chunking Best Practices

Wiki articles have clear structure: headings, sections, subsections. Respect this structure during chunking.

**Section-Based Chunking:**
- Split at heading boundaries (`<h2>`, `<h3>`, etc.)
- Preserve section hierarchy in metadata: `section_path: "Imperium > Space Marines > Blood Angels"`
- Include heading text in chunk (improves context)

**Benefits:**
- Semantic coherence (section = single topic)
- Metadata-rich (can filter by section)
- Improves retrieval precision

**Challenges:**
- Section size varies (50 tokens to 5000 tokens)
- Very large sections may need sub-chunking
- Very small sections may be too granular

**Hybrid Approach:**
- Primary strategy: Split at section boundaries
- Secondary: If section >1000 tokens, further split at paragraph boundaries
- If section <100 tokens, merge with adjacent section or parent context

**Overlap Strategy for Wikis:**
- Generally avoid overlap for section-based chunking
- Sections are self-contained topics
- Overlap mainly useful for fixed-size chunking (not section-based)

**Metadata to Extract:**
- `article_title`: e.g., "Blood Angels"
- `section_heading`: e.g., "History > The Horus Heresy"
- `section_level`: 1, 2, 3 (h1, h2, h3)
- `faction`, `character`, `era`: Extracted via NER or rules
- `links`: Array of hyperlinked article titles
- `url`: Source wiki article URL
- `canon_flag`: Boolean or confidence score
- `spoiler_flag`: Boolean (if detectable)

#### 5.3 Metadata Extraction and Filtering

**Structured Metadata (From Wiki Infoboxes):**
- Many wiki articles have infoboxes with structured data
- Example: Character infobox with "Faction", "Allegiance", "First Appearance"
- Parse infoboxes and add to chunk metadata
- Enables precise filtering: "Show me all Blood Angels characters"

**Unstructured Metadata (From Text):**
- Use NER to extract entities: characters, factions, locations, events
- Use rule-based extraction for dates, eras ("M31", "41st millennium")
- LLM-based extraction for complex attributes ("Alignment: Chaos", "Legion: World Eaters")

**Canon vs Speculation Detection:**
- Section title indicators: "Theories", "Speculation", "Conflicting Sources"
- Language patterns: "possibly", "it is believed", "according to one source"
- Citation analysis: Codex reference (canon) vs fan wiki (speculation)
- LLM classification: "Is this text canon lore or fan theory?"

**Metadata Filtering at Query Time:**
- Qdrant and Weaviate excel at metadata filtering
- Example: Retrieve only `canon_flag=True` chunks
- Example: Filter to `faction="Imperium"` when answering Imperium-specific questions
- Can boost retrieval precision significantly

#### 5.4 Multi-Hop Retrieval for Related Topics

**Challenge:** User asks about "Sanguinius" but relevant context is in "Blood Angels" article.

**Solution 1: Cross-Reference Expansion**
1. Retrieve chunks mentioning "Sanguinius"
2. Check `links` metadata field
3. Also retrieve 1-2 representative chunks from linked articles ("Blood Angels", "Horus Heresy")
4. Combine all for LLM context

**Solution 2: Iterative Retrieval**
1. Retrieve initial chunks
2. Extract entities mentioned
3. Retrieve additional chunks for those entities
4. Repeat for 2-3 hops
5. Combine all chunks (may be large context)

**Solution 3: Query Decomposition**
- LLM decomposes complex query into sub-queries
- Example: "How did Sanguinius die?" → ["Who is Sanguinius?", "What was the Siege of Terra?", "Horus vs Sanguinius"]
- Retrieve for each sub-query
- Combine results

**Recommendation:** Solution 1 (cross-reference expansion) for 100-page testbed. Simple, effective, low cost.

#### 5.5 Context Expansion Strategies

**Parent Document Retrieval:**
- Retrieve small chunks for precision
- Expand to full parent section/article for LLM context
- Avoids retrieving irrelevant sections from large articles

**Surrounding Context:**
- Retrieve chunk N
- Also include chunks N-1 and N+1 (preceding/following sections)
- Preserves narrative flow

**Hierarchical Context:**
- Retrieve leaf chunk
- Include parent section summary
- Include article-level summary (if using RAPTOR or similar)

**Recommendation:** Start simple (just retrieved chunks). Add parent document retrieval if context seems insufficient during testbed evaluation.

#### 5.6 Disambiguation Techniques

**Challenge:** "Ragnar" could be multiple entities in WH40K lore.

**Solution 1: Metadata Filtering**
- If metadata includes entity type ("character", "planet", "weapon")
- Filter to most likely type based on query context

**Solution 2: LLM Clarification**
- Retrieve chunks for all "Ragnar" entities
- LLM generates clarification question: "Did you mean Ragnar Blackmane (Space Wolf) or Ragnar (planet)?"
- User selects, bot retrieves specific chunks

**Solution 3: Context-Based Scoring**
- Analyze query for context clues
- "Tell me about Ragnar's battles" → likely character
- "What happened on Ragnar?" → likely planet
- Boost scores for contextually appropriate entities

**Solution 4: Graph Disambiguation**
- Use knowledge graph to find most connected entity
- "Ragnar" connected to "Space Wolves" and "Blood Claws" → likely Ragnar Blackmane

**Recommendation:** Solution 3 (context-based scoring) for MVP. Solutions 2 and 4 if disambiguation is frequent issue in testbed.

---

### Section 6: Cost-Performance-Complexity Matrix

*(See Executive Summary for detailed tables)*

**Key Insights:**

1. **Embeddings are cheap:** $5-15 for 10,000 articles, even with premium models
2. **Per-query costs dominated by LLM generation:** Retrieval is nearly free
3. **Hybrid search adds minimal cost:** ~20% storage overhead, negligible query cost
4. **RAPTOR setup expensive:** $100-300 for summarization, but query cost same as baseline
5. **Graph-RAG cost depends on approach:** Metadata links ~$5, full graph $50-150
6. **Reranking adds latency but improves accuracy:** ColBERT ~$0.001/query
7. **Self-RAG/CRAG increase LLM calls:** 2-5x generation cost, use selectively

**Cost-Optimized Stack for WH40K:**
- Chroma (free, self-hosted) or Qdrant Cloud (pay-per-usage)
- OpenAI text-embedding-3-small ($0.00002/1K tokens)
- rank-bm25 for BM25 (free, open-source)
- Mistral or Gemini for generation (4-10x cheaper than GPT-4)
- RAGAS for evaluation (free, open-source)

**Total Estimated Cost:**
- Setup (full wiki): $10-25
- Per-query: $0.01-0.03 (mostly LLM generation)
- 1000 queries/month: $10-30/month

---

### Section 7: Evaluation Methodology

#### 7.1 Retrieval Metrics Explained

**Precision@K:**
- Of the top-K retrieved chunks, how many are relevant?
- Formula: `Precision@K = (# relevant in top-K) / K`
- Example: Retrieve 10 chunks, 7 relevant → Precision@10 = 0.7

**Recall@K:**
- Of all relevant chunks in corpus, how many are in top-K?
- Formula: `Recall@K = (# relevant in top-K) / (total relevant chunks)`
- Example: 20 relevant chunks exist, 7 in top-10 → Recall@10 = 0.35

**Mean Reciprocal Rank (MRR):**
- Average reciprocal rank of first relevant result
- Formula: `MRR = (1/N) Σ (1 / rank_first_relevant)`
- Example: First relevant at rank 3 → RR = 1/3 = 0.33
- Higher MRR = relevant results ranked higher

**Normalized Discounted Cumulative Gain (NDCG@K):**
- Graded relevance with position weighting
- Highly relevant at top = higher score
- Formula: `NDCG@K = DCG@K / IDCG@K` (normalized by ideal ranking)
- Range: 0 to 1, higher is better

**Which Metrics to Use?**
- **Recall@10:** Ensure relevant content is retrieved
- **MRR:** Check if best result is ranked first
- **NDCG@10:** If you have graded relevance labels (highly relevant vs somewhat relevant)

#### 7.2 Answer Quality Evaluation Frameworks

**RAGAS (Recommended for Testbed)**
- Open-source, LLM-based evaluation
- Metrics:
  - **Faithfulness:** Is answer grounded in retrieved context?
  - **Answer Relevance:** Does answer address the question?
  - **Context Precision:** Are retrieved chunks relevant to question?
  - **Context Recall:** Are all necessary chunks retrieved?
- No human labels needed (reference-free)
- Works with any LLM (GPT-4, Claude, etc.)

**DeepEval (More Features)**
- Includes RAGAS metrics plus additional ones
- Hallucination detection
- Toxicity and bias checks
- Answer correctness (requires reference answers)
- Pytest integration for automated testing

**Manual Human Evaluation (Gold Standard)**
- Sample 50-100 queries
- Human judges rate:
  - Accuracy (1-5 scale)
  - Completeness (1-5 scale)
  - Helpfulness (1-5 scale)
- Expensive but most reliable

**User Feedback (Production)**
- Helpful/Not Helpful buttons
- Click-through rate on follow-up questions
- Track: % queries with "Helpful" clicked
- Target: >80% helpful rate

#### 7.3 How to Construct WH40K Lore Evaluation Dataset

**Step 1: Gather Representative Queries**
- Sample user questions from brainstorming
- Generate queries covering:
  - **Factual:** "Who is Roboute Guilliman?"
  - **Conceptual:** "What is the Warp?"
  - **Relationship:** "How are Blood Angels connected to Sanguinius?"
  - **Comparison:** "Difference between Space Marines and Chaos Space Marines?"
  - **Timeline:** "What happened during the Horus Heresy?"
  - **Speculation:** "What are theories about the Emperor's plan?"
- Target: 50-100 queries

**Step 2: Create Ground Truth (If Possible)**
- For each query, manually identify:
  - Which wiki articles are relevant
  - Which sections within those articles
  - Ideal answer (if feasible)
- Time-consuming but enables Recall@K calculation

**Step 3: Automated Evaluation Setup**
- Run each query through RAG system
- Log:
  - Retrieved chunks
  - Generated answer
  - Latency
  - Cost (tokens used)
- Use RAGAS to score:
  - Faithfulness
  - Answer Relevance
  - Context Precision

**Step 4: Human Spot-Check**
- Sample 20-30 answers
- Human evaluates accuracy
- Calibrate against RAGAS scores
- If RAGAS faithfulness >0.8 correlates with human "accurate", trust RAGAS

#### 7.4 Benchmark Design for 100-Page Testbed

**Testbed Composition:**
- 100 diverse wiki articles
- Cover:
  - Major factions (Imperium, Chaos, Xenos)
  - Key characters (Primarchs, heroes, villains)
  - Major events (Horus Heresy, 13th Black Crusade)
  - Varied article lengths (short bios, long comprehensive articles)

**Evaluation Protocol:**
1. Scrape and process 100 articles
2. Create 50 test queries (see above categories)
3. Run each architecture:
   - Pure vector
   - Hybrid (vector + BM25)
   - Lightweight graph (metadata links)
   - RAPTOR (on 30-page subset due to cost)
4. Measure for each:
   - Recall@10, MRR, NDCG@10
   - RAGAS Faithfulness, Answer Relevance
   - Latency (p50, p95)
   - Cost per query
5. Human spot-check 20 answers
6. Select winning architecture

**Success Criteria:**
- Recall@10 > 0.7 (70% of relevant chunks retrieved)
- RAGAS Faithfulness > 0.8 (minimal hallucination)
- p95 latency < 2 seconds
- Cost < €0.05 per query

#### 7.5 A/B Testing Strategies

**In Production:**
1. Implement baseline architecture (e.g., hybrid search)
2. Deploy to subset of users (50%)
3. Implement candidate architecture (e.g., RAPTOR)
4. Deploy to other 50%
5. Compare:
   - Helpful/Not Helpful click rates
   - User engagement (follow-up questions asked)
   - Session length
6. Roll out winner to 100% of users

**Statistical Significance:**
- Collect 200-500 queries per variant
- Use chi-square test for Helpful rate difference
- p < 0.05 for statistical significance

---

### Section 8: Implementation Guidance

#### 8.1 Recommended Architecture for MVP/Testbed

**Winning Architecture: Hybrid Search with Metadata Links**

**Rationale:**
1. ✅ Handles both semantic and exact-match queries
2. ✅ Preserves wiki hyperlinks for future graph capabilities
3. ✅ Low implementation complexity
4. ✅ Minimal cost increase vs pure vector
5. ✅ Proven performance in research (IBM Blended RAG)
6. ✅ Easy to add reranking or graph later

**Stack:**
- **Vector DB:** Chroma (current) or Qdrant (better metadata filtering)
- **Embeddings:** OpenAI text-embedding-3-small (cheap, good quality)
- **BM25:** rank-bm25 Python library
- **Fusion:** Reciprocal Rank Fusion (simple implementation)
- **Metadata:** article_title, section_heading, links, faction, era, canon_flag, url
- **LLM:** Multi-provider (OpenAI, Gemini, Claude, Mistral, Grok) via base class
- **Evaluation:** RAGAS for automated metrics

#### 8.2 Step-by-Step Implementation Approach

**Phase 1: Baseline Setup (Days 1-3)**
1. ✅ Scrape 100-page testbed as markdown
2. ✅ Implement section-based chunking
3. ✅ Extract metadata (title, section, url, links array)
4. ✅ Embed chunks with OpenAI text-embedding-3-small
5. ✅ Store in Chroma with metadata
6. ✅ Implement pure vector search baseline
7. ✅ Test retrieval quality manually with 10 queries

**Phase 2: Hybrid Search (Days 4-5)**
1. ✅ Tokenize chunks for BM25 indexing
2. ✅ Build BM25 index with rank-bm25
3. ✅ Implement Reciprocal Rank Fusion
4. ✅ Test hybrid vs pure vector on 20 queries
5. ✅ Measure Recall@10, MRR improvements

**Phase 3: RAPTOR Experiment (Days 6-7)**
1. ✅ Select 30-page subset
2. ✅ Implement clustering (GMM or HDBSCAN)
3. ✅ Generate summaries with LLM (GPT-3.5-turbo to save cost)
4. ✅ Build 2-3 level tree
5. ✅ Implement tree-based retrieval
6. ✅ Compare to hybrid baseline on broad queries

**Phase 4: Evaluation (Days 8-9)**
1. ✅ Create 50-query evaluation set
2. ✅ Run all architectures (pure vector, hybrid, RAPTOR)
3. ✅ Measure RAGAS metrics (Faithfulness, Answer Relevance)
4. ✅ Measure Recall@10, MRR, NDCG@10
5. ✅ Track latency and cost
6. ✅ Human spot-check 20 answers
7. ✅ Generate comparison report

**Phase 5: Production Decision (Day 10)**
1. ✅ Select winning architecture based on data
2. ✅ Optimize hyperparameters (k, chunk size, fusion weights)
3. ✅ Document decision rationale
4. ✅ Plan full wiki scaling

#### 8.3 Python Libraries and Frameworks to Use

**Core RAG Framework:**
- **LangChain** or **LlamaIndex:** High-level RAG orchestration (choose one)
- **Haystack:** Alternative, more modular (consider if LangChain too heavy)

**Vector Database:**
- **Chroma:** Easy, local, free (current choice)
- **Qdrant:** Best metadata filtering, cloud or self-hosted (upgrade option)
- **Weaviate:** Built-in hybrid search, production-ready
- **FAISS:** Lightweight, no metadata filtering (only for vector baseline)

**Embeddings:**
- **OpenAI:** text-embedding-3-small ($0.00002/1K tokens)
- **Cohere:** embed-english-v3.0 (alternative, similar cost)
- **Sentence-Transformers:** Open-source, free (for cost-conscious prototype)

**BM25/Sparse Retrieval:**
- **rank-bm25:** Pure Python, simple, works well
- **Elasticsearch:** More features but heavyweight (defer unless needed)

**LLM Providers:**
- **OpenAI:** GPT-4, GPT-3.5-turbo (via openai Python SDK)
- **Anthropic:** Claude (via anthropic Python SDK)
- **Google:** Gemini (via google-generativeai SDK)
- **xAI:** Grok (via API)
- **Mistral:** Mistral models (via mistralai SDK)

**Evaluation:**
- **RAGAS:** Open-source, LLM-based metrics (`pip install ragas`)
- **DeepEval:** More features (`pip install deepeval`)
- **TrueLens:** Observability for RAG systems

**Utilities:**
- **BeautifulSoup4 / Scrapy:** Web scraping
- **markdownify:** HTML to markdown conversion
- **scikit-learn:** Clustering for RAPTOR (GMM)
- **NetworkX:** Graph operations (if metadata links → graph)

#### 8.4 Common Pitfalls and How to Avoid Them

**Pitfall 1: Chunk Size Too Large**
- Symptom: LLM context window exceeded, irrelevant info in chunks
- Solution: Target 200-500 tokens per chunk, split large sections

**Pitfall 2: Ignoring Metadata Filtering**
- Symptom: Retrieving spoilers when user wants spoiler-free
- Solution: Use `canon_flag`, `spoiler_flag` metadata filters at query time

**Pitfall 3: BM25 on Raw HTML**
- Symptom: BM25 matches on HTML tags, not content
- Solution: Clean text thoroughly before BM25 indexing (strip tags, normalize)

**Pitfall 4: Over-Retrieval**
- Symptom: Retrieving top-50 chunks overwhelms LLM context
- Solution: Retrieve 20 chunks, rerank to top-5-10 for LLM

**Pitfall 5: No Deduplication**
- Symptom: Same content retrieved multiple times (from overlapping chunks)
- Solution: Deduplicate by chunk ID or content hash before sending to LLM

**Pitfall 6: Ignoring User Feedback**
- Symptom: Building complex architecture without validating with real users
- Solution: Implement Helpful/Not Helpful buttons from day 1, iterate on feedback

**Pitfall 7: Premature Optimization**
- Symptom: Implementing RAPTOR/graph before proving baseline insufficient
- Solution: Start simple (hybrid search), add complexity only if testbed shows gaps

**Pitfall 8: Forgetting to Handle Hungarian Translation**
- Symptom: Retrieval works in English, fails when user queries in Hungarian
- Solution: Embed query in same language as documents (English), translate only for LLM generation

#### 8.5 Iterative Refinement Strategy

**Week 1-2: Testbed & Baseline**
- Implement hybrid search (vector + BM25)
- Measure performance on 50 queries
- Identify failure modes (query types that fail)

**Week 3: Targeted Improvements**
- If broad queries fail → Test RAPTOR on subset
- If relationship queries fail → Test metadata link expansion
- If accuracy low → Add reranking (LLM-based or ColBERT)

**Week 4: Optimization**
- Tune hyperparameters (k, chunk size, fusion alpha)
- Optimize metadata extraction (NER, canon detection)
- Benchmark latency and cost

**Week 5+: Full Wiki Scale**
- Scale winning architecture to full wiki
- Monitor production metrics (Helpful rate, latency)
- A/B test incremental improvements

**Continuous Iteration:**
- Monthly review of user feedback
- Quarterly evaluation of new RAG techniques (research moves fast)
- Ongoing refinement of chunking, metadata, prompts

---

## Supporting Materials

### Comparison Tables

#### RAG Architecture Feature Matrix

| Architecture | Complexity | Setup Cost | Per-Query Cost | Latency | Best For |
|--------------|------------|------------|----------------|---------|----------|
| Pure Vector | LOW | $5-15 | $0.01-0.05 | 200-500ms | Conceptual queries, paraphrases |
| Hybrid (Vector+BM25) | LOW-MED | $6-18 | $0.01-0.05 | 300-700ms | **Mixed query types, proper nouns** |
| Graph-RAG (Metadata) | LOW | $7-20 | $0.01-0.05 | 300-700ms | **Wiki cross-references, related topics** |
| Graph-RAG (Full) | HIGH | $55-165 | $0.01-0.05 | 500ms-2s | Multi-hop reasoning, entity relationships |
| RAPTOR | MEDIUM | $105-315 | $0.01-0.05 | 300ms-1.2s | Hierarchical content, broad queries |
| Advanced RAG + Reranking | MEDIUM | $10-30 | $0.015-0.06 | 500ms-1s | High precision needs |
| Self-RAG / CRAG | HIGH | $20-50 | $0.02-0.10 | 500ms-1.5s | Adaptive retrieval, hallucination reduction |

#### Vector Database Comparison

| Database | Metadata Filtering | Hybrid Search | Deployment | Cost | Best For |
|----------|-------------------|---------------|------------|------|----------|
| **Chroma** | Good | Via integration | Local/cloud | Free (self-hosted) | **Prototyping, small-scale** |
| **Qdrant** | **Excellent** | Native (sparse+dense) | Local/cloud | Pay-per-usage | **Production, complex filters** |
| **Weaviate** | Excellent | Native (alpha param) | Local/cloud | Free/paid tiers | Production, hybrid search |
| **Pinecone** | Good | Native | Cloud-only | Pay-per-usage | Managed, turnkey scaling |
| **FAISS** | None | No | Local library | Free | Baseline experiments |
| **Milvus** | Good | Via plugin | Local/cloud | Free (self-hosted) | GPU acceleration, large scale |

#### Cost Breakdown by Architecture (10K Wiki Articles)

| Cost Component | Pure Vector | Hybrid | Graph (Meta) | Graph (Full) | RAPTOR |
|----------------|-------------|--------|--------------|--------------|--------|
| Scraping | $0 | $0 | $0 | $0 | $0 |
| Embedding (OpenAI) | $10 | $10 | $10 | $10 | $10 |
| BM25 Indexing | - | $1 | $1 | $1 | $1 |
| Metadata Extraction | $2 | $2 | $5 | $50 | $2 |
| Clustering | - | - | - | - | $5 |
| Summarization (LLM) | - | - | - | $100 | $200 |
| **Total Setup** | **$12** | **$13** | **$16** | **$161** | **$218** |
| **Per-Query** | **$0.02** | **$0.02** | **$0.02** | **$0.02** | **$0.02** |

#### Latency Expectations Table

| Architecture | Retrieval (p95) | LLM Generation (p95) | Total (p95) |
|--------------|-----------------|---------------------|-------------|
| Pure Vector | 50-100ms | 800ms-1.5s | 850ms-1.6s |
| Hybrid | 100-150ms | 800ms-1.5s | 900ms-1.65s |
| Graph-RAG (metadata) | 100-150ms | 800ms-1.5s | 900ms-1.65s |
| Graph-RAG (full) | 300-700ms | 800ms-1.5s | 1.1s-2.2s |
| RAPTOR | 150-300ms | 800ms-1.5s | 950ms-1.8s |
| With Reranking | +200-400ms | 800ms-1.5s | 1.2s-2.3s |

---

## Sources

### Graph RAG
- [Knowledge Graphs for RAG - DeepLearning.AI](https://www.deeplearning.ai/short-courses/knowledge-graphs-rag/)
- [RAG Tutorial: How to Build a RAG System on a Knowledge Graph - Neo4j](https://neo4j.com/blog/developer/rag-tutorial/)
- [Enhancing the Accuracy of RAG Applications With Knowledge Graphs - Neo4j](https://medium.com/neo4j/enhancing-the-accuracy-of-rag-applications-with-knowledge-graphs-ad5e2ffab663)
- [Using a Knowledge Graph to Implement a RAG Application - DataCamp](https://www.datacamp.com/tutorial/knowledge-graph-rag)
- [How to Implement Graph RAG Using Knowledge Graphs and Vector Databases - Medium](https://medium.com/data-science/how-to-implement-graph-rag-using-knowledge-graphs-and-vector-databases-60bb69a22759)
- [GraphRAG Explained: Enhancing RAG with Knowledge Graphs - Zilliz](https://medium.com/@zilliz_learn/graphrag-explained-enhancing-rag-with-knowledge-graphs-3312065f99e1)

### Hybrid Retrieval
- [Optimizing RAG with Hybrid Search & Reranking - VectorHub](https://superlinked.com/vectorhub/articles/optimizing-rag-with-hybrid-search-reranking)
- [Hybrid Search Explained - Weaviate](https://weaviate.io/blog/hybrid-search-explained)
- [Dense vector + Sparse vector + Full text search + Tensor reranker - Infinity](https://infiniflow.org/blog/best-hybrid-search-solution)
- [Blended RAG: Improving RAG Accuracy with Semantic Search and Hybrid Query-Based Retrievers](https://arxiv.org/html/2404.07220)
- [Implementing Hybrid Retrieval (BM25 + FAISS) in RAG - Chitika](https://www.chitika.com/hybrid-retrieval-rag/)
- [Hybrid search with Postgres Native BM25 and VectorChord](https://blog.vectorchord.ai/hybrid-search-with-postgres-native-bm25-and-vectorchord)
- [Improving RAG Performance: WTF is Hybrid Search? - Fuzzy Labs](https://www.fuzzylabs.ai/blog-post/improving-rag-performance-hybrid-search)

### Wiki & Structured Data
- [RAG using structured data: Overview - Kuzu](https://blog.kuzudb.com/post/llms-graphs-part-1/)
- [How to Build Accurate RAG Over Structured Databases - Medium](https://medium.com/madhukarkumar/chapter-1-how-to-build-accurate-rag-over-structured-and-semi-structured-databases-996c68098dba)
- [RAG for Structured Data: Benefits, Challenges & Examples - AI21](https://www.ai21.com/knowledge/rag-for-structured-data/)
- [SRAG: Structured Retrieval-Augmented Generation for Multi-Entity Question Answering over Wikipedia Graph](https://arxiv.org/abs/2503.01346)
- [The unintended impact of Wikipedia on RAG best practices - Medium](https://sarah-packowski.medium.com/the-unintended-impact-of-wikipedia-on-rag-best-practices-00821aa2d9aa)

### Evaluation Metrics
- [LLM Evaluation Metrics: RAG, Offline metrics (MRR, NDCG), ROUGE, BLEU - Medium](https://medium.com/@ramachandran1985/llm-evaluation-metrics-rag-offline-metrics-mrr-ndcg-rouge-bleu-1145761da900)
- [Evaluation Metrics for RAG Systems - GeeksforGeeks](https://www.geeksforgeeks.org/nlp/evaluation-metrics-for-retrieval-augmented-generation-rag-systems/)
- [Metrics for Evaluation of Retrieval in RAG Systems - DeconvoluteAI](https://deconvoluteai.com/blog/rag/metrics-retrieval)
- [RAG Evaluation: Don't let customers tell you first - Pinecone](https://www.pinecone.io/learn/series/vector-databases-in-production-for-busy-engineers/rag-evaluation/)
- [RAG Evaluation Metrics Explained: Recall@K, MRR, Faithfulness - LangCopilot](https://langcopilot.com/posts/2025-09-17-rag-evaluation-101-from-recall-k-to-answer-faithfulness)
- [Best Practices in RAG Evaluation: A Comprehensive Guide - Qdrant](https://qdrant.tech/blog/rag-evaluation-guide/)

### Vector Databases
- [Vector Database Comparison: Pinecone vs Weaviate vs Qdrant vs FAISS vs Milvus vs Chroma - LiquidMetal AI](https://liquidmetal.ai/casesAndBlogs/vector-comparison/)
- [Best 17 Vector Databases for 2025](https://lakefs.io/blog/best-vector-databases/)
- [The 7 Best Vector Databases in 2026 - DataCamp](https://www.datacamp.com/blog/the-top-5-vector-databases)
- [Vector Stores for RAG Comparison - Rost Glukhov](https://www.glukhov.org/post/2025/12/vector-stores-for-rag-comparison/)
- [Top 7 Open-Source Vector Databases: Faiss vs. Chroma & More](https://research.aimultiple.com/open-source-vector-databases/)

### Chunking Strategies
- [Chunking Strategies to Improve Your RAG Performance - Weaviate](https://weaviate.io/blog/chunking-strategies-for-rag)
- [The Ultimate Guide to Chunking Strategies for RAG - Databricks](https://community.databricks.com/t5/technical-blog/the-ultimate-guide-to-chunking-strategies-for-rag-applications/ba-p/113089)
- [Semantic Chunking for RAG: Better Context, Better Results - Multimodal](https://www.multimodal.dev/post/semantic-chunking-for-rag)
- [7 Chunking Strategies in RAG You Need To Know - F22 Labs](https://www.f22labs.com/blogs/7-chunking-strategies-in-rag-you-need-to-know/)
- [Breaking up is hard to do: Chunking in RAG applications - Stack Overflow](https://stackoverflow.blog/2024/12/27/breaking-up-is-hard-to-do-chunking-in-rag-applications/)
- [Best Chunking Strategies for RAG in 2025 - Firecrawl](https://www.firecrawl.dev/blog/best-chunking-strategies-rag-2025)

### Advanced RAG (Self-RAG, CRAG)
- [Corrective Retrieval Augmented Generation - arXiv](https://arxiv.org/abs/2401.15884)
- [CRAG GitHub Repository](https://github.com/HuskyInSalt/CRAG)
- [The 2025 Guide to Retrieval-Augmented Generation - EdenAI](https://www.edenai.co/post/the-2025-guide-to-retrieval-augmented-generation-rag)
- [Implementing cRAG to Prevent AI Hallucinations - Chitika](https://www.chitika.com/corrective-rag-hallucinations/)
- [Self-RAG GitHub Repository](https://github.com/AkariAsai/self-rag)
- [Corrective RAG (CRAG) - Cobus Greyling](https://cobusgreyling.medium.com/corrective-rag-crag-5e40467099f8)

---

*Research conducted using BMAD-METHOD™ framework*
*Report generated: 2025-12-17*
