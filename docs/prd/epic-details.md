# Epic Details

## Epic 1: Foundation & Data Pipeline

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

## Epic 2: Core RAG Query System

*[To be detailed in next iteration]*

---

## Epic 3: Discord Bot Integration & User Interactions

*[To be detailed in next iteration]*

---

## Epic 4: Trivia System & Gamification

*[To be detailed in next iteration]*

---

## Epic 5: Multi-LLM Support & Server Configuration

*[To be detailed in next iteration]*

---
