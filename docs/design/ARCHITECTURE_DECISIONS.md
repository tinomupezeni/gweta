# Gweta Architecture Decisions Log

This document tracks architectural discussions and decisions for Gweta's evolution from a data quality tool to an intelligent RAG curation engine.

---

## Discussion 1: Vision & Direction
**Date:** 2025-02-06

### Context
Gweta v0.1.x provides basic data acquisition, validation, and ingestion for RAG systems. The question arose: how should Gweta evolve to provide enterprise-grade value?

### The Problem with Current RAG Tooling
- Tools exist for chunking, embedding, storage, retrieval
- **Nobody solves curation at scale**
- 60%+ of chunks in typical RAG systems are noise
- "Garbage in, garbage out" - bad data defeats good models

### Vision Statement
> "Gweta doesn't just clean your data—it understands your AI's mission and ensures every chunk in your knowledge base earns its place."

### Key Insight
Current tools ask: "Is this chunk well-formed?"
Gweta should ask: "Does this chunk help the AI answer the questions it will be asked?"

**This is the paradigm shift: from quality validation to relevance curation.**

---

## Discussion 2: Intelligence Approach
**Date:** 2025-02-06

### Options Considered

| Option | Approach | Pros | Cons |
|--------|----------|------|------|
| A | Pure heuristics | Fast, no deps | Limited accuracy |
| B | Hybrid (embeddings + classifiers) | Local, accurate, no API cost | More complex |
| C | LLM-based | Most accurate | Expensive, API dependency |

### Decision: Option B - Hybrid Approach

**Rationale:**
- Runs fully local (privacy, no costs)
- Fast enough for real-time processing
- Accurate enough for production use
- No external API dependencies
- Aligns with enterprise requirements

### Architecture Components

```
GWETA INTELLIGENCE LAYER
├── Embedding Engine (sentence-transformers)
│   └── Semantic similarity to system intent
├── Classifier Bank
│   ├── Domain Detector (legal, business, tech, etc.)
│   ├── Quality Assessor (coherence, completeness)
│   └── Intent Matcher (relevance to system purpose)
└── Decision Engine
    └── Weighted combination → INGEST / REVIEW / REJECT
```

### Resource Requirements
- Total model size: ~140MB
- Memory footprint: ~500MB
- Speed: ~1000 chunks/sec on CPU
- No GPU required (but faster with one)

---

## Discussion 3: System Intent
**Date:** 2025-02-06

### Concept
Each Gweta deployment serves a specific system (e.g., Simuka career platform). The system's "intent" defines what content is relevant.

### Intent Definition Format
```yaml
system:
  name: "System Name"
  description: "What the system does"
  target_users: [who uses it]
  core_questions: [what queries should it answer]
  relevant_topics: [what to include]
  irrelevant_topics: [what to exclude]
  geographic_focus: "Region"
  quality_requirements:
    min_relevance_score: 0.6
    freshness_days: 365
```

### How Intent is Used
1. Intent description → embedded as vector
2. Each chunk → embedded as vector
3. Semantic similarity computed
4. Combined with classifier outputs
5. Final relevance score determines fate

---

## Discussion 4: Phased Roadmap
**Date:** 2025-02-06

### Phase 1: Intent-Aware MVP (v0.2)
- System intent definition (YAML)
- Embedding-based relevance scoring
- Pre-trained models only
- No custom classifiers yet
- **Goal:** Prove the concept works

### Phase 2: Classification Layer (v0.3)
- Add domain detector
- Add quality assessor
- Pre-trained classifiers
- Combined relevance scoring
- **Goal:** Improve accuracy

### Phase 3: Learning System (v0.4)
- Feedback collection
- Model fine-tuning
- Domain packs (downloadable)
- African-specific adaptations
- **Goal:** System gets smarter

### Phase 4: Enterprise Features (v1.0)
- Multi-system support
- Admin dashboard
- Audit trails
- API for integration
- **Goal:** Production-ready

---

## Discussion 5: Simuka Integration
**Date:** 2025-02-06

### Context
Simuka is an AI-powered career activation platform for Zimbabwean graduates. Gweta serves as its data engine.

### Simuka's Data Needs
- Zimbabwe business regulations (ZIMRA, company registration)
- Career pathway guidance by sector
- Market intelligence (pricing, opportunities)
- Incubator and funding information
- Success stories and examples

### What Gweta Provides for Simuka
- Acquires data from ZW-specific sources
- Validates for quality AND relevance to career guidance
- Filters out content that doesn't serve graduate users
- Maintains freshness (rejects stale info)
- Enriches with sector/pathway metadata

### Phase 1 Value for Simuka
See dedicated section in design docs.

---

## Open Questions

1. **Pre-built domain packs?**
   - Should Gweta ship downloadable domain packs?
   - `gweta install domain:african-business`
   - Decision: TBD

2. **Feedback mechanism?**
   - How does the system learn what's useful?
   - Explicit ratings vs implicit signals
   - Decision: TBD

3. **Multi-system support?**
   - One Gweta instance, multiple systems?
   - Or one Gweta per system?
   - Decision: TBD

4. **African training data?**
   - Pre-trained models are Western-biased
   - Need African-specific fine-tuning?
   - Decision: TBD (likely yes for v0.4)

---

## Changelog

| Date | Discussion | Decision |
|------|------------|----------|
| 2025-02-06 | Vision & Direction | Relevance-aware curation engine |
| 2025-02-06 | Intelligence Approach | Option B: Hybrid (embeddings + classifiers) |
| 2025-02-06 | System Intent | YAML-based intent definition |
| 2025-02-06 | Roadmap | 4-phase approach, v0.2 first |
