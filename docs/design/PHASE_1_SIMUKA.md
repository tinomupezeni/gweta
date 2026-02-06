# Phase 1: Intent-Aware MVP for Simuka

## What Phase 1 Delivers

**Gweta v0.2** adds a single powerful capability: **embedding-based relevance scoring against system intent.**

Instead of storing everything, Gweta asks: "Does this content help Simuka's users?"

---

## The Problem Phase 1 Solves

### Current Situation (Without Intent-Aware Gweta)

You're building Simuka's knowledge base by crawling:
- ZIMRA website
- Company registration guides
- TechZim articles
- General business blogs
- News sites

**What happens:**

```
Crawl 1,000 pages
    ↓
Extract 5,000 chunks
    ↓
Store ALL 5,000 in vector DB
    ↓
User asks: "How do I register a business in Zimbabwe?"
    ↓
RAG retrieves 5 chunks:
  - ✓ 2 relevant (ZW business registration)
  - ✗ 1 about US LLC formation (crawled from linked article)
  - ✗ 1 about corporate restructuring (too advanced)
  - ✗ 1 news about business closures (negative, not helpful)
    ↓
AI generates mediocre answer mixing useful and useless info
```

**The problem:** You stored everything, so retrieval returns noise alongside signal.

---

### With Phase 1 Gweta

```
Crawl 1,000 pages
    ↓
Extract 5,000 chunks
    ↓
FOR EACH CHUNK:
  │
  ├─→ Embed chunk
  ├─→ Compare to Simuka's intent embedding
  ├─→ Score relevance (0.0 - 1.0)
  │
  └─→ Decision:
        Score ≥ 0.7  → INGEST (clearly relevant)
        Score 0.4-0.7 → FLAG (maybe relevant, review)
        Score < 0.4  → REJECT (not relevant)
    ↓
Store only 2,000 relevant chunks (60% rejection)
    ↓
User asks: "How do I register a business in Zimbabwe?"
    ↓
RAG retrieves 5 chunks:
  - ✓ 5 relevant (all about ZW business registration)
    ↓
AI generates excellent, focused answer
```

**The result:** Higher quality answers because the knowledge base only contains relevant content.

---

## How It Works for Simuka

### Step 1: Define Simuka's Intent

```yaml
# simuka_intent.yaml
system:
  name: "Simuka Career Platform"

  description: |
    AI career advisor helping Zimbabwean university graduates
    transition to employment or entrepreneurship. Provides
    practical, actionable guidance with local context.

  target_users:
    - Recent university graduates in Zimbabwe
    - Students preparing for job market
    - Young entrepreneurs starting first business

  core_questions:
    - "How do I register a business in Zimbabwe?"
    - "What freelance services can I offer with my degree?"
    - "How much capital do I need to start?"
    - "Where can I find my first clients?"
    - "What are ZIMRA requirements for small business?"
    - "How do I price my services?"
    - "What sectors are growing in Zimbabwe?"

  relevant_topics:
    - Zimbabwe business registration and compliance
    - ZIMRA tax requirements for small business
    - Freelancing and consulting in Africa
    - Startup costs and funding options
    - Career paths by degree/sector
    - EcoCash and mobile payments
    - Local market conditions and pricing
    - Success stories from Zimbabwe/Africa
    - Practical entrepreneurship guides
    - University-to-work transition

  irrelevant_topics:
    - US/EU/Asian business regulations
    - Large enterprise / corporate content
    - Academic research papers
    - Content older than 2022
    - Generic motivational content
    - Cryptocurrency/forex trading
    - Multi-level marketing

  geographic_focus:
    primary: Zimbabwe
    secondary: Southern Africa, East Africa

  quality_requirements:
    min_relevance_score: 0.6
    freshness_cutoff: "2022-01-01"
```

### Step 2: Gweta Creates Intent Embedding

When Simuka's intent file is loaded, Gweta:

1. Combines description + core_questions + relevant_topics into text
2. Generates embedding vector using sentence-transformers
3. Stores this as the "intent vector" for comparison

```python
# Internally
intent_text = f"""
{intent.description}

This system should answer questions like:
{' '.join(intent.core_questions)}

Relevant topics include:
{' '.join(intent.relevant_topics)}
"""

intent_embedding = embedding_model.encode(intent_text)
```

### Step 3: Score Each Chunk

For every chunk that passes basic quality filters:

```python
# For each chunk
chunk_embedding = embedding_model.encode(chunk.text)

# Cosine similarity to intent
relevance_score = cosine_similarity(chunk_embedding, intent_embedding)

# Decision
if relevance_score >= 0.7:
    action = "INGEST"
elif relevance_score >= 0.4:
    action = "REVIEW"
else:
    action = "REJECT"
```

### Step 4: Enrich and Store

Relevant chunks get enriched metadata:

```python
chunk.metadata["relevance_score"] = 0.82
chunk.metadata["matched_topics"] = ["business registration", "ZIMRA"]
chunk.metadata["intent_version"] = "simuka_v1"
```

---

## Concrete Example: Processing a Web Page

**Source:** TechZim article "Starting a Tech Business in Zimbabwe 2024"

**Chunks extracted:**

| Chunk | Content Summary | Relevance Score | Decision |
|-------|-----------------|-----------------|----------|
| 1 | "Registering a PBC costs $70-100 at Companies Registry..." | 0.89 | ✅ INGEST |
| 2 | "ZIMRA requires registration within 30 days..." | 0.85 | ✅ INGEST |
| 3 | "The author previously worked at Microsoft Seattle..." | 0.21 | ❌ REJECT |
| 4 | "Subscribe to our newsletter for updates..." | 0.08 | ❌ REJECT |
| 5 | "Freelance developers can charge $20-50/hr..." | 0.91 | ✅ INGEST |
| 6 | "Silicon Valley trends suggest AI will..." | 0.35 | ❌ REJECT |
| 7 | "EcoCash merchant registration is free..." | 0.88 | ✅ INGEST |

**Result:** 4 of 7 chunks ingested. The noise (author bio, newsletter CTA, US-centric content) is automatically filtered.

---

## What Simuka Gains

### Before Phase 1
- Store everything, hope retrieval works
- 40-60% of retrieved chunks are noise
- AI answers are diluted with irrelevant info
- Users get generic, unfocused guidance

### After Phase 1
- Only store content that serves Simuka's mission
- 90%+ of retrieved chunks are relevant
- AI answers are focused and actionable
- Users get Zimbabwe-specific, practical guidance

---

## Measurable Outcomes

| Metric | Before | After Phase 1 |
|--------|--------|---------------|
| Chunks stored | 100% of extracted | ~40% (60% filtered) |
| Storage cost | Higher | 60% reduction |
| Retrieval precision | ~50% | ~90% |
| Answer quality | Mixed | Focused |
| User satisfaction | Variable | Higher |

---

## What Phase 1 Does NOT Do (Yet)

These come in later phases:

| Feature | Phase |
|---------|-------|
| Domain classification (legal vs business vs tech) | Phase 2 |
| Quality assessment (coherence, completeness) | Phase 2 |
| Learning from feedback | Phase 3 |
| Fine-tuned African context | Phase 3 |
| Multi-system support | Phase 4 |

Phase 1 is intentionally simple: **one embedding model, one intent, one relevance score.**

If this works (and it will), we add complexity. If it doesn't, we learn why before over-engineering.

---

## Technical Requirements for Phase 1

### New Components
```
gweta/
├── intelligence/
│   ├── __init__.py
│   ├── intent.py          # Intent loading and embedding
│   ├── embeddings.py      # Embedding model wrapper
│   └── relevance.py       # Relevance scoring logic
```

### Dependencies
```
sentence-transformers>=2.2  # Embedding model
numpy>=1.24                 # Vector operations
```

### Model
- **Default:** `all-MiniLM-L6-v2` (80MB, fast)
- **Optional:** `all-mpnet-base-v2` (420MB, more accurate)

### Configuration
```python
# gweta.yaml
intelligence:
  enabled: true
  intent_file: "simuka_intent.yaml"
  embedding_model: "all-MiniLM-L6-v2"
  relevance_threshold: 0.6
  review_threshold: 0.4
```

---

## API Changes for Phase 1

### Current (v0.1)
```python
from gweta import ChromaStore, ChunkValidator

validator = ChunkValidator()
report = validator.validate_batch(chunks)
store.add(report.accepted())  # Adds all that pass quality
```

### Phase 1 (v0.2)
```python
from gweta import ChromaStore, ChunkValidator
from gweta.intelligence import IntentFilter

# Load system intent
intent = IntentFilter.from_yaml("simuka_intent.yaml")

# Validate quality
validator = ChunkValidator()
quality_report = validator.validate_batch(chunks)

# Filter for relevance
relevance_report = intent.filter_batch(quality_report.accepted())

# Only store relevant chunks
store.add(relevance_report.accepted())

# See what was filtered
print(f"Quality passed: {quality_report.passed}")
print(f"Relevance passed: {relevance_report.passed}")
print(f"Total rejection rate: {relevance_report.rejection_rate:.1%}")
```

### Or simpler unified API
```python
from gweta import Pipeline

pipeline = Pipeline(
    intent="simuka_intent.yaml",
    store=ChromaStore("simuka_kb")
)

# One call does everything
result = await pipeline.ingest("https://techzim.co.zw/business-guide/")

print(f"Ingested: {result.ingested}")
print(f"Rejected: {result.rejected}")
print(f"Avg relevance: {result.avg_relevance_score:.2f}")
```

---

## Summary

**Phase 1 answers one question:**

> "Is this chunk relevant to what Simuka's users need?"

Everything else (domain detection, quality assessment, learning) comes later.

This simple addition—embedding similarity to intent—will filter 50-70% of noise and dramatically improve Simuka's answer quality.

That's the value. That's what we build first.
