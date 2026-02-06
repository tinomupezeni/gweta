# Intelligence Layer

**Intent-Aware Content Filtering for RAG Systems**

The intelligence layer is Gweta's ML-powered capability that ensures only relevant content reaches your vector store. Instead of storing everything and hoping for the best, Gweta understands your system's purpose and filters accordingly.

## The Problem

Traditional RAG pipelines suffer from a fundamental issue:

```
BEFORE: Crawl everything → Store everything → Noisy retrieval → Bad answers
```

When you crawl a website about "Zimbabwe business registration", you might also get:
- US tax regulations (wrong country)
- Cryptocurrency promotions (spam)
- Enterprise ERP content (wrong audience)

All this noise pollutes your vector store and degrades retrieval quality.

## The Solution

```
AFTER: Crawl → Filter by Intent → Store only relevant → Clean retrieval → Good answers
```

Gweta's intelligence layer:
1. Understands what your RAG system is meant to do (your "intent")
2. Scores each chunk by semantic similarity to that intent
3. Accepts, reviews, or rejects chunks based on relevance
4. Only stores what matters

## Installation

```bash
pip install gweta[intelligence]
```

This installs `sentence-transformers` for embedding-based relevance scoring.

## Quick Start

### 1. Define Your System Intent

Create a YAML file describing what your RAG system does:

```yaml
# intents/my_system.yaml
system:
  name: "My Knowledge Base"

  description: |
    Answers questions about Zimbabwe business registration,
    freelancing, and entrepreneurship for graduates.

  # Questions your system should answer well
  core_questions:
    - "How do I register a business in Zimbabwe?"
    - "What are ZIMRA tax requirements?"
    - "How do I accept EcoCash payments?"

  # Topics to include
  relevant_topics:
    - Zimbabwe business registration
    - ZIMRA tax requirements
    - EcoCash merchant integration
    - freelancing in Africa
    - entrepreneurship

  # Topics to reject
  irrelevant_topics:
    - US tax law
    - cryptocurrency trading
    - forex signals
    - enterprise ERP systems

  # Quality thresholds
  quality_requirements:
    min_relevance_score: 0.6    # Accept if >= 0.6
    review_threshold: 0.4        # Review if 0.4-0.6, reject if < 0.4
```

### 2. Use the Pipeline

```python
from gweta.intelligence import Pipeline, SystemIntent
from gweta import ChromaStore

# Load intent from YAML
intent = SystemIntent.from_yaml("intents/my_system.yaml")

# Create pipeline with your vector store
store = ChromaStore(collection_name="my-knowledge-base")
pipeline = Pipeline(intent=intent, store=store)

# Ingest chunks with automatic filtering
result = await pipeline.ingest(chunks)

print(f"Ingested: {result.ingested} chunks")
print(f"Rejected: {result.rejected_count} irrelevant chunks")
print(f"Acceptance rate: {result.acceptance_rate:.0%}")
```

### 3. Preview Before Ingesting

```python
# Filter without storing - preview what would happen
report = pipeline.filter_only(chunks)

print(f"Would accept: {report.accepted_count}")
print(f"Would reject: {report.rejected_count}")
print(f"Needs review: {report.review_count}")

# See rejected chunks and reasons
for result in report.results:
    if result.rejected:
        print(f"REJECT: {result.chunk.source}")
        print(f"  Reason: {result.rejection_reason}")
        print(f"  Score: {result.relevance_score:.2f}")
```

## Core Components

### SystemIntent

Defines what your RAG system is meant to do.

```python
from gweta.intelligence import SystemIntent

# Create programmatically
intent = SystemIntent(
    name="Simuka Career Platform",
    description="Career guidance for Zimbabwean graduates",
    target_users=["graduates", "young entrepreneurs"],
    core_questions=[
        "How do I register a business in Zimbabwe?",
        "What freelance services can I offer?",
    ],
    relevant_topics=[
        "Zimbabwe business",
        "ZIMRA",
        "EcoCash",
        "freelancing",
    ],
    irrelevant_topics=[
        "US regulations",
        "cryptocurrency",
        "forex",
    ],
)

# Or load from YAML
intent = SystemIntent.from_yaml("intents/simuka.yaml")

# Check if a topic is irrelevant
if intent.is_irrelevant_topic("Invest in Bitcoin now!"):
    print("This content would be rejected")
```

**Key Properties:**

| Property | Description |
|----------|-------------|
| `name` | System name |
| `description` | What the system does |
| `target_users` | Who uses the system |
| `core_questions` | Questions it should answer well |
| `relevant_topics` | Topics to include |
| `irrelevant_topics` | Topics to reject |
| `min_relevance_score` | Threshold for auto-accept (default: 0.6) |
| `review_threshold` | Threshold for review queue (default: 0.4) |

### RelevanceFilter

Scores and filters chunks based on semantic similarity to intent.

```python
from gweta.intelligence import RelevanceFilter

filter = RelevanceFilter(intent=intent)

# Score a single chunk
result = filter.filter(chunk)
print(f"Score: {result.relevance_score:.2f}")
print(f"Decision: {result.decision}")  # ACCEPT, REVIEW, or REJECT
print(f"Matched topics: {result.matched_topics}")

# Filter a batch
report = filter.filter_batch(chunks)
print(f"Accepted: {report.accepted_count}")
print(f"Rejected: {report.rejected_count}")

# Get just the accepted chunks (with metadata added)
accepted_chunks = report.accepted()
```

**How Scoring Works:**

1. Your intent is converted to an embedding (from description + core questions + topics)
2. Each chunk is embedded using the same model
3. Cosine similarity is computed between chunk and intent embeddings
4. Similarity is normalized to 0-1 range
5. Decision is made based on thresholds:
   - `>= min_relevance_score` → ACCEPT
   - `>= review_threshold` → REVIEW
   - `< review_threshold` → REJECT

**Irrelevant Topic Detection:**

Before scoring, chunks are checked against `irrelevant_topics`. If any irrelevant topic is found in the text, the chunk is immediately rejected with score 0.0.

### EmbeddingEngine

Wrapper around sentence-transformers with lazy loading.

```python
from gweta.intelligence import EmbeddingEngine

# Uses all-MiniLM-L6-v2 by default (fast, 80MB)
engine = EmbeddingEngine()

# Or specify a different model
engine = EmbeddingEngine(model_name="all-mpnet-base-v2")

# Embed text
vector = engine.embed("Zimbabwe business registration guide")

# Batch embed
vectors = engine.embed_batch([
    "Register a company in Zimbabwe",
    "ZIMRA tax requirements",
])

# Compute similarity
similarity = engine.similarity(vector1, vector2)
```

**Available Models:**

| Model | Size | Speed | Quality |
|-------|------|-------|---------|
| `all-MiniLM-L6-v2` (default) | 80MB | Fast | Good |
| `all-mpnet-base-v2` | 420MB | Medium | Better |
| `all-MiniLM-L12-v2` | 120MB | Medium | Good |

### Pipeline

Unified API combining quality validation + relevance filtering + ingestion.

```python
from gweta.intelligence import Pipeline

# With a store - full pipeline
pipeline = Pipeline(intent=intent, store=store)
result = await pipeline.ingest(chunks)

# Without a store - filter only
pipeline = Pipeline(intent=intent, store=None)
report = pipeline.filter_only(chunks)

# Score a single chunk
scores = pipeline.score_chunk(chunk)
print(f"Quality: {scores['quality_score']}")
print(f"Relevance: {scores['relevance_score']:.2f}")
print(f"Would ingest: {scores['would_ingest']}")
```

## YAML Intent Schema

Full schema for intent YAML files:

```yaml
system:
  # Required
  name: "System Name"
  description: "What the system does"

  # Optional - who uses this system
  target_users:
    - User type 1
    - User type 2

  # Optional but recommended - questions it should answer
  core_questions:
    - "Question 1?"
    - "Question 2?"

  # Optional but recommended - topics to include
  relevant_topics:
    - topic1
    - topic2

  # Optional but recommended - topics to reject
  irrelevant_topics:
    - bad_topic1
    - bad_topic2

  # Optional - geographic focus
  geographic_focus:
    primary: Zimbabwe
    secondary:
      - South Africa
      - Botswana

  # Optional - quality thresholds
  quality_requirements:
    min_relevance_score: 0.6      # Accept threshold
    review_threshold: 0.4          # Review threshold
    freshness_cutoff: "2022-01-01" # Reject content before this
    prefer_official_sources: true

  # Optional - custom metadata
  metadata:
    version: "1.0"
    sectors:
      - IT
      - Commerce
```

## Example: Simuka Career Platform

Here's a complete example for the Simuka career platform:

```yaml
# intents/simuka.yaml
system:
  name: "Simuka Career Platform"

  description: |
    AI-powered career activation platform helping Zimbabwean university
    graduates transition from education to employment or entrepreneurship.
    Provides practical, actionable guidance with local context.

  target_users:
    - Recent university graduates in Zimbabwe
    - Final year students preparing for job market
    - Young entrepreneurs starting their first business

  core_questions:
    - "How do I register a business in Zimbabwe?"
    - "What freelance services can I offer with my degree?"
    - "How much capital do I need to start a small business?"
    - "Where can I find my first clients?"
    - "What are ZIMRA requirements for small business?"
    - "How do I accept payments via EcoCash?"

  relevant_topics:
    - Zimbabwe business registration
    - Private Business Corporation (PBC)
    - ZIMRA tax requirements
    - EcoCash merchant integration
    - Freelancing in Africa
    - Startup funding Zimbabwe
    - Career paths by degree
    - Agricultural value addition
    - Export procedures Zimbabwe

  irrelevant_topics:
    - US business regulations
    - EU tax law
    - Cryptocurrency trading
    - Forex trading
    - Multi-level marketing
    - Large enterprise solutions

  geographic_focus:
    primary: Zimbabwe
    secondary:
      - Southern Africa
      - Sub-Saharan Africa

  quality_requirements:
    min_relevance_score: 0.6
    review_threshold: 0.4
    freshness_cutoff: "2022-01-01"
```

Usage:

```python
import asyncio
from gweta.intelligence import Pipeline, SystemIntent
from gweta import ChromaStore, GwetaCrawler

async def build_simuka_kb():
    # Load intent
    intent = SystemIntent.from_yaml("intents/simuka.yaml")

    # Create store and pipeline
    store = ChromaStore(collection_name="simuka-kb")
    pipeline = Pipeline(intent=intent, store=store)

    # Crawl Zimbabwe business resources
    crawler = GwetaCrawler()
    crawl_result = await crawler.crawl(
        url="https://www.zimra.co.zw",
        depth=2,
    )

    # Ingest with filtering
    result = await pipeline.ingest(crawl_result.chunks)

    print(f"Crawled: {crawl_result.pages_crawled} pages")
    print(f"Chunks: {len(crawl_result.chunks)}")
    print(f"Ingested: {result.ingested} relevant chunks")
    print(f"Rejected: {result.rejected_count} irrelevant")
    print(f"Acceptance rate: {result.acceptance_rate:.0%}")

asyncio.run(build_simuka_kb())
```

## Best Practices

### 1. Write Good Core Questions

Core questions heavily influence relevance scoring. Write questions that:
- Represent what users actually ask
- Cover the breadth of your domain
- Are specific enough to be meaningful

```yaml
# Good
core_questions:
  - "How do I register a Private Business Corporation in Zimbabwe?"
  - "What are the ZIMRA tax requirements for freelancers?"

# Too vague
core_questions:
  - "How do I start a business?"
  - "What about taxes?"
```

### 2. Be Specific with Irrelevant Topics

Irrelevant topics are checked as substrings. Be specific:

```yaml
# Good - specific
irrelevant_topics:
  - cryptocurrency trading
  - forex signals
  - US tax law

# Risky - might reject valid content
irrelevant_topics:
  - money  # Would reject "how to make money freelancing"
  - tax    # Would reject "ZIMRA tax requirements"
```

### 3. Tune Thresholds for Your Domain

Start with defaults and adjust:

```yaml
quality_requirements:
  # Strict - fewer chunks, higher quality
  min_relevance_score: 0.7
  review_threshold: 0.5

  # Lenient - more chunks, some noise
  min_relevance_score: 0.5
  review_threshold: 0.3
```

### 4. Use the Review Queue

Chunks scoring between `review_threshold` and `min_relevance_score` go to a review queue:

```python
report = filter.filter_batch(chunks)

# Handle review queue
for result in report.for_review():
    print(f"Review: {result.chunk.text[:100]}...")
    print(f"Score: {result.relevance_score:.2f}")
    # Manually decide to include or exclude
```

### 5. Preview Before Ingesting

Always preview on a sample before full ingestion:

```python
# Test on first 100 chunks
sample = chunks[:100]
report = pipeline.filter_only(sample)

print(f"Acceptance rate: {report.acceptance_rate:.0%}")

# Check rejected chunks make sense
for r in report.results:
    if r.rejected:
        print(f"Rejected: {r.chunk.text[:80]}...")
        print(f"  Reason: {r.rejection_reason}")
```

## Performance

| Operation | Time (1000 chunks) |
|-----------|-------------------|
| Load model (first time) | ~2-5 seconds |
| Embed intent | ~10ms |
| Filter 1000 chunks | ~3-5 seconds |
| Memory usage | ~300MB (model + embeddings) |

The embedding model is loaded lazily on first use and cached for subsequent operations.

## Next Steps

- [Full Pipeline Example](../examples/full-pipeline.md) - End-to-end usage
- [API Reference](../api/reference.md) - Complete API documentation
- [Architecture](../concepts/architecture.md) - How Gweta works
