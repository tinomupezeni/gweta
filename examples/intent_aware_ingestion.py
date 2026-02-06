"""Intent-Aware Ingestion Example.

This example demonstrates Gweta's intelligence layer - filtering
content based on relevance to your system's purpose.

## What This Example Does

1. Loads a system intent (what your RAG system is meant to do)
2. Creates sample chunks (some relevant, some not)
3. Filters chunks based on relevance to the intent
4. Shows which chunks would be ingested vs rejected

## Prerequisites

```bash
pip install gweta[intelligence]
```
"""

import asyncio
from pathlib import Path

from gweta import Chunk
from gweta.intelligence import (
    SystemIntent,
    RelevanceFilter,
    Pipeline,
)


def create_sample_chunks() -> list[Chunk]:
    """Create sample chunks for demonstration."""
    return [
        # RELEVANT - Zimbabwe business content
        Chunk(
            id="chunk-001",
            text="""To register a Private Business Corporation (PBC) in Zimbabwe,
            you need to: 1) Reserve your company name at the Companies Registry
            for $20. 2) Prepare Form CR6 and company regulations. 3) Submit to
            the Registrar with $50-80 fee. 4) Receive certificate in 7-10 days.
            5) Register with ZIMRA within 30 days for tax purposes.""",
            source="zimra-guide",
            metadata={"topic": "business-registration"},
        ),
        # RELEVANT - Local freelancing
        Chunk(
            id="chunk-002",
            text="""Freelance web developers in Zimbabwe can charge $200-500 for
            basic websites and $500-2000 for complex applications. Accept payments
            via EcoCash (merchant registration is free) or USD bank transfer.
            Join Impact Hub Harare and TechZim communities for networking.""",
            source="tech-guide",
            metadata={"topic": "freelancing", "sector": "IT"},
        ),
        # RELEVANT - Agriculture
        Chunk(
            id="chunk-003",
            text="""Value addition opportunities for agriculture graduates include
            honey processing ($200-500 startup), dried fruit production ($300-800),
            and peanut butter manufacturing ($500-1500). Contact ZIMTRADE for
            export assistance and certification requirements.""",
            source="agri-guide",
            metadata={"topic": "agribusiness", "sector": "agriculture"},
        ),
        # IRRELEVANT - US regulations
        Chunk(
            id="chunk-004",
            text="""To form an LLC in Delaware, you need to file a Certificate
            of Formation with the Delaware Division of Corporations. The filing
            fee is $90. Delaware is popular due to its business-friendly laws
            and Court of Chancery.""",
            source="us-business-guide",
            metadata={"topic": "us-business"},
        ),
        # IRRELEVANT - Cryptocurrency
        Chunk(
            id="chunk-005",
            text="""Invest in Bitcoin and Ethereum for passive income. Our
            cryptocurrency trading course teaches you how to make $1000/day
            trading crypto. Sign up now for our exclusive forex signals.""",
            source="crypto-spam",
            metadata={"topic": "cryptocurrency"},
        ),
        # MARGINALLY RELEVANT - Generic business
        Chunk(
            id="chunk-006",
            text="""Starting a business requires careful planning. Create a
            business plan, identify your target market, and secure funding.
            Marketing is essential for attracting customers.""",
            source="generic-guide",
            metadata={"topic": "business-general"},
        ),
        # RELEVANT - EcoCash
        Chunk(
            id="chunk-007",
            text="""EcoCash merchant registration is free and can be done at any
            Econet shop. You'll receive a merchant code for till payments.
            Transaction fees are 1-2%. Settlement to bank accounts takes 24-48
            hours. EcoCash serves 6.7 million Zimbabweans.""",
            source="ecocash-guide",
            metadata={"topic": "payments"},
        ),
        # IRRELEVANT - Corporate/Enterprise
        Chunk(
            id="chunk-008",
            text="""Enterprise resource planning (ERP) systems like SAP and Oracle
            help large corporations manage their operations. Implementation
            typically costs $500,000-$5,000,000 and takes 12-24 months.""",
            source="enterprise-guide",
            metadata={"topic": "enterprise"},
        ),
    ]


async def demo_relevance_filter():
    """Demonstrate the RelevanceFilter."""
    print("=" * 70)
    print("GWETA INTELLIGENCE LAYER DEMO")
    print("Intent-Aware Content Filtering")
    print("=" * 70)

    # Load or create intent
    intent_path = Path(__file__).parent / "intents" / "simuka.yaml"

    if intent_path.exists():
        print(f"\nLoading intent from: {intent_path}")
        intent = SystemIntent.from_yaml(intent_path)
    else:
        print("\nCreating intent programmatically...")
        intent = SystemIntent(
            name="Simuka Career Platform",
            description="Career guidance for Zimbabwean graduates",
            core_questions=[
                "How do I register a business in Zimbabwe?",
                "What freelance services can I offer?",
                "How do I accept EcoCash payments?",
            ],
            relevant_topics=[
                "Zimbabwe business registration",
                "ZIMRA",
                "EcoCash",
                "freelancing",
                "entrepreneurship",
            ],
            irrelevant_topics=[
                "US business",
                "cryptocurrency",
                "forex",
                "enterprise",
            ],
        )

    print(f"\nIntent: {intent.name}")
    print(f"Core questions: {len(intent.core_questions)}")
    print(f"Relevant topics: {len(intent.relevant_topics)}")
    print(f"Irrelevant topics: {len(intent.irrelevant_topics)}")

    # Create filter
    print("\n" + "-" * 70)
    print("Creating RelevanceFilter (loading embedding model)...")
    print("-" * 70)

    filter = RelevanceFilter(intent)

    # Create sample chunks
    chunks = create_sample_chunks()
    print(f"\nProcessing {len(chunks)} sample chunks...")

    # Filter chunks
    report = filter.filter_batch(chunks)

    # Display results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"\nTotal chunks: {report.total_chunks}")
    print(f"Accepted: {report.accepted_count} ({report.acceptance_rate:.0%})")
    print(f"Review: {report.review_count}")
    print(f"Rejected: {report.rejected_count} ({report.rejection_rate:.0%})")
    print(f"Avg relevance score: {report.avg_relevance_score:.2f}")

    print("\n" + "-" * 70)
    print("ACCEPTED CHUNKS (would be ingested)")
    print("-" * 70)

    for result in report.results:
        if result.accepted:
            print(f"\n[{result.chunk.id}] Score: {result.relevance_score:.2f}")
            print(f"  Source: {result.chunk.source}")
            print(f"  Topics: {', '.join(result.matched_topics) or 'none detected'}")
            preview = result.chunk.text[:80].replace("\n", " ")
            print(f"  Preview: {preview}...")

    print("\n" + "-" * 70)
    print("REJECTED CHUNKS (would be filtered out)")
    print("-" * 70)

    for result in report.results:
        if result.rejected:
            print(f"\n[{result.chunk.id}] Score: {result.relevance_score:.2f}")
            print(f"  Source: {result.chunk.source}")
            print(f"  Reason: {result.rejection_reason}")
            preview = result.chunk.text[:80].replace("\n", " ")
            print(f"  Preview: {preview}...")

    print("\n" + "-" * 70)
    print("REVIEW QUEUE (needs manual review)")
    print("-" * 70)

    review_chunks = [r for r in report.results if r.needs_review]
    if review_chunks:
        for result in review_chunks:
            print(f"\n[{result.chunk.id}] Score: {result.relevance_score:.2f}")
            print(f"  Source: {result.chunk.source}")
    else:
        print("\nNo chunks in review queue.")

    # Show what would be ingested
    accepted_chunks = report.accepted()
    print("\n" + "=" * 70)
    print(f"READY FOR INGESTION: {len(accepted_chunks)} chunks")
    print("=" * 70)

    for chunk in accepted_chunks:
        print(f"  - {chunk.id}: relevance={chunk.metadata.get('relevance_score', 0):.2f}")


async def demo_pipeline():
    """Demonstrate the unified Pipeline."""
    print("\n" + "=" * 70)
    print("PIPELINE DEMO")
    print("=" * 70)

    intent = SystemIntent(
        name="Demo System",
        description="Demo intent for testing",
        core_questions=["How to register business in Zimbabwe?"],
        relevant_topics=["Zimbabwe", "business", "ZIMRA"],
        irrelevant_topics=["cryptocurrency", "US regulations"],
    )

    # Pipeline without store (filter only)
    pipeline = Pipeline(intent=intent, store=None)

    chunks = create_sample_chunks()

    print(f"\nProcessing {len(chunks)} chunks through pipeline...")

    # Use filter_only to preview
    report = pipeline.filter_only(chunks)

    print(f"\nQuality + Relevance filtering:")
    print(f"  Would ingest: {report.accepted_count}")
    print(f"  Would reject: {report.rejected_count}")

    # Score a single chunk
    print("\n" + "-" * 70)
    print("SINGLE CHUNK SCORING")
    print("-" * 70)

    test_chunk = Chunk(
        text="Register your business with ZIMRA within 30 days of incorporation.",
        source="test",
    )

    scores = pipeline.score_chunk(test_chunk)
    print(f"\nChunk: '{scores['text_preview']}'")
    print(f"Quality score: {scores['quality_score']}")
    print(f"Relevance score: {scores['relevance_score']:.2f}")
    print(f"Decision: {scores['relevance_decision']}")
    print(f"Would ingest: {scores['would_ingest']}")


async def main():
    """Run all demos."""
    await demo_relevance_filter()
    await demo_pipeline()

    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    print("\nKey takeaways:")
    print("1. Define your system's intent in YAML")
    print("2. Gweta automatically filters irrelevant content")
    print("3. Only relevant chunks reach your vector store")
    print("4. Better data = better RAG answers")


if __name__ == "__main__":
    asyncio.run(main())
