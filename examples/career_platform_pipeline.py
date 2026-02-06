"""Career Activation Platform - Gweta Data Pipeline

This example demonstrates how Gweta serves as the data acquisition,
validation, and ingestion engine for an AI-powered Career Platform
targeting Zimbabwean graduates.

The pipeline:
1. Acquires data from multiple Zimbabwe-specific sources
2. Validates for relevance, freshness, and accuracy
3. Enriches with sector/pathway metadata
4. Ingests into vector store for RAG retrieval

Usage:
    pip install gweta[all]
    python career_platform_pipeline.py
"""

import asyncio
from pathlib import Path
from datetime import date, timedelta

from gweta import Chunk, ChunkValidator, ChromaStore
from gweta.acquire import GwetaCrawler, PDFExtractor, APIClient
from gweta.validate.rules import DomainRuleEngine
from gweta.validate.health import HealthChecker
from gweta.ingest.chunkers import RecursiveChunker
from gweta.core.logging import get_logger

logger = get_logger(__name__)


# ============================================================
# CONFIGURATION
# ============================================================

# Zimbabwe data sources for the Career Platform
SOURCES = {
    "regulations": [
        {
            "url": "https://www.zimra.co.zw/domestic-taxes",
            "name": "ZIMRA Tax Guide",
            "sector": "all",
            "doc_type": "regulation",
            "authority": 1.0,
            "refresh_days": 30,
        },
        {
            "url": "https://www.companies.gov.zw/",
            "name": "Companies Registry",
            "sector": "all",
            "doc_type": "regulation",
            "authority": 1.0,
            "refresh_days": 30,
        },
    ],
    "market_intelligence": [
        {
            "url": "https://www.techzim.co.zw/category/startups/",
            "name": "TechZim Startups",
            "sector": "it",
            "doc_type": "market",
            "authority": 0.7,
            "refresh_days": 7,
        },
    ],
    "incubators": [
        {
            "url": "https://www.impacthubharare.net/",
            "name": "Impact Hub Harare",
            "sector": "all",
            "doc_type": "resource",
            "authority": 0.85,
            "refresh_days": 30,
        },
    ],
}

# Sector-to-pathway mapping
SECTOR_PATHWAYS = {
    "it": ["employment", "freelance", "startup"],
    "commerce": ["employment", "freelance", "startup"],
    "agriculture": ["employment", "agripreneur", "cooperative"],
    "education": ["employment", "tutoring", "edtech"],
    "engineering": ["employment", "consulting", "contracting"],
    "health": ["employment", "private_practice", "telehealth"],
}


# ============================================================
# DATA ACQUISITION
# ============================================================

async def acquire_web_sources(sources: list[dict]) -> list[Chunk]:
    """Crawl web sources and extract chunks."""
    crawler = GwetaCrawler()
    chunker = RecursiveChunker(
        chunk_size=250,  # 100-300 words optimal for RAG
        chunk_overlap=50,  # ~20% overlap
    )

    all_chunks = []

    for source in sources:
        logger.info(f"Crawling: {source['name']}")

        try:
            result = await crawler.crawl(
                url=source["url"],
                depth=2,
                allowed_domains=[source["url"].split("/")[2]],
            )

            # Enrich chunks with source metadata
            for chunk in result.chunks:
                chunk.metadata.update({
                    "source_name": source["name"],
                    "sector": source["sector"],
                    "doc_type": source["doc_type"],
                    "authority_score": source["authority"],
                    "refresh_days": source["refresh_days"],
                    "acquired_date": date.today().isoformat(),
                    "stale_after": (
                        date.today() + timedelta(days=source["refresh_days"])
                    ).isoformat(),
                })
                all_chunks.append(chunk)

            logger.info(f"  Acquired {len(result.chunks)} chunks")

        except Exception as e:
            logger.warning(f"  Failed to crawl {source['name']}: {e}")

    return all_chunks


async def acquire_pdf_content(pdf_dir: Path) -> list[Chunk]:
    """Extract content from PDF documents (regulations, guides)."""
    extractor = PDFExtractor()
    all_chunks = []

    if not pdf_dir.exists():
        logger.info(f"PDF directory not found: {pdf_dir}")
        return all_chunks

    for pdf_path in pdf_dir.glob("*.pdf"):
        logger.info(f"Extracting: {pdf_path.name}")

        try:
            result = await extractor.extract(
                source=pdf_path,
                extract_tables=True,
            )

            # Infer sector from filename
            sector = "all"
            for s in SECTOR_PATHWAYS.keys():
                if s in pdf_path.stem.lower():
                    sector = s
                    break

            for chunk in result.chunks:
                chunk.metadata.update({
                    "source_name": pdf_path.stem,
                    "sector": sector,
                    "doc_type": "guide",
                    "authority_score": 0.8,
                    "acquired_date": date.today().isoformat(),
                })
                all_chunks.append(chunk)

            logger.info(f"  Extracted {len(result.chunks)} chunks")

        except Exception as e:
            logger.warning(f"  Failed to extract {pdf_path.name}: {e}")

    return all_chunks


# ============================================================
# VALIDATION PIPELINE
# ============================================================

def setup_zimbabwe_rules() -> DomainRuleEngine:
    """Load Zimbabwe-specific validation rules."""
    rules_path = Path(__file__).parent / "rules" / "zimbabwe_career.yaml"

    if rules_path.exists():
        return DomainRuleEngine.from_yaml(str(rules_path))
    else:
        logger.warning("Zimbabwe rules not found, using basic validation")
        return DomainRuleEngine(rules=[], known_facts=[])


async def validate_for_career_platform(
    chunks: list[Chunk],
    rule_engine: DomainRuleEngine,
) -> list[Chunk]:
    """
    Validate chunks for career platform relevance.

    Validation layers:
    1. Basic quality (length, coherence, density)
    2. Domain rules (Zimbabwe-specific validation)
    3. Freshness check (not stale)
    4. Sector relevance scoring
    """
    logger.info(f"Validating {len(chunks)} chunks...")

    # Layer 1: Basic quality validation
    validator = ChunkValidator(
        min_length=50,
        min_density=0.3,
    )
    report = validator.validate_batch(chunks)

    logger.info(f"  Basic validation: {report.passed}/{report.total_chunks} passed")
    logger.info(f"  Average quality: {report.avg_quality_score:.2f}")

    # Get chunks that passed basic validation
    passed_chunks = [
        r.chunk for r in report.chunks
        if r.passed and r.quality_score >= 0.6
    ]

    # Layer 2: Domain rules validation
    validated_chunks = []
    domain_failures = 0

    for chunk in passed_chunks:
        result = rule_engine.validate_chunk(chunk)

        if result.passed:
            # Add validation metadata
            chunk.metadata["domain_validated"] = True
            chunk.metadata["domain_score"] = result.score
            validated_chunks.append(chunk)
        else:
            domain_failures += 1
            # Log rejections for review
            logger.debug(f"  Domain rejection: {result.issues}")

    logger.info(f"  Domain validation: {len(validated_chunks)} passed, {domain_failures} failed")

    # Layer 3: Freshness check
    fresh_chunks = []
    stale_count = 0
    today = date.today().isoformat()

    for chunk in validated_chunks:
        stale_after = chunk.metadata.get("stale_after", "2099-12-31")
        if today <= stale_after:
            fresh_chunks.append(chunk)
        else:
            stale_count += 1

    if stale_count > 0:
        logger.warning(f"  Freshness check: {stale_count} stale chunks filtered")

    # Layer 4: Sector relevance enrichment
    for chunk in fresh_chunks:
        sector = chunk.metadata.get("sector", "all")
        if sector != "all" and sector in SECTOR_PATHWAYS:
            chunk.metadata["pathways"] = SECTOR_PATHWAYS[sector]

    logger.info(f"  Final validated chunks: {len(fresh_chunks)}")
    return fresh_chunks


# ============================================================
# INGESTION
# ============================================================

async def ingest_to_knowledge_base(
    chunks: list[Chunk],
    collection_name: str = "career_platform_kb",
) -> ChromaStore:
    """Ingest validated chunks to vector store."""
    logger.info(f"Ingesting {len(chunks)} chunks to '{collection_name}'...")

    store = ChromaStore(
        collection_name=collection_name,
        persist_directory="./career_platform_data",
    )

    result = await store.add(chunks)

    logger.info(f"  Added: {result.added}")
    logger.info(f"  Skipped (duplicates): {result.skipped}")

    return store


async def run_health_check(store: ChromaStore) -> None:
    """Monitor knowledge base health."""
    logger.info("Running knowledge base health check...")

    checker = HealthChecker(store)
    report = await checker.full_health_check()

    logger.info(f"  Total chunks: {report.total_chunks}")
    logger.info(f"  Average quality: {report.avg_quality_score:.2f}")
    logger.info(f"  Duplicate groups: {report.duplicates.duplicate_groups}")

    if report.stale_chunks:
        logger.warning(f"  Stale chunks needing refresh: {len(report.stale_chunks)}")

    if report.recommendations:
        logger.info("  Recommendations:")
        for rec in report.recommendations[:5]:
            logger.info(f"    - {rec}")


# ============================================================
# MAIN PIPELINE
# ============================================================

async def main():
    """Run the complete Career Platform data pipeline."""
    print("=" * 70)
    print("CAREER ACTIVATION PLATFORM - GWETA DATA PIPELINE")
    print("Acquire → Validate → Ingest")
    print("=" * 70)

    all_chunks = []

    # ===== ACQUISITION PHASE =====
    print("\n[PHASE 1] DATA ACQUISITION")
    print("-" * 40)

    # Web sources
    for category, sources in SOURCES.items():
        print(f"\nAcquiring {category}...")
        try:
            chunks = await acquire_web_sources(sources)
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"  Skipped (error): {e}")

    # PDF documents
    pdf_dir = Path("./documents/guides")
    pdf_chunks = await acquire_pdf_content(pdf_dir)
    all_chunks.extend(pdf_chunks)

    # If no real data, create demo chunks
    if not all_chunks:
        print("\n[DEMO] Creating sample career content...")
        all_chunks = create_demo_chunks()

    print(f"\nTotal acquired: {len(all_chunks)} chunks")

    # ===== VALIDATION PHASE =====
    print("\n[PHASE 2] VALIDATION")
    print("-" * 40)

    rule_engine = setup_zimbabwe_rules()
    validated_chunks = await validate_for_career_platform(all_chunks, rule_engine)

    if not validated_chunks:
        print("\nNo chunks passed validation. Exiting.")
        return

    # ===== INGESTION PHASE =====
    print("\n[PHASE 3] INGESTION")
    print("-" * 40)

    store = await ingest_to_knowledge_base(validated_chunks)

    # ===== HEALTH CHECK =====
    print("\n[PHASE 4] HEALTH CHECK")
    print("-" * 40)

    await run_health_check(store)

    # ===== SUMMARY =====
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"  Chunks acquired:   {len(all_chunks)}")
    print(f"  Chunks validated:  {len(validated_chunks)}")
    print(f"  Rejection rate:    {(1 - len(validated_chunks)/len(all_chunks))*100:.1f}%")
    print(f"  Collection:        career_platform_kb")
    print(f"  Storage:           ./career_platform_data")
    print("\nKnowledge base ready for RAG queries!")


def create_demo_chunks() -> list[Chunk]:
    """Create demo chunks for testing without network."""
    return [
        # Business Registration
        Chunk(
            text="""To register a Private Business Corporation (PBC) in Zimbabwe,
            you need to follow these steps: 1) Reserve your company name at the
            Companies Registry ($20 fee). 2) Prepare incorporation documents
            including Form CR6 and company regulations. 3) Submit to the Registrar
            of Companies with a $50-80 registration fee. 4) Receive your certificate
            within 7-10 working days. 5) Register with ZIMRA within 30 days of
            incorporation for tax purposes. Total cost: approximately $70-100 USD.""",
            source="companies.gov.zw",
            metadata={
                "source_name": "Companies Registry Guide",
                "sector": "all",
                "doc_type": "regulation",
                "authority_score": 1.0,
                "acquired_date": date.today().isoformat(),
            },
        ),
        # IT Freelancing
        Chunk(
            text="""Starting a freelance web development business in Zimbabwe requires
            minimal capital. Key steps: 1) Build a portfolio website showcasing 3-5
            projects. 2) Set competitive pricing ($200-500 for basic websites,
            $500-2000 for complex applications). 3) Join local tech communities
            like TechZim and Impact Hub Harare. 4) Use WhatsApp Business for client
            communication. 5) Accept payments via EcoCash (6.7 million users) or
            USD bank transfer. Initial investment: $50-100 for domain and hosting.""",
            source="techzim.co.zw",
            metadata={
                "source_name": "TechZim Freelance Guide",
                "sector": "it",
                "doc_type": "guide",
                "authority_score": 0.7,
                "pathways": ["freelance", "startup"],
                "acquired_date": date.today().isoformat(),
            },
        ),
        # Accounting Services
        Chunk(
            text="""Commerce and accounting graduates can start bookkeeping services
            for Zimbabwe's SME sector with minimal investment. Target clients:
            informal traders, small retailers, and startups. Services to offer:
            monthly bookkeeping ($30-50/month), ZIMRA tax filing assistance
            ($20-50 per filing), and basic financial consulting ($50-100/session).
            Register as a PBC or operate as sole trader initially. Key tools:
            Excel or free accounting software, WhatsApp for communication,
            EcoCash for payments. The informal sector (80% of economy) is
            underserved by formal accounting services.""",
            source="business-guide",
            metadata={
                "source_name": "Accounting Services Guide",
                "sector": "commerce",
                "doc_type": "guide",
                "authority_score": 0.8,
                "pathways": ["freelance", "startup"],
                "acquired_date": date.today().isoformat(),
            },
        ),
        # Agriculture Value Addition
        Chunk(
            text="""Agricultural value addition opportunities for graduates in Zimbabwe
            include: 1) Honey processing and packaging (startup $200-500).
            2) Dried fruit production from local orchards (startup $300-800).
            3) Peanut butter manufacturing (startup $500-1500).
            4) Mushroom cultivation (startup $100-300).
            Key success factors: quality packaging, HACCP certification for export,
            and connection to local markets via WhatsApp groups. Zimbabwe's
            horticulture exports reached $80M in 2024. Contact ZIMTRADE for
            export assistance and market linkages.""",
            source="agriculture-guide",
            metadata={
                "source_name": "Agri Value Addition Guide",
                "sector": "agriculture",
                "doc_type": "guide",
                "authority_score": 0.8,
                "pathways": ["agripreneur"],
                "acquired_date": date.today().isoformat(),
            },
        ),
        # EcoCash Integration
        Chunk(
            text="""EcoCash integration for small businesses: EcoCash serves 6.7 million
            Zimbabweans, making it essential for any business. To accept EcoCash
            payments: 1) Register as a merchant at any Econet shop (free).
            2) Receive your merchant code for till number payments.
            3) Use Paynow gateway for online payment integration.
            Transaction fees: 1-2% for merchant payments. Settlement to bank
            account within 24-48 hours. Alternative: personal EcoCash for
            micro-businesses (higher fees but instant setup).""",
            source="econet.co.zw",
            metadata={
                "source_name": "EcoCash Business Guide",
                "sector": "all",
                "doc_type": "resource",
                "authority_score": 0.9,
                "acquired_date": date.today().isoformat(),
            },
        ),
    ]


if __name__ == "__main__":
    asyncio.run(main())
