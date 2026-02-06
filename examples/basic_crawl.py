"""Basic web crawling example.

Demonstrates how to use GwetaCrawler to fetch and validate web content.

Note: This example requires crawl4ai to be installed:
    pip install crawl4ai
"""

import asyncio

from gweta.acquire.crawler import GwetaCrawler
from gweta.core.registry import SourceAuthorityRegistry


async def main():
    # Create authority registry with trusted sources
    registry = SourceAuthorityRegistry()
    registry.add_source(
        domain="*.python.org",
        name="Python Documentation",
        authority=5,
        freshness_days=30,
    )
    registry.add_source(
        domain="*.example.com",
        name="Example Site",
        authority=3,
        freshness_days=90,
    )

    # Create crawler with custom settings
    crawler = GwetaCrawler(
        authority_registry=registry,
        quality_threshold=0.6,
        chunk_size=500,
    )

    print("Gweta Web Crawler Example")
    print("=" * 50)

    # Note: This URL is for demonstration. Replace with a real URL to test.
    url = "https://docs.python.org/3/tutorial/index.html"

    print(f"Crawling: {url}")
    print(f"Quality threshold: {crawler.quality_threshold}")
    print()

    try:
        # Crawl the URL
        result = await crawler.crawl(
            url=url,
            depth=1,  # Only crawl the starting page
            max_pages=5,
        )

        # Print results
        print(result.summary())
        print()

        # Show accepted chunks
        if result.chunks:
            print(f"Sample accepted chunks ({len(result.chunks)} total):")
            for i, chunk in enumerate(result.chunks[:3]):
                print(f"\n  Chunk {i + 1}:")
                print(f"    Source: {chunk.source}")
                print(f"    Quality: {chunk.quality_score:.2f}")
                print(f"    Text: {chunk.text[:100]}...")
        else:
            print("No chunks were accepted.")

        # Show rejected chunks
        if result.rejected_chunks:
            print(f"\nRejected chunks: {len(result.rejected_chunks)}")

        # Show any errors
        if result.errors:
            print(f"\nErrors encountered: {len(result.errors)}")
            for error in result.errors[:3]:
                print(f"  - {error.url}: {error.error}")

    except ImportError as e:
        print(f"Import error: {e}")
        print("\nMake sure crawl4ai is installed: pip install crawl4ai")
    except Exception as e:
        print(f"Crawl failed: {e}")


def demo_registry():
    """Demonstrate SourceAuthorityRegistry usage."""
    print("\nSource Authority Registry Demo")
    print("=" * 50)

    registry = SourceAuthorityRegistry()

    # Add trusted sources
    registry.add_source("docs.python.org", "Python Docs", authority=5, freshness_days=30)
    registry.add_source("*.gov.zw", "Zimbabwe Government", authority=4, freshness_days=90)
    registry.add_source("medium.com", "Medium", authority=2, freshness_days=7)

    # Block untrusted sources
    registry.block_domain("spam-site.com")

    # Test URLs
    test_urls = [
        "https://docs.python.org/3/tutorial/",
        "https://finance.gov.zw/budget-2024/",
        "https://medium.com/article/something",
        "https://spam-site.com/bad",
        "https://unknown-site.org/page",
    ]

    print("\nURL Authority Check:")
    for url in test_urls:
        is_allowed = registry.is_allowed(url)
        authority = registry.get_authority(url)
        source = registry.get_source(url)
        status = "ALLOWED" if is_allowed else "BLOCKED"
        print(f"  {url}")
        print(f"    Status: {status}, Authority: {authority}, Source: {source.name}")


if __name__ == "__main__":
    # Run registry demo (doesn't require crawl4ai)
    demo_registry()

    # Run crawler demo
    print("\n")
    asyncio.run(main())
