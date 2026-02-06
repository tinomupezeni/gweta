"""PDF extraction example.

Demonstrates how to use PDFExtractor to extract and validate PDF content.

Note: This example requires pdfplumber to be installed:
    pip install pdfplumber
"""

import asyncio
from pathlib import Path

from gweta.acquire.pdf import PDFExtractor


async def main():
    print("Gweta PDF Extraction Example")
    print("=" * 50)

    # Create extractor with custom settings
    extractor = PDFExtractor(
        chunk_size=500,
        chunk_overlap=50,
    )

    # Example: Extract from a PDF file
    # Replace this path with an actual PDF file to test
    pdf_path = Path("sample.pdf")

    if not pdf_path.exists():
        print(f"\nNote: No PDF file found at '{pdf_path}'")
        print("To test PDF extraction, create a sample.pdf file or modify the path.")
        print("\nShowing extraction capabilities instead:")
        demo_extraction_features()
        return

    print(f"\nExtracting from: {pdf_path}")

    try:
        # Extract content from PDF
        result = await extractor.extract(
            source=pdf_path,
            extract_tables=True,
            create_chunks=True,
        )

        # Print summary
        print(f"\n{result.summary()}")

        # Show metadata
        if result.metadata:
            print("\nPDF Metadata:")
            for key, value in list(result.metadata.items())[:5]:
                print(f"  {key}: {value}")

        # Show pages
        print(f"\nPages extracted: {result.total_pages}")
        for page in result.pages[:3]:
            print(f"  Page {page.number}:")
            print(f"    Quality: {page.quality_score:.2f}")
            print(f"    Scanned: {page.is_scanned}")
            print(f"    Text preview: {page.text[:100]}...")

        # Show tables
        if result.tables:
            print(f"\nTables extracted: {len(result.tables)}")
            for i, table in enumerate(result.tables[:2]):
                print(f"  Table {i + 1} (page {table.page}):")
                print(f"    Rows: {len(table.data)}")
                print(f"    Quality: {table.quality_score:.2f}")
                if table.headers:
                    print(f"    Headers: {table.headers}")

        # Show chunks
        print(f"\nChunks created: {len(result.chunks)}")
        for i, chunk in enumerate(result.chunks[:3]):
            print(f"  Chunk {i + 1}:")
            print(f"    Quality: {chunk.quality_score:.2f}")
            print(f"    Page: {chunk.metadata.get('page')}")
            print(f"    Text: {chunk.text[:80]}...")

        # Show issues
        if result.issues:
            print(f"\nQuality issues: {len(result.issues)}")
            for issue in result.issues:
                print(f"  [{issue.severity}] {issue.code}: {issue.message}")

    except ImportError as e:
        print(f"Import error: {e}")
        print("\nMake sure pdfplumber is installed: pip install pdfplumber")
    except Exception as e:
        print(f"Extraction failed: {e}")


def demo_extraction_features():
    """Demonstrate extraction features without a real PDF."""
    print("\nPDFExtractor Features:")
    print("-" * 40)
    print("""
    1. Text Extraction:
       - Extracts text from all pages
       - Detects scanned/OCR pages
       - Scores page quality

    2. Table Extraction:
       - Detects and extracts tables
       - Identifies headers
       - Scores table quality

    3. Quality Validation:
       - Checks for OCR issues
       - Validates content completeness
       - Reports quality issues

    4. Automatic Chunking:
       - Splits content into chunks
       - Preserves page metadata
       - Applies quality scores

    Example usage:

        from gweta.acquire.pdf import PDFExtractor

        extractor = PDFExtractor(chunk_size=500)
        result = await extractor.extract("document.pdf")

        print(f"Pages: {result.total_pages}")
        print(f"Tables: {len(result.tables)}")
        print(f"Chunks: {len(result.chunks)}")

        # Access chunks for ingestion
        for chunk in result.chunks:
            print(chunk.text, chunk.quality_score)
    """)


if __name__ == "__main__":
    asyncio.run(main())
