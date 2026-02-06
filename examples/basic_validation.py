"""Basic chunk validation example.

Demonstrates how to use the ChunkValidator to assess text quality.
"""

from gweta.core.types import Chunk
from gweta.validate.chunks import ChunkValidator, ChunkValidatorConfig


def main():
    # Create sample chunks with varying quality
    chunks = [
        Chunk(
            text=(
                "This is a well-written paragraph that provides substantial "
                "information about the topic. It has proper structure and "
                "delivers meaningful content to the reader."
            ),
            source="docs/intro.md",
            metadata={"section": "introduction"},
        ),
        Chunk(
            text="Short text",
            source="docs/bad.md",
            metadata={},
        ),
        Chunk(
            text=(
                "Another quality chunk with detailed explanations and "
                "examples. This content is informative and well-organized, "
                "making it suitable for knowledge base inclusion."
            ),
            source="docs/guide.md",
            metadata={"section": "tutorial"},
        ),
    ]

    # Create validator with default config
    validator = ChunkValidator()
    print(f"Validator config: min_quality={validator.config.min_quality_score}")
    print()

    # Validate batch
    report = validator.validate_batch(chunks)

    # Print summary
    print("Validation Report")
    print("=================")
    print(f"Total chunks: {report.total_chunks}")
    print(f"Passed: {report.passed}")
    print(f"Failed: {report.failed}")
    print(f"Avg quality: {report.avg_quality_score:.2f}")
    print()

    # Show individual results
    print("Individual Results:")
    for i, result in enumerate(report.chunks):
        status = "PASS" if result.passed else "FAIL"
        print(f"  [{status}] Chunk {i+1}: score={result.quality_score:.2f}")
        if result.issues:
            for issue in result.issues:
                print(f"         -> {issue.code}: {issue.message}")

    # Get accepted chunks
    accepted = report.accepted()
    print()
    print(f"Accepted {len(accepted)} chunks for ingestion")


if __name__ == "__main__":
    main()
