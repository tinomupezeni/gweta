"""Full ingestion pipeline.

This module provides the IngestionPipeline class that
orchestrates the complete flow from acquisition to
vector store loading.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from gweta.core.config import GwetaSettings, get_settings
from gweta.core.logging import get_logger
from gweta.core.types import Chunk, QualityReport
from gweta.ingest.chunkers import RecursiveChunker, get_chunker
from gweta.ingest.stores.base import BaseStore
from gweta.validate.chunks import ChunkValidator
from gweta.validate.extraction import ExtractionValidator

logger = get_logger(__name__)


@dataclass
class PipelineResult:
    """Results from a pipeline run.

    Attributes:
        source: Source that was processed
        chunks_created: Total chunks created
        chunks_loaded: Chunks that passed validation and were loaded
        chunks_rejected: Chunks that failed validation
        quality_report: Validation report
        duration_seconds: Total processing time
    """
    source: str
    chunks_created: int = 0
    chunks_loaded: int = 0
    chunks_rejected: int = 0
    quality_report: QualityReport | None = None
    duration_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """Generate human-readable summary."""
        return (
            f"Pipeline Results for {self.source}\n"
            f"  Chunks Created: {self.chunks_created}\n"
            f"  Chunks Loaded: {self.chunks_loaded}\n"
            f"  Chunks Rejected: {self.chunks_rejected}\n"
            f"  Duration: {self.duration_seconds:.1f}s"
        )


class IngestionPipeline:
    """Complete ingestion pipeline from text to vector store.

    Orchestrates:
    1. Extraction validation
    2. Chunking
    3. Chunk validation
    4. Vector store loading

    Example:
        >>> pipeline = IngestionPipeline(store=chroma_store)
        >>> result = await pipeline.ingest_text(
        ...     text="Long document content...",
        ...     source="document.pdf",
        ... )
        >>> print(result.summary())
    """

    def __init__(
        self,
        store: BaseStore,
        config: GwetaSettings | None = None,
        chunker: Any | None = None,
        extraction_validator: ExtractionValidator | None = None,
        chunk_validator: ChunkValidator | None = None,
    ) -> None:
        """Initialize IngestionPipeline.

        Args:
            store: Vector store to load chunks into
            config: Gweta settings
            chunker: Chunker to use (defaults to RecursiveChunker)
            extraction_validator: Extraction validator
            chunk_validator: Chunk validator
        """
        self.store = store
        self.config = config or get_settings()
        self.chunker = chunker or RecursiveChunker()
        self.extraction_validator = extraction_validator or ExtractionValidator(self.config)
        self.chunk_validator = chunk_validator or ChunkValidator(self.config)

    async def ingest_text(
        self,
        text: str,
        source: str = "",
        metadata: dict[str, Any] | None = None,
        skip_extraction_validation: bool = False,
    ) -> PipelineResult:
        """Ingest raw text through the full pipeline.

        Args:
            text: Text content to ingest
            source: Source identifier
            metadata: Metadata to attach to chunks
            skip_extraction_validation: Whether to skip extraction validation

        Returns:
            PipelineResult with ingestion statistics
        """
        start_time = datetime.now()
        result = PipelineResult(source=source)
        metadata = metadata or {}

        try:
            # Step 1: Validate extraction (optional)
            if not skip_extraction_validation:
                extraction_result = self.extraction_validator.validate(text)
                if not extraction_result.is_acceptable:
                    result.errors.append(
                        f"Extraction validation failed: {extraction_result.issues}"
                    )
                    # Continue anyway but log warning
                    logger.warning(f"Extraction issues for {source}: {extraction_result.issues}")

            # Step 2: Chunk the text
            chunks = self.chunker.chunk(text, metadata=metadata, source=source)
            result.chunks_created = len(chunks)

            if not chunks:
                result.errors.append("No chunks created from text")
                return result

            # Step 3: Validate chunks
            quality_report = self.chunk_validator.validate_batch(chunks)
            result.quality_report = quality_report

            # Step 4: Load accepted chunks
            accepted = quality_report.accepted()
            result.chunks_loaded = len(accepted)
            result.chunks_rejected = len(quality_report.rejected())

            if accepted:
                await self.store.add(accepted)
                logger.info(f"Loaded {len(accepted)} chunks from {source}")

        except Exception as e:
            logger.error(f"Pipeline error for {source}: {e}")
            result.errors.append(str(e))

        result.duration_seconds = (datetime.now() - start_time).total_seconds()
        return result

    async def ingest_chunks(
        self,
        chunks: list[Chunk],
        validate: bool = True,
    ) -> PipelineResult:
        """Ingest pre-chunked content.

        Args:
            chunks: Pre-created chunks
            validate: Whether to validate chunks

        Returns:
            PipelineResult
        """
        start_time = datetime.now()
        source = chunks[0].source if chunks else "unknown"
        result = PipelineResult(source=source, chunks_created=len(chunks))

        try:
            if validate:
                quality_report = self.chunk_validator.validate_batch(chunks)
                result.quality_report = quality_report
                accepted = quality_report.accepted()
            else:
                accepted = chunks

            result.chunks_loaded = len(accepted)
            result.chunks_rejected = len(chunks) - len(accepted)

            if accepted:
                await self.store.add(accepted)

        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            result.errors.append(str(e))

        result.duration_seconds = (datetime.now() - start_time).total_seconds()
        return result

    def ingest_text_sync(
        self,
        text: str,
        **kwargs: Any,
    ) -> PipelineResult:
        """Synchronous wrapper for ingest_text().

        Args:
            text: Text to ingest
            **kwargs: Additional arguments

        Returns:
            PipelineResult
        """
        import asyncio
        return asyncio.run(self.ingest_text(text, **kwargs))
