"""PDF extraction with quality validation.

This module provides the PDFExtractor class for extracting
text and tables from PDF documents with quality scoring.
"""

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from gweta.core.config import GwetaSettings, get_settings
from gweta.core.exceptions import PDFExtractionError
from gweta.core.logging import get_logger
from gweta.core.types import Chunk, QualityIssue

logger = get_logger(__name__)


@dataclass
class PDFPage:
    """Extracted content from a single PDF page.

    Attributes:
        number: Page number (1-indexed)
        text: Extracted text content
        quality_score: Quality score for this page (0.0-1.0)
        is_scanned: Whether page appears to be scanned/OCR'd
        ocr_confidence: Estimated OCR confidence if scanned
    """
    number: int
    text: str
    quality_score: float = 1.0
    is_scanned: bool = False
    ocr_confidence: float | None = None


@dataclass
class PDFTable:
    """Extracted table from a PDF.

    Attributes:
        page: Page number where table was found
        data: Table data as list of rows (each row is list of cells)
        quality_score: Extraction quality score
        headers: Detected header row (if any)
    """
    page: int
    data: list[list[str]]
    quality_score: float = 1.0
    headers: list[str] | None = None


@dataclass
class PDFExtractionResult:
    """Results from PDF extraction.

    Attributes:
        pages: List of extracted pages
        tables: List of extracted tables
        metadata: PDF metadata (title, author, etc.)
        quality_score: Overall extraction quality
        issues: List of quality issues found
        chunks: Pre-chunked content ready for ingestion
    """
    pages: list[PDFPage] = field(default_factory=list)
    tables: list[PDFTable] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    quality_score: float = 1.0
    issues: list[QualityIssue] = field(default_factory=list)
    chunks: list[Chunk] = field(default_factory=list)

    @property
    def total_pages(self) -> int:
        """Get total number of pages."""
        return len(self.pages)

    @property
    def full_text(self) -> str:
        """Get full document text."""
        return "\n\n".join(page.text for page in self.pages)

    def summary(self) -> str:
        """Generate human-readable summary."""
        return (
            f"PDF Extraction Results\n"
            f"  Pages: {self.total_pages}\n"
            f"  Tables: {len(self.tables)}\n"
            f"  Quality Score: {self.quality_score:.2f}\n"
            f"  Issues: {len(self.issues)}"
        )


class PDFExtractor:
    """Extract and validate PDF content.

    Uses pdfplumber for text and table extraction with
    quality scoring for OCR confidence and completeness.

    Example:
        >>> extractor = PDFExtractor()
        >>> result = await extractor.extract("document.pdf")
        >>> print(result.summary())
        >>> for chunk in result.chunks:
        ...     print(chunk.text[:100])
    """

    def __init__(
        self,
        config: GwetaSettings | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
        """Initialize PDFExtractor.

        Args:
            config: Gweta settings
            chunk_size: Size of chunks in characters
            chunk_overlap: Overlap between chunks
        """
        self.config = config or get_settings()
        self.chunk_size = chunk_size or self.config.default_chunk_size
        self.chunk_overlap = chunk_overlap or self.config.default_chunk_overlap

    async def extract(
        self,
        source: Path | str | bytes,
        extract_tables: bool = True,
        extract_images: bool = False,
        create_chunks: bool = True,
    ) -> PDFExtractionResult:
        """Extract content from a PDF file.

        Args:
            source: Path to PDF file, or PDF bytes
            extract_tables: Whether to extract tables
            extract_images: Whether to extract images (not yet implemented)
            create_chunks: Whether to create chunks from content

        Returns:
            PDFExtractionResult with extracted content

        Raises:
            PDFExtractionError: If extraction fails
        """
        try:
            import pdfplumber
        except ImportError as e:
            raise ImportError(
                "pdfplumber is required for PDF extraction. "
                "Install it with: pip install pdfplumber"
            ) from e

        result = PDFExtractionResult()
        file_path: str | None = None

        try:
            # Handle different source types
            if isinstance(source, bytes):
                import io
                pdf_file = io.BytesIO(source)
            else:
                file_path = str(source)
                if not Path(file_path).exists():
                    raise PDFExtractionError(
                        f"PDF file not found: {file_path}",
                        file_path=file_path,
                    )
                pdf_file = file_path

            with pdfplumber.open(pdf_file) as pdf:
                # Extract metadata
                result.metadata = pdf.metadata or {}

                # Process each page
                for page_num, page in enumerate(pdf.pages, start=1):
                    extracted_page = self._extract_page(page, page_num)
                    result.pages.append(extracted_page)

                    # Extract tables if requested
                    if extract_tables:
                        tables = self._extract_tables(page, page_num)
                        result.tables.extend(tables)

            # Calculate overall quality score
            if result.pages:
                total_score = sum(p.quality_score for p in result.pages)
                result.quality_score = total_score / len(result.pages)

            # Validate and create issues
            result.issues = self._validate_extraction(result)

            # Create chunks if requested
            if create_chunks:
                result.chunks = self._create_chunks(result, file_path)

            logger.info(f"PDF extraction completed: {result.summary()}")

        except PDFExtractionError:
            raise
        except Exception as e:
            raise PDFExtractionError(
                f"Failed to extract PDF: {e}",
                file_path=file_path,
            ) from e

        return result

    def _extract_page(self, page: Any, page_num: int) -> PDFPage:
        """Extract content from a single PDF page.

        Args:
            page: pdfplumber page object
            page_num: Page number

        Returns:
            PDFPage with extracted content
        """
        text = page.extract_text() or ""

        # Estimate if page is scanned (heuristic: very little extractable text
        # relative to page size, or high special character ratio)
        is_scanned = False
        ocr_confidence = None

        if len(text.strip()) < 50 and page.width * page.height > 100000:
            is_scanned = True
            ocr_confidence = 0.0  # No OCR performed

        # Calculate quality score
        quality_score = self._score_page_quality(text, is_scanned)

        return PDFPage(
            number=page_num,
            text=text,
            quality_score=quality_score,
            is_scanned=is_scanned,
            ocr_confidence=ocr_confidence,
        )

    def _score_page_quality(self, text: str, is_scanned: bool) -> float:
        """Score the extraction quality of a page.

        Args:
            text: Extracted text
            is_scanned: Whether page appears to be scanned

        Returns:
            Quality score (0.0-1.0)
        """
        score = 1.0

        # Penalize scanned pages without OCR
        if is_scanned:
            score *= 0.3

        # Penalize very short pages
        if len(text.strip()) < self.config.min_text_length:
            score *= 0.5

        # Check for gibberish (high ratio of special characters)
        if text:
            special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
            special_ratio = special_chars / len(text)
            if special_ratio > self.config.max_gibberish_ratio:
                score *= (1.0 - special_ratio)

        return max(0.0, min(1.0, score))

    def _extract_tables(self, page: Any, page_num: int) -> list[PDFTable]:
        """Extract tables from a PDF page.

        Args:
            page: pdfplumber page object
            page_num: Page number

        Returns:
            List of extracted tables
        """
        tables: list[PDFTable] = []

        try:
            extracted_tables = page.extract_tables() or []
            for table_data in extracted_tables:
                if not table_data:
                    continue

                # Clean up table data
                cleaned_data = [
                    [str(cell) if cell else "" for cell in row]
                    for row in table_data
                ]

                # Try to detect headers (first row)
                headers = cleaned_data[0] if cleaned_data else None

                tables.append(
                    PDFTable(
                        page=page_num,
                        data=cleaned_data,
                        quality_score=self._score_table_quality(cleaned_data),
                        headers=headers,
                    )
                )
        except Exception as e:
            logger.warning(f"Failed to extract tables from page {page_num}: {e}")

        return tables

    def _score_table_quality(self, data: list[list[str]]) -> float:
        """Score the extraction quality of a table.

        Args:
            data: Table data

        Returns:
            Quality score (0.0-1.0)
        """
        if not data:
            return 0.0

        # Check for empty cells
        total_cells = sum(len(row) for row in data)
        empty_cells = sum(1 for row in data for cell in row if not cell.strip())

        if total_cells == 0:
            return 0.0

        empty_ratio = empty_cells / total_cells
        return max(0.0, 1.0 - empty_ratio)

    def _validate_extraction(self, result: PDFExtractionResult) -> list[QualityIssue]:
        """Validate extraction results and generate issues.

        Args:
            result: Extraction result to validate

        Returns:
            List of quality issues
        """
        issues: list[QualityIssue] = []

        # Check for scanned pages without OCR
        scanned_pages = [p for p in result.pages if p.is_scanned]
        if scanned_pages:
            issues.append(
                QualityIssue(
                    code="SCANNED_PAGES",
                    severity="warning",
                    message=f"{len(scanned_pages)} page(s) appear to be scanned without OCR",
                    location=f"Pages: {', '.join(str(p.number) for p in scanned_pages)}",
                )
            )

        # Check for low quality pages
        low_quality = [p for p in result.pages if p.quality_score < 0.5]
        if low_quality:
            issues.append(
                QualityIssue(
                    code="LOW_QUALITY_PAGES",
                    severity="warning",
                    message=f"{len(low_quality)} page(s) have low extraction quality",
                    location=f"Pages: {', '.join(str(p.number) for p in low_quality)}",
                )
            )

        # Check overall text length
        full_text = result.full_text
        if len(full_text.strip()) < self.config.min_text_length:
            issues.append(
                QualityIssue(
                    code="INSUFFICIENT_CONTENT",
                    severity="error",
                    message=f"Extracted text too short ({len(full_text)} chars)",
                )
            )

        return issues

    def _create_chunks(
        self,
        result: PDFExtractionResult,
        source_path: str | None,
    ) -> list[Chunk]:
        """Create chunks from extraction result.

        Args:
            result: Extraction result
            source_path: Path to source PDF

        Returns:
            List of chunks
        """
        chunks: list[Chunk] = []

        for page in result.pages:
            if not page.text.strip():
                continue

            # Simple chunking by paragraphs
            paragraphs = page.text.split("\n\n")
            current_chunk = ""

            for para in paragraphs:
                if len(current_chunk) + len(para) > self.chunk_size:
                    if current_chunk:
                        chunks.append(
                            Chunk(
                                text=current_chunk.strip(),
                                source=source_path or "pdf",
                                quality_score=page.quality_score,
                                metadata={
                                    "page": page.number,
                                    "source_type": "pdf",
                                    "is_scanned": page.is_scanned,
                                },
                            )
                        )
                    current_chunk = para
                else:
                    current_chunk = f"{current_chunk}\n\n{para}" if current_chunk else para

            if current_chunk:
                chunks.append(
                    Chunk(
                        text=current_chunk.strip(),
                        source=source_path or "pdf",
                        quality_score=page.quality_score,
                        metadata={
                            "page": page.number,
                            "source_type": "pdf",
                            "is_scanned": page.is_scanned,
                        },
                    )
                )

        return chunks

    def extract_sync(
        self,
        source: Path | str | bytes,
        **kwargs: Any,
    ) -> PDFExtractionResult:
        """Synchronous wrapper for extract().

        Args:
            source: Path to PDF or PDF bytes
            **kwargs: Additional arguments

        Returns:
            PDFExtractionResult
        """
        return asyncio.run(self.extract(source, **kwargs))
