"""Core data types for Gweta.

This module defines the fundamental data structures used throughout Gweta:
- Chunk: Universal chunk representation with quality metadata
- Source: Data source with authority tracking
- QualityReport: Batch validation results
- QualityDetails: Multi-dimensional quality breakdown
- QualityIssue: Single quality problem
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal
from uuid import uuid4


@dataclass
class QualityIssue:
    """Single quality problem detected during validation.

    Attributes:
        code: Machine-readable issue code (e.g., "LOW_DENSITY", "DUPLICATE")
        severity: Issue severity level
        message: Human-readable description of the issue
        location: Optional location within the chunk where issue was found
    """
    code: str
    severity: Literal["error", "warning", "info"]
    message: str
    location: str | None = None

    def __str__(self) -> str:
        loc = f" at {self.location}" if self.location else ""
        return f"[{self.severity.upper()}] {self.code}: {self.message}{loc}"


@dataclass
class QualityDetails:
    """Multi-dimensional quality breakdown for a chunk.

    Provides granular quality metrics across different dimensions,
    allowing users to understand WHERE quality issues exist.

    Attributes:
        extraction_score: Layer 1 score - quality of text extraction (0.0-1.0)
        coherence_score: Layer 2 score - how well chunk stands alone (0.0-1.0)
        density_score: Information density - signal vs noise ratio (0.0-1.0)
        duplicate_score: Uniqueness score - 1.0 = unique, 0.0 = exact duplicate
        issues: List of specific quality issues found
    """
    extraction_score: float = 1.0
    coherence_score: float = 1.0
    density_score: float = 1.0
    duplicate_score: float = 1.0
    issues: list[QualityIssue] = field(default_factory=list)

    @property
    def aggregate_score(self) -> float:
        """Calculate weighted aggregate quality score."""
        weights = {
            "extraction": 0.25,
            "coherence": 0.25,
            "density": 0.25,
            "duplicate": 0.25,
        }
        return (
            self.extraction_score * weights["extraction"]
            + self.coherence_score * weights["coherence"]
            + self.density_score * weights["density"]
            + self.duplicate_score * weights["duplicate"]
        )

    def has_errors(self) -> bool:
        """Check if there are any error-level issues."""
        return any(issue.severity == "error" for issue in self.issues)

    def has_warnings(self) -> bool:
        """Check if there are any warning-level issues."""
        return any(issue.severity == "warning" for issue in self.issues)


@dataclass
class Chunk:
    """Universal chunk representation with quality metadata.

    This is Gweta's core data structure. All adapters convert to/from
    this format, ensuring consistent handling across different frameworks.

    Attributes:
        id: Unique identifier for the chunk
        text: The actual chunk content
        metadata: Arbitrary metadata (source, page, etc.)
        source: Source identifier (URL, file path, etc.)
        quality_score: Overall quality score (0.0-1.0), None if not validated
        quality_details: Detailed quality breakdown, None if not validated
        created_at: When the chunk was created
    """
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    source: str = ""
    id: str = field(default_factory=lambda: str(uuid4()))
    quality_score: float | None = None
    quality_details: QualityDetails | None = None
    created_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self) -> None:
        """Validate chunk after initialization."""
        if not isinstance(self.metadata, dict):
            self.metadata = {}

    @property
    def is_validated(self) -> bool:
        """Check if this chunk has been validated."""
        return self.quality_score is not None

    @property
    def is_acceptable(self) -> bool:
        """Check if chunk passes minimum quality threshold.

        Returns True if not validated (optimistic default).
        """
        if self.quality_details is None:
            return True
        return not self.quality_details.has_errors()

    def to_dict(self) -> dict[str, Any]:
        """Convert chunk to dictionary for serialization."""
        return {
            "id": self.id,
            "text": self.text,
            "metadata": self.metadata,
            "source": self.source,
            "quality_score": self.quality_score,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Chunk":
        """Create a Chunk from a dictionary."""
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        elif created_at is None:
            created_at = datetime.now()

        return cls(
            id=data.get("id", str(uuid4())),
            text=data.get("text", ""),
            metadata=data.get("metadata", {}),
            source=data.get("source", ""),
            quality_score=data.get("quality_score"),
            created_at=created_at,
        )


@dataclass
class Source:
    """Data source with authority tracking.

    Represents a data source (website, API, database) with metadata
    about its trustworthiness and freshness requirements.

    Attributes:
        id: Unique identifier for the source
        name: Human-readable name
        url: URL or connection string (optional)
        authority_tier: Authority level 1-5 (5 = primary legislation, 1 = blog)
        freshness_days: Maximum age in days before content is considered stale
        last_crawled: When the source was last crawled/fetched
    """
    id: str
    name: str
    url: str | None = None
    authority_tier: int = 3
    freshness_days: int = 90
    last_crawled: datetime | None = None

    def __post_init__(self) -> None:
        """Validate authority tier."""
        if not 1 <= self.authority_tier <= 5:
            raise ValueError(f"authority_tier must be 1-5, got {self.authority_tier}")

    @property
    def is_stale(self) -> bool:
        """Check if source content is stale based on freshness_days."""
        if self.last_crawled is None:
            return True
        age = (datetime.now() - self.last_crawled).days
        return age > self.freshness_days

    def to_dict(self) -> dict[str, Any]:
        """Convert source to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "url": self.url,
            "authority_tier": self.authority_tier,
            "freshness_days": self.freshness_days,
            "last_crawled": self.last_crawled.isoformat() if self.last_crawled else None,
        }


@dataclass
class ChunkResult:
    """Single chunk validation result.

    Attributes:
        chunk: The validated chunk (with quality_score populated)
        passed: Whether the chunk passed validation
        quality_score: Overall quality score (0.0-1.0)
        issues: List of quality issues found
    """
    chunk: Chunk
    passed: bool
    quality_score: float
    issues: list[QualityIssue] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        """Check if there are error-level issues."""
        return any(i.severity == "error" for i in self.issues)

    @property
    def has_warnings(self) -> bool:
        """Check if there are warning-level issues."""
        return any(i.severity == "warning" for i in self.issues)


@dataclass
class QualityReport:
    """Validation results for a batch of chunks.

    Provides aggregate statistics and per-chunk results for
    batch validation operations.

    Attributes:
        total_chunks: Total number of chunks validated
        passed: Number of chunks that passed validation
        failed: Number of chunks that failed validation
        warnings: Number of chunks with warnings (but passed)
        avg_quality_score: Average quality score across all chunks
        issues_by_type: Count of issues grouped by issue code
        chunks: Per-chunk validation results
    """
    total_chunks: int
    passed: int
    failed: int
    warnings: int
    avg_quality_score: float
    issues_by_type: dict[str, int] = field(default_factory=dict)
    chunks: list[ChunkResult] = field(default_factory=list)

    def accepted(self) -> list[Chunk]:
        """Get chunks that passed validation."""
        return [r.chunk for r in self.chunks if r.passed]

    def rejected(self) -> list[Chunk]:
        """Get chunks that failed validation."""
        return [r.chunk for r in self.chunks if not r.passed]

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Validated {self.total_chunks} chunks:",
            f"  - Passed: {self.passed}",
            f"  - Failed: {self.failed}",
            f"  - Warnings: {self.warnings}",
            f"  - Avg Quality: {self.avg_quality_score:.2f}",
        ]
        if self.issues_by_type:
            lines.append("  - Issues by type:")
            for issue_type, count in sorted(self.issues_by_type.items()):
                lines.append(f"      {issue_type}: {count}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary for serialization."""
        return {
            "total_chunks": self.total_chunks,
            "passed": self.passed,
            "failed": self.failed,
            "warnings": self.warnings,
            "avg_quality_score": self.avg_quality_score,
            "issues_by_type": self.issues_by_type,
            "chunks": [
                {
                    "chunk_id": r.chunk.id,
                    "passed": r.passed,
                    "quality_score": r.quality_score,
                    "issues": [str(i) for i in r.issues],
                }
                for r in self.chunks
            ],
        }
