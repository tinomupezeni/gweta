"""System Intent definition and loading.

The SystemIntent class represents what a RAG system is meant to do.
It defines the purpose, target users, and relevant topics that
guide content curation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from gweta.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class QualityRequirements:
    """Quality thresholds for content curation."""

    min_relevance_score: float = 0.6
    review_threshold: float = 0.4
    freshness_cutoff: str | None = None  # ISO date string
    prefer_official_sources: bool = True


@dataclass
class GeographicFocus:
    """Geographic targeting for content."""

    primary: str | None = None
    secondary: list[str] = field(default_factory=list)


@dataclass
class SystemIntent:
    """Defines what a RAG system is meant to do.

    The intent guides content curation by defining:
    - What the system does (description)
    - Who uses it (target_users)
    - What questions it should answer (core_questions)
    - What topics are relevant/irrelevant

    Example:
        >>> intent = SystemIntent.from_yaml("simuka_intent.yaml")
        >>> print(intent.name)
        "Simuka Career Platform"
        >>> print(intent.core_questions[:2])
        ["How do I register a business?", "What services can I offer?"]
    """

    name: str
    description: str
    target_users: list[str] = field(default_factory=list)
    core_questions: list[str] = field(default_factory=list)
    relevant_topics: list[str] = field(default_factory=list)
    irrelevant_topics: list[str] = field(default_factory=list)
    geographic_focus: GeographicFocus = field(default_factory=GeographicFocus)
    quality_requirements: QualityRequirements = field(
        default_factory=QualityRequirements
    )
    metadata: dict[str, Any] = field(default_factory=dict)

    # Computed after loading
    _intent_text: str = field(default="", repr=False)

    def __post_init__(self) -> None:
        """Generate intent text for embedding."""
        self._intent_text = self._build_intent_text()

    def _build_intent_text(self) -> str:
        """Build the text representation of intent for embedding.

        This combines the key elements of the intent into a single
        text that captures what content should be relevant.
        """
        parts = []

        # System description
        if self.description:
            parts.append(f"System purpose: {self.description.strip()}")

        # Target users
        if self.target_users:
            users = ", ".join(self.target_users)
            parts.append(f"Target users: {users}")

        # Core questions (most important for relevance)
        if self.core_questions:
            questions = "\n- ".join(self.core_questions)
            parts.append(f"Questions this system should answer:\n- {questions}")

        # Relevant topics
        if self.relevant_topics:
            topics = ", ".join(self.relevant_topics)
            parts.append(f"Relevant topics: {topics}")

        # Geographic focus
        if self.geographic_focus.primary:
            geo = self.geographic_focus.primary
            if self.geographic_focus.secondary:
                geo += f" (also: {', '.join(self.geographic_focus.secondary)})"
            parts.append(f"Geographic focus: {geo}")

        return "\n\n".join(parts)

    @property
    def intent_text(self) -> str:
        """Get the text representation for embedding."""
        return self._intent_text

    @property
    def min_relevance_score(self) -> float:
        """Minimum score to accept content."""
        return self.quality_requirements.min_relevance_score

    @property
    def review_threshold(self) -> float:
        """Score below which content is rejected outright."""
        return self.quality_requirements.review_threshold

    @classmethod
    def from_yaml(cls, path: str | Path) -> SystemIntent:
        """Load intent from YAML file.

        Args:
            path: Path to YAML file

        Returns:
            SystemIntent instance

        Example YAML:
            system:
              name: "My RAG System"
              description: "Helps users with X"
              core_questions:
                - "How do I do X?"
                - "What is Y?"
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Intent file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SystemIntent:
        """Create intent from dictionary.

        Args:
            data: Dictionary with intent configuration

        Returns:
            SystemIntent instance
        """
        # Handle nested 'system' key
        if "system" in data:
            data = data["system"]

        # Extract geographic focus
        geo_data = data.get("geographic_focus", {})
        if isinstance(geo_data, str):
            geographic_focus = GeographicFocus(primary=geo_data)
        elif isinstance(geo_data, dict):
            geographic_focus = GeographicFocus(
                primary=geo_data.get("primary"),
                secondary=geo_data.get("secondary", []),
            )
        else:
            geographic_focus = GeographicFocus()

        # Extract quality requirements
        quality_data = data.get("quality_requirements", {})
        quality_requirements = QualityRequirements(
            min_relevance_score=quality_data.get("min_relevance_score", 0.6),
            review_threshold=quality_data.get("review_threshold", 0.4),
            freshness_cutoff=quality_data.get("freshness_cutoff"),
            prefer_official_sources=quality_data.get("prefer_official_sources", True),
        )

        return cls(
            name=data.get("name", "Unnamed System"),
            description=data.get("description", ""),
            target_users=data.get("target_users", []),
            core_questions=data.get("core_questions", []),
            relevant_topics=data.get("relevant_topics", []),
            irrelevant_topics=data.get("irrelevant_topics", []),
            geographic_focus=geographic_focus,
            quality_requirements=quality_requirements,
            metadata=data.get("metadata", {}),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "system": {
                "name": self.name,
                "description": self.description,
                "target_users": self.target_users,
                "core_questions": self.core_questions,
                "relevant_topics": self.relevant_topics,
                "irrelevant_topics": self.irrelevant_topics,
                "geographic_focus": {
                    "primary": self.geographic_focus.primary,
                    "secondary": self.geographic_focus.secondary,
                },
                "quality_requirements": {
                    "min_relevance_score": self.quality_requirements.min_relevance_score,
                    "review_threshold": self.quality_requirements.review_threshold,
                    "freshness_cutoff": self.quality_requirements.freshness_cutoff,
                    "prefer_official_sources": self.quality_requirements.prefer_official_sources,
                },
                "metadata": self.metadata,
            }
        }

    def to_yaml(self) -> str:
        """Convert to YAML string."""
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)

    def is_irrelevant_topic(self, text: str) -> bool:
        """Quick check if text contains explicitly irrelevant topics.

        This is a fast pre-filter before embedding-based scoring.
        """
        text_lower = text.lower()
        for topic in self.irrelevant_topics:
            if topic.lower() in text_lower:
                return True
        return False

    def __repr__(self) -> str:
        return (
            f"SystemIntent(name={self.name!r}, "
            f"questions={len(self.core_questions)}, "
            f"topics={len(self.relevant_topics)})"
        )
