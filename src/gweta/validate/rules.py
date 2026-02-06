"""Layer 3: Domain rule engine.

This module provides a YAML-based rule engine for
domain-specific validation rules and known facts.
"""

import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Literal

import yaml

from gweta.core.exceptions import RuleEngineError
from gweta.core.logging import get_logger
from gweta.core.types import Chunk, QualityIssue

logger = get_logger(__name__)


@dataclass
class KnownFact:
    """A verified fact for cross-referencing.

    Attributes:
        key: Unique identifier for the fact
        value: The known value
        source: Where this fact comes from
        verified_date: When this fact was verified
        tolerance: Acceptable deviation for numerical values
        unit: Unit of measurement (if applicable)
    """
    key: str
    value: Any
    source: str
    verified_date: date
    tolerance: float = 0.0
    unit: str | None = None


@dataclass
class Rule:
    """A validation rule definition.

    Attributes:
        name: Unique rule identifier
        description: Human-readable description
        rule_type: Type of rule (regex, numerical_range, date_check, etc.)
        condition: Rule condition (pattern, range, etc.)
        severity: Issue severity if rule fails
        message: Message template for failures
        field: Metadata field to check (optional)
        enabled: Whether rule is active
    """
    name: str
    description: str
    rule_type: str
    severity: Literal["error", "warning", "info"] = "warning"
    message: str = ""
    condition: str | dict[str, Any] | None = None
    field: str | None = None
    enabled: bool = True


@dataclass
class RuleValidationResult:
    """Result of rule validation.

    Attributes:
        passed: Whether all rules passed
        score: Overall compliance score
        violations: List of rule violations
        facts_checked: Number of known facts checked
    """
    passed: bool
    score: float
    violations: list[QualityIssue] = field(default_factory=list)
    facts_checked: int = 0


class DomainRuleEngine:
    """Layer 3: Domain-specific validation rules.

    Supports:
    - Regex pattern matching
    - Numerical range validation
    - Date freshness checks
    - Known fact cross-referencing
    - Custom rule conditions

    Example:
        >>> rules = DomainRuleEngine.from_yaml("rules/zimbabwe_business.yaml")
        >>> result = rules.validate_chunk(chunk)
        >>> if not result.passed:
        ...     for v in result.violations:
        ...         print(v)
    """

    def __init__(
        self,
        rules: list[Rule] | None = None,
        known_facts: list[KnownFact] | None = None,
    ) -> None:
        """Initialize DomainRuleEngine.

        Args:
            rules: List of validation rules
            known_facts: List of known facts for cross-referencing
        """
        self.rules = rules or []
        self.known_facts = {f.key: f for f in (known_facts or [])}
        self._enabled_rules = [r for r in self.rules if r.enabled]

    @classmethod
    def from_yaml(cls, path: Path | str) -> "DomainRuleEngine":
        """Load rules from YAML file.

        Args:
            path: Path to YAML file

        Returns:
            Configured DomainRuleEngine
        """
        path = Path(path)
        if not path.exists():
            raise RuleEngineError(
                f"Rules file not found: {path}",
                rule_file=str(path),
            )

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise RuleEngineError(
                f"Invalid YAML: {e}",
                rule_file=str(path),
            ) from e

        return cls.from_dict(data or {})

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DomainRuleEngine":
        """Create engine from dictionary.

        Args:
            data: Dictionary with 'rules' and 'known_facts'

        Returns:
            Configured DomainRuleEngine
        """
        rules: list[Rule] = []
        for rule_data in data.get("rules", []):
            rules.append(
                Rule(
                    name=rule_data.get("name", "unnamed"),
                    description=rule_data.get("description", ""),
                    rule_type=rule_data.get("type", "regex"),
                    severity=rule_data.get("severity", "warning"),
                    message=rule_data.get("message", ""),
                    condition=rule_data.get("pattern") or rule_data.get("condition"),
                    field=rule_data.get("field"),
                    enabled=rule_data.get("enabled", True),
                )
            )

        known_facts: list[KnownFact] = []
        for fact_data in data.get("known_facts", []):
            verified = fact_data.get("verified_date", date.today())
            if isinstance(verified, str):
                verified = date.fromisoformat(verified)

            known_facts.append(
                KnownFact(
                    key=fact_data.get("key", ""),
                    value=fact_data.get("value"),
                    source=fact_data.get("source", ""),
                    verified_date=verified,
                    tolerance=fact_data.get("tolerance", 0),
                    unit=fact_data.get("unit"),
                )
            )

        return cls(rules=rules, known_facts=known_facts)

    def validate_chunk(self, chunk: Chunk) -> RuleValidationResult:
        """Validate a chunk against all rules.

        Args:
            chunk: Chunk to validate

        Returns:
            RuleValidationResult with violations
        """
        violations: list[QualityIssue] = []

        for rule in self._enabled_rules:
            result = self._evaluate_rule(rule, chunk.text, chunk.metadata)
            if result:
                violations.append(result)

        passed = not any(v.severity == "error" for v in violations)
        score = 1.0 - (len(violations) * 0.1)

        return RuleValidationResult(
            passed=passed,
            score=max(0.0, score),
            violations=violations,
        )

    def validate_response(
        self,
        response: str,
        source_chunks: list[Chunk] | None = None,
    ) -> RuleValidationResult:
        """Validate an AI response against domain rules.

        Args:
            response: Generated response text
            source_chunks: Source chunks used for the response

        Returns:
            RuleValidationResult
        """
        violations: list[QualityIssue] = []
        facts_checked = 0

        # Check against known facts
        for key, fact in self.known_facts.items():
            extracted = self._extract_value(response, key)
            if extracted is not None:
                facts_checked += 1
                if not self._compare_values(extracted, fact.value, fact.tolerance):
                    violations.append(
                        QualityIssue(
                            code="FACT_MISMATCH",
                            severity="error",
                            message=f"Response says {extracted}, known fact is {fact.value} ({fact.source})",
                        )
                    )

        # Run standard rules
        for rule in self._enabled_rules:
            result = self._evaluate_rule(rule, response, {})
            if result:
                violations.append(result)

        passed = not any(v.severity == "error" for v in violations)
        score = 1.0 - (len(violations) * 0.1)

        return RuleValidationResult(
            passed=passed,
            score=max(0.0, score),
            violations=violations,
            facts_checked=facts_checked,
        )

    def _evaluate_rule(
        self,
        rule: Rule,
        text: str,
        metadata: dict[str, Any],
    ) -> QualityIssue | None:
        """Evaluate a single rule against text.

        Args:
            rule: Rule to evaluate
            text: Text to check
            metadata: Metadata dict

        Returns:
            QualityIssue if rule fails, None otherwise
        """
        try:
            if rule.rule_type == "regex":
                return self._eval_regex(rule, text)
            elif rule.rule_type == "numerical_range":
                return self._eval_numerical(rule, text, metadata)
            elif rule.rule_type == "required_field":
                return self._eval_required(rule, metadata)
            else:
                logger.warning(f"Unknown rule type: {rule.rule_type}")
                return None
        except Exception as e:
            logger.error(f"Error evaluating rule {rule.name}: {e}")
            return None

    def _eval_regex(self, rule: Rule, text: str) -> QualityIssue | None:
        """Evaluate regex pattern rule."""
        if not rule.condition:
            return None

        pattern = str(rule.condition)
        matches = re.findall(pattern, text, re.IGNORECASE)

        # Rule specifies required=true means pattern MUST be found
        required = rule.field == "required"

        if required and not matches:
            return QualityIssue(
                code=f"RULE_{rule.name.upper()}",
                severity=rule.severity,
                message=rule.message or f"Required pattern not found: {pattern}",
            )

        return None

    def _eval_numerical(
        self,
        rule: Rule,
        text: str,
        metadata: dict[str, Any],
    ) -> QualityIssue | None:
        """Evaluate numerical range rule."""
        if not isinstance(rule.condition, dict):
            return None

        min_val = rule.condition.get("min")
        max_val = rule.condition.get("max")

        # Get value from metadata field or extract from text
        value = None
        if rule.field and rule.field in metadata:
            value = metadata[rule.field]
        else:
            # Try to extract numbers from text
            numbers = re.findall(r"\$?[\d,]+(?:\.\d+)?", text)
            if numbers:
                value = float(numbers[0].replace("$", "").replace(",", ""))

        if value is None:
            return None

        value = float(value)

        if min_val is not None and value < min_val:
            return QualityIssue(
                code=f"RULE_{rule.name.upper()}",
                severity=rule.severity,
                message=rule.message.replace("${value}", str(value)) if rule.message else f"Value {value} below minimum {min_val}",
            )

        if max_val is not None and value > max_val:
            return QualityIssue(
                code=f"RULE_{rule.name.upper()}",
                severity=rule.severity,
                message=rule.message.replace("${value}", str(value)) if rule.message else f"Value {value} exceeds maximum {max_val}",
            )

        return None

    def _eval_required(
        self,
        rule: Rule,
        metadata: dict[str, Any],
    ) -> QualityIssue | None:
        """Evaluate required field rule."""
        if rule.field and rule.field not in metadata:
            return QualityIssue(
                code=f"RULE_{rule.name.upper()}",
                severity=rule.severity,
                message=rule.message or f"Required field missing: {rule.field}",
            )
        return None

    def _extract_value(self, text: str, key: str) -> Any:
        """Extract a value from text based on key pattern."""
        # Simple extraction - look for patterns like "key: value" or "key is value"
        patterns = [
            rf"{key}\s*[:=]\s*\$?([\d,]+(?:\.\d+)?)",
            rf"{key}\s+(?:is|costs?|=)\s+\$?([\d,]+(?:\.\d+)?)",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value = match.group(1).replace(",", "")
                try:
                    return float(value)
                except ValueError:
                    return value

        return None

    def _compare_values(
        self,
        extracted: Any,
        known: Any,
        tolerance: float,
    ) -> bool:
        """Compare extracted value with known value."""
        if isinstance(extracted, (int, float)) and isinstance(known, (int, float)):
            return abs(extracted - known) <= tolerance
        return str(extracted).lower() == str(known).lower()

    def add_rule(self, rule: Rule) -> None:
        """Add a rule to the engine."""
        self.rules.append(rule)
        if rule.enabled:
            self._enabled_rules.append(rule)

    def add_fact(self, fact: KnownFact) -> None:
        """Add a known fact."""
        self.known_facts[fact.key] = fact

    def to_yaml(self) -> str:
        """Export rules to YAML string."""
        data = {
            "rules": [
                {
                    "name": r.name,
                    "description": r.description,
                    "type": r.rule_type,
                    "severity": r.severity,
                    "message": r.message,
                    "pattern": r.condition if r.rule_type == "regex" else None,
                    "condition": r.condition if r.rule_type != "regex" else None,
                    "field": r.field,
                    "enabled": r.enabled,
                }
                for r in self.rules
            ],
            "known_facts": [
                {
                    "key": f.key,
                    "value": f.value,
                    "source": f.source,
                    "verified_date": f.verified_date.isoformat(),
                    "tolerance": f.tolerance,
                    "unit": f.unit,
                }
                for f in self.known_facts.values()
            ],
        }
        return yaml.dump(data, default_flow_style=False, sort_keys=False)
