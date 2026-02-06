"""Tests for Domain Rule Engine.

These tests verify the DomainRuleEngine functionality including:
- YAML rule loading
- Regex pattern validation
- Numerical range validation
- Required field validation
- Known fact cross-referencing
"""

import pytest
from datetime import date
from pathlib import Path
from unittest.mock import patch, mock_open

from gweta.core.types import Chunk
from gweta.validate.rules import (
    DomainRuleEngine,
    Rule,
    KnownFact,
    RuleValidationResult,
)


class TestRuleDataclasses:
    """Tests for Rule and KnownFact dataclasses."""

    def test_rule_creation(self):
        """Test Rule dataclass creation."""
        rule = Rule(
            name="test_rule",
            description="A test rule",
            rule_type="regex",
            condition=r"\d+",
            severity="warning",
            message="Must contain numbers",
        )

        assert rule.name == "test_rule"
        assert rule.rule_type == "regex"
        assert rule.severity == "warning"
        assert rule.enabled is True

    def test_rule_defaults(self):
        """Test Rule default values."""
        rule = Rule(
            name="minimal",
            description="Minimal rule",
            rule_type="regex",
        )

        assert rule.severity == "warning"
        assert rule.message == ""
        assert rule.condition is None
        assert rule.field is None
        assert rule.enabled is True

    def test_known_fact_creation(self):
        """Test KnownFact dataclass creation."""
        fact = KnownFact(
            key="min_wage",
            value=350,
            source="Labour Act",
            verified_date=date(2024, 1, 1),
            tolerance=10,
            unit="USD",
        )

        assert fact.key == "min_wage"
        assert fact.value == 350
        assert fact.tolerance == 10
        assert fact.unit == "USD"

    def test_known_fact_defaults(self):
        """Test KnownFact default values."""
        fact = KnownFact(
            key="test",
            value=100,
            source="test",
            verified_date=date.today(),
        )

        assert fact.tolerance == 0.0
        assert fact.unit is None


class TestDomainRuleEngineInit:
    """Tests for DomainRuleEngine initialization."""

    def test_empty_init(self):
        """Test initialization with no rules."""
        engine = DomainRuleEngine()

        assert engine.rules == []
        assert engine.known_facts == {}

    def test_init_with_rules(self):
        """Test initialization with rules."""
        rules = [
            Rule(name="r1", description="Rule 1", rule_type="regex"),
            Rule(name="r2", description="Rule 2", rule_type="regex", enabled=False),
        ]

        engine = DomainRuleEngine(rules=rules)

        assert len(engine.rules) == 2
        assert len(engine._enabled_rules) == 1

    def test_init_with_facts(self):
        """Test initialization with known facts."""
        facts = [
            KnownFact(
                key="fact1",
                value=100,
                source="test",
                verified_date=date.today(),
            ),
            KnownFact(
                key="fact2",
                value="value",
                source="test",
                verified_date=date.today(),
            ),
        ]

        engine = DomainRuleEngine(known_facts=facts)

        assert len(engine.known_facts) == 2
        assert "fact1" in engine.known_facts
        assert "fact2" in engine.known_facts


class TestDomainRuleEngineFromDict:
    """Tests for loading rules from dictionary."""

    def test_from_dict_basic(self):
        """Test loading from basic dictionary."""
        data = {
            "rules": [
                {
                    "name": "test_rule",
                    "description": "Test description",
                    "type": "regex",
                    "pattern": r"\d+",
                    "severity": "error",
                    "message": "Must have numbers",
                }
            ],
            "known_facts": [
                {
                    "key": "test_fact",
                    "value": 42,
                    "source": "test source",
                    "verified_date": "2024-01-01",
                }
            ],
        }

        engine = DomainRuleEngine.from_dict(data)

        assert len(engine.rules) == 1
        assert engine.rules[0].name == "test_rule"
        assert engine.rules[0].condition == r"\d+"

        assert len(engine.known_facts) == 1
        assert engine.known_facts["test_fact"].value == 42

    def test_from_dict_empty(self):
        """Test loading from empty dictionary."""
        engine = DomainRuleEngine.from_dict({})

        assert engine.rules == []
        assert engine.known_facts == {}

    def test_from_dict_with_condition(self):
        """Test loading numerical range rule."""
        data = {
            "rules": [
                {
                    "name": "range_rule",
                    "description": "Range check",
                    "type": "numerical_range",
                    "condition": {"min": 0, "max": 100},
                }
            ]
        }

        engine = DomainRuleEngine.from_dict(data)

        assert engine.rules[0].condition == {"min": 0, "max": 100}


class TestDomainRuleEngineYAML:
    """Tests for YAML loading."""

    def test_from_yaml_file_not_found(self):
        """Test error when YAML file doesn't exist."""
        from gweta.core.exceptions import RuleEngineError

        with pytest.raises(RuleEngineError) as exc_info:
            DomainRuleEngine.from_yaml("/nonexistent/file.yaml")

        assert "not found" in str(exc_info.value)

    def test_from_yaml_valid(self, tmp_path):
        """Test loading valid YAML file."""
        yaml_content = """
rules:
  - name: test_rule
    description: Test
    type: regex
    pattern: "test"
known_facts:
  - key: fact1
    value: 100
    source: test
    verified_date: "2024-01-01"
"""
        yaml_file = tmp_path / "rules.yaml"
        yaml_file.write_text(yaml_content)

        engine = DomainRuleEngine.from_yaml(yaml_file)

        assert len(engine.rules) == 1
        assert len(engine.known_facts) == 1


class TestRegexRuleValidation:
    """Tests for regex pattern validation."""

    def test_regex_required_pattern_found(self):
        """Test required pattern is found."""
        rule = Rule(
            name="phone",
            description="Phone validation",
            rule_type="regex",
            condition=r"\d{3}-\d{4}",
            field="required",
        )

        engine = DomainRuleEngine(rules=[rule])
        chunk = Chunk(text="Call 123-4567 for info", source="test")

        result = engine.validate_chunk(chunk)

        assert result.passed
        assert len(result.violations) == 0

    def test_regex_required_pattern_missing(self):
        """Test required pattern is missing."""
        rule = Rule(
            name="phone",
            description="Phone validation",
            rule_type="regex",
            condition=r"\d{3}-\d{4}",
            field="required",
            severity="error",
            message="Phone number required",
        )

        engine = DomainRuleEngine(rules=[rule])
        chunk = Chunk(text="No phone number here", source="test")

        result = engine.validate_chunk(chunk)

        assert not result.passed
        assert len(result.violations) == 1
        assert result.violations[0].code == "RULE_PHONE"


class TestNumericalRangeValidation:
    """Tests for numerical range validation."""

    def test_value_in_range(self):
        """Test value within range."""
        rule = Rule(
            name="price",
            description="Price validation",
            rule_type="numerical_range",
            condition={"min": 0, "max": 1000},
        )

        engine = DomainRuleEngine(rules=[rule])
        chunk = Chunk(text="Price is $500", source="test")

        result = engine.validate_chunk(chunk)

        assert result.passed

    def test_value_below_min(self):
        """Test value below minimum."""
        rule = Rule(
            name="price",
            description="Price validation",
            rule_type="numerical_range",
            condition={"min": 100, "max": 1000},
            severity="error",
        )

        engine = DomainRuleEngine(rules=[rule])
        chunk = Chunk(text="Price is $50", source="test")

        result = engine.validate_chunk(chunk)

        assert not result.passed
        assert any("below" in v.message.lower() for v in result.violations)

    def test_value_above_max(self):
        """Test value above maximum."""
        rule = Rule(
            name="price",
            description="Price validation",
            rule_type="numerical_range",
            condition={"min": 0, "max": 100},
            severity="error",
        )

        engine = DomainRuleEngine(rules=[rule])
        chunk = Chunk(text="Total: 500", source="test")

        result = engine.validate_chunk(chunk)

        assert not result.passed

    def test_value_from_metadata(self):
        """Test extracting value from metadata."""
        rule = Rule(
            name="quantity",
            description="Quantity check",
            rule_type="numerical_range",
            field="quantity",
            condition={"min": 1, "max": 100},
            severity="warning",
        )

        engine = DomainRuleEngine(rules=[rule])
        chunk = Chunk(
            text="Order placed",
            source="test",
            metadata={"quantity": 150},
        )

        result = engine.validate_chunk(chunk)

        assert len(result.violations) == 1


class TestRequiredFieldValidation:
    """Tests for required field validation."""

    def test_required_field_present(self):
        """Test required field is present."""
        rule = Rule(
            name="author_required",
            description="Author must be present",
            rule_type="required_field",
            field="author",
        )

        engine = DomainRuleEngine(rules=[rule])
        chunk = Chunk(
            text="Content",
            source="test",
            metadata={"author": "John"},
        )

        result = engine.validate_chunk(chunk)

        assert result.passed

    def test_required_field_missing(self):
        """Test required field is missing."""
        rule = Rule(
            name="author_required",
            description="Author must be present",
            rule_type="required_field",
            field="author",
            severity="error",
        )

        engine = DomainRuleEngine(rules=[rule])
        chunk = Chunk(text="Content", source="test", metadata={})

        result = engine.validate_chunk(chunk)

        assert not result.passed
        assert result.violations[0].code == "RULE_AUTHOR_REQUIRED"


class TestKnownFactValidation:
    """Tests for known fact cross-referencing."""

    def test_fact_matches(self):
        """Test fact value matches."""
        fact = KnownFact(
            key="tax_rate",
            value=15,
            source="Tax Code",
            verified_date=date.today(),
            tolerance=0,
        )

        engine = DomainRuleEngine(known_facts=[fact])
        response = "The tax_rate is 15 percent."

        result = engine.validate_response(response)

        assert result.passed
        assert result.facts_checked == 1

    def test_fact_mismatch(self):
        """Test fact value doesn't match."""
        fact = KnownFact(
            key="tax_rate",
            value=15,
            source="Tax Code",
            verified_date=date.today(),
            tolerance=0,
        )

        engine = DomainRuleEngine(known_facts=[fact])
        response = "The tax_rate is 20 percent."

        result = engine.validate_response(response)

        assert not result.passed
        assert result.facts_checked == 1
        assert any("FACT_MISMATCH" in v.code for v in result.violations)

    def test_fact_within_tolerance(self):
        """Test fact value within tolerance."""
        fact = KnownFact(
            key="price",
            value=100,
            source="Official List",
            verified_date=date.today(),
            tolerance=5,
        )

        engine = DomainRuleEngine(known_facts=[fact])
        response = "The price is 103."

        result = engine.validate_response(response)

        assert result.passed


class TestRuleValidationResult:
    """Tests for RuleValidationResult dataclass."""

    def test_result_defaults(self):
        """Test result default values."""
        result = RuleValidationResult(passed=True, score=1.0)

        assert result.violations == []
        assert result.facts_checked == 0

    def test_score_calculation(self):
        """Test score is adjusted for violations."""
        engine = DomainRuleEngine(
            rules=[
                Rule(
                    name="r1",
                    description="Rule 1",
                    rule_type="required_field",
                    field="f1",
                ),
                Rule(
                    name="r2",
                    description="Rule 2",
                    rule_type="required_field",
                    field="f2",
                ),
            ]
        )

        chunk = Chunk(text="test", source="test", metadata={})
        result = engine.validate_chunk(chunk)

        # Each violation reduces score by 0.1
        assert result.score < 1.0


class TestDynamicRuleManagement:
    """Tests for adding rules and facts dynamically."""

    def test_add_rule(self):
        """Test adding a rule dynamically."""
        engine = DomainRuleEngine()

        engine.add_rule(
            Rule(
                name="new_rule",
                description="Added dynamically",
                rule_type="regex",
                condition=r"\d+",
            )
        )

        assert len(engine.rules) == 1
        assert len(engine._enabled_rules) == 1

    def test_add_disabled_rule(self):
        """Test adding a disabled rule."""
        engine = DomainRuleEngine()

        engine.add_rule(
            Rule(
                name="disabled",
                description="Disabled rule",
                rule_type="regex",
                enabled=False,
            )
        )

        assert len(engine.rules) == 1
        assert len(engine._enabled_rules) == 0

    def test_add_fact(self):
        """Test adding a fact dynamically."""
        engine = DomainRuleEngine()

        engine.add_fact(
            KnownFact(
                key="new_fact",
                value=42,
                source="test",
                verified_date=date.today(),
            )
        )

        assert "new_fact" in engine.known_facts


class TestYAMLExport:
    """Tests for YAML export functionality."""

    def test_to_yaml(self):
        """Test exporting rules to YAML."""
        engine = DomainRuleEngine(
            rules=[
                Rule(
                    name="test",
                    description="Test rule",
                    rule_type="regex",
                    condition=r"\d+",
                )
            ],
            known_facts=[
                KnownFact(
                    key="fact1",
                    value=100,
                    source="test",
                    verified_date=date(2024, 1, 1),
                )
            ],
        )

        yaml_output = engine.to_yaml()

        assert "name: test" in yaml_output
        assert "fact1" in yaml_output
        assert "2024-01-01" in yaml_output
