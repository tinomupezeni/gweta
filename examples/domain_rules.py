"""Domain Rules Example.

This example demonstrates how to use the DomainRuleEngine
for domain-specific validation with YAML-based rules and
known facts for cross-referencing.

## Key Features

1. **YAML-based Rules**: Define rules in a human-readable format
2. **Pattern Matching**: Regex-based content validation
3. **Numerical Ranges**: Validate numbers are within expected bounds
4. **Required Fields**: Ensure metadata contains required fields
5. **Known Facts**: Cross-reference content against verified facts

## Example Domains

- Zimbabwe business regulations
- Financial compliance
- Medical/legal document validation
"""

import asyncio
from datetime import date
from pathlib import Path

from gweta.core.types import Chunk
from gweta.validate.rules import (
    DomainRuleEngine,
    Rule,
    KnownFact,
)


def example_programmatic_rules():
    """Example: Creating rules programmatically."""
    print("=" * 60)
    print("Example 1: Programmatic Rule Definition")
    print("=" * 60)

    # Create rules programmatically
    rules = [
        Rule(
            name="zim_currency_format",
            description="Validate Zimbabwe currency format",
            rule_type="regex",
            condition=r"\$\d+|USD\s*\d+|ZWL\s*\d+",
            field="required",  # Pattern must be present
            severity="warning",
            message="Currency values should use standard format (USD/ZWL)",
        ),
        Rule(
            name="company_registration",
            description="Check for company registration number",
            rule_type="regex",
            condition=r"\b[A-Z]{2}\d{4,8}\b",
            field="required",
            severity="error",
            message="Document should include company registration number",
        ),
        Rule(
            name="valid_year_range",
            description="Check year references are reasonable",
            rule_type="numerical_range",
            condition={"min": 1980, "max": 2030},
            severity="warning",
            message="Year ${value} seems outside expected range",
        ),
    ]

    # Create known facts for cross-referencing
    known_facts = [
        KnownFact(
            key="minimum_wage",
            value=350,
            source="Zimbabwe Labour Act 2024",
            verified_date=date(2024, 1, 1),
            tolerance=10,
            unit="USD",
        ),
        KnownFact(
            key="vat_rate",
            value=15,
            source="ZIMRA Tax Guidelines",
            verified_date=date(2024, 1, 1),
            tolerance=0,
            unit="%",
        ),
        KnownFact(
            key="company_tax_rate",
            value=25,
            source="ZIMRA Corporate Tax",
            verified_date=date(2024, 1, 1),
            tolerance=0,
            unit="%",
        ),
    ]

    # Initialize engine
    engine = DomainRuleEngine(rules=rules, known_facts=known_facts)

    # Create test chunks
    chunks = [
        Chunk(
            text="ABC Corp (Reg: ZW12345) reported revenue of $50,000 USD in 2024.",
            source="annual_report.pdf",
            metadata={"document_type": "financial"},
        ),
        Chunk(
            text="The company paid employees based on minimum_wage: 350 USD.",
            source="hr_policy.pdf",
            metadata={"document_type": "policy"},
        ),
        Chunk(
            text="Tax calculations use vat_rate: 18 percent which is applied to all sales.",
            source="accounting.pdf",
            metadata={"document_type": "financial"},
        ),
    ]

    # Validate chunks
    print("\nValidating chunks against domain rules:")
    print("-" * 40)

    for i, chunk in enumerate(chunks, 1):
        result = engine.validate_chunk(chunk)
        print(f"\nChunk {i}: {chunk.source}")
        print(f"  Passed: {result.passed}")
        print(f"  Score: {result.score:.2f}")
        if result.violations:
            for v in result.violations:
                print(f"  [{v.severity.upper()}] {v.code}: {v.message}")
        else:
            print("  No violations")

    # Validate AI response against known facts
    print("\n" + "=" * 60)
    print("Validating AI Response Against Known Facts")
    print("=" * 60)

    response = """
    Based on current Zimbabwe regulations, the minimum_wage is 350 USD per month.
    The VAT rate (vat_rate) is 18% which is applied at point of sale.
    Companies pay company_tax_rate: 25 percent on profits.
    """

    result = engine.validate_response(response)
    print(f"\nFacts checked: {result.facts_checked}")
    print(f"Passed: {result.passed}")
    print(f"Score: {result.score:.2f}")

    if result.violations:
        print("\nViolations found:")
        for v in result.violations:
            print(f"  [{v.severity.upper()}] {v.code}: {v.message}")

    return engine


def example_yaml_rules():
    """Example: Loading rules from YAML."""
    print("\n" + "=" * 60)
    print("Example 2: YAML-Based Rules")
    print("=" * 60)

    # Define YAML content (normally from file)
    yaml_content = """
rules:
  - name: phone_number_format
    description: Validate phone numbers are in standard format
    type: regex
    pattern: "\\+263\\s?\\d{2}\\s?\\d{3}\\s?\\d{4}"
    field: required
    severity: warning
    message: Phone numbers should use +263 format

  - name: email_format
    description: Check for valid email addresses
    type: regex
    pattern: "[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}"
    severity: info
    message: Email format validation
    enabled: true

  - name: price_range
    description: Prices should be within reasonable range
    type: numerical_range
    condition:
      min: 0.01
      max: 1000000
    severity: warning
    message: Price ${value} seems unusual

  - name: document_author
    description: Documents must have an author
    type: required_field
    field: author
    severity: error
    message: Document is missing required author metadata

known_facts:
  - key: exchange_rate
    value: 28000
    source: RBZ Official Rate
    verified_date: "2024-06-01"
    tolerance: 1000
    unit: ZWL/USD

  - key: inflation_rate
    value: 55.3
    source: ZIMSTAT Monthly Report
    verified_date: "2024-05-15"
    tolerance: 5
    unit: "%"
"""

    # Create engine from dictionary (simulating YAML load)
    import yaml
    data = yaml.safe_load(yaml_content)
    engine = DomainRuleEngine.from_dict(data)

    print(f"\nLoaded {len(engine.rules)} rules")
    print(f"Loaded {len(engine.known_facts)} known facts")

    # List rules
    print("\nRules:")
    for rule in engine.rules:
        status = "[ON]" if rule.enabled else "[OFF]"
        print(f"  {status} {rule.name} ({rule.rule_type}) - {rule.severity}")

    # List known facts
    print("\nKnown Facts:")
    for key, fact in engine.known_facts.items():
        print(f"  {key}: {fact.value} {fact.unit or ''} (from {fact.source})")

    # Test validation
    chunks = [
        Chunk(
            text="Contact us at +263 77 123 4567 or email@example.com",
            source="contact.txt",
            metadata={"author": "John Doe"},
        ),
        Chunk(
            text="Call 077-123-4567 for more info",
            source="flyer.txt",
            metadata={},  # Missing author
        ),
    ]

    print("\nValidating chunks:")
    print("-" * 40)

    for chunk in chunks:
        result = engine.validate_chunk(chunk)
        print(f"\n{chunk.source}:")
        print(f"  Passed: {result.passed}, Score: {result.score:.2f}")
        for v in result.violations:
            print(f"  [{v.severity.upper()}] {v.message}")

    # Export to YAML
    print("\n" + "=" * 60)
    print("Exporting Rules to YAML")
    print("=" * 60)
    print("\nGenerated YAML (first 30 lines):")
    yaml_output = engine.to_yaml()
    lines = yaml_output.split('\n')[:30]
    print('\n'.join(lines))
    if len(yaml_output.split('\n')) > 30:
        print("...")

    return engine


def example_industry_specific():
    """Example: Industry-specific rule sets."""
    print("\n" + "=" * 60)
    print("Example 3: Industry-Specific Rules")
    print("=" * 60)

    # Medical document rules
    medical_rules = [
        Rule(
            name="patient_id",
            description="Patient ID format",
            rule_type="regex",
            condition=r"PAT-\d{6}",
            field="required",
            severity="error",
            message="Missing or invalid patient ID",
        ),
        Rule(
            name="dosage_range",
            description="Medication dosage reasonable",
            rule_type="numerical_range",
            condition={"min": 0, "max": 5000},
            severity="warning",
            message="Dosage ${value}mg seems unusual",
        ),
    ]

    # Legal document rules
    legal_rules = [
        Rule(
            name="case_number",
            description="Legal case number format",
            rule_type="regex",
            condition=r"CASE-\d{4}-\d{4}",
            field="required",
            severity="error",
            message="Missing case reference number",
        ),
        Rule(
            name="court_reference",
            description="Court must be specified",
            rule_type="required_field",
            field="court",
            severity="error",
            message="Court jurisdiction must be specified",
        ),
    ]

    # Financial document rules
    financial_rules = [
        Rule(
            name="account_number",
            description="Bank account format",
            rule_type="regex",
            condition=r"\d{10,16}",
            severity="warning",
            message="No valid account number found",
        ),
        Rule(
            name="transaction_limit",
            description="Transaction amount limits",
            rule_type="numerical_range",
            condition={"min": 0.01, "max": 10000000},
            severity="error",
            message="Transaction ${value} exceeds limits",
        ),
    ]

    domains = [
        ("Medical", medical_rules, []),
        ("Legal", legal_rules, []),
        ("Financial", financial_rules, []),
    ]

    for name, rules, facts in domains:
        engine = DomainRuleEngine(rules=rules, known_facts=facts)
        print(f"\n{name} Domain: {len(rules)} rules configured")
        for rule in rules:
            print(f"  - {rule.name}: {rule.description}")


def example_dynamic_rules():
    """Example: Adding rules dynamically."""
    print("\n" + "=" * 60)
    print("Example 4: Dynamic Rule Management")
    print("=" * 60)

    engine = DomainRuleEngine()

    # Add rules one by one
    engine.add_rule(Rule(
        name="profanity_check",
        description="Check for inappropriate content",
        rule_type="regex",
        condition=r"\b(bad|inappropriate)\b",  # Simplified example
        severity="error",
        message="Content may contain inappropriate language",
    ))

    # Add known facts dynamically
    engine.add_fact(KnownFact(
        key="current_president",
        value="E.D. Mnangagwa",
        source="Government of Zimbabwe",
        verified_date=date.today(),
    ))

    print(f"Engine has {len(engine.rules)} rules and {len(engine.known_facts)} facts")

    # Test response validation
    response = "The current_president is E.D. Mnangagwa."
    result = engine.validate_response(response)
    print(f"\nValidating fact reference:")
    print(f"  Facts checked: {result.facts_checked}")
    print(f"  Passed: {result.passed}")


def main():
    """Run all examples."""
    print("Gweta Domain Rules Engine Examples")
    print("=" * 60)
    print()

    example_programmatic_rules()
    example_yaml_rules()
    example_industry_specific()
    example_dynamic_rules()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
