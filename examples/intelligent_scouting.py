"""Intelligent Scouting Example.

This example demonstrates how to use the IntelligenceScout to
autonomously find and extract data from the web based on a
natural language goal.

## Prerequisites

```bash
pip install "gweta[all]"
export OPENAI_API_KEY="sk-..."  # or any other supported LLM key
```
"""

import asyncio
import os
from gweta.intelligence import IntelligenceScout

async def run_scout():
    # Check for API key
    if not os.environ.get("OPENAI_API_KEY") and not os.environ.get("ANTHROPIC_API_KEY"):
        print("Warning: No LLM API key found in environment variables.")
        print("Please set OPENAI_API_KEY or ANTHROPIC_API_KEY to run this example.")
        return

    # Initialize the scout
    # You can specify any model supported by LiteLLM
    scout = IntelligenceScout(model="gpt-4o-mini")

    # Define a goal
    goal = "Find the current business registration fees for private limited companies in Zimbabwe"

    print(f"Goal: {goal}")
    print("Scouting the web (this may take a minute)...")

    # Perform scouting
    # We'll limit to 3 pages for this example
    result = await scout.scout(
        goal=goal,
        max_pages=3,
        extract_schema={
            "type": "object",
            "properties": {
                "fees": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "description": {"type": "string"},
                            "amount": {"type": "string"},
                            "currency": {"type": "string"}
                        }
                    }
                },
                "source_url": {"type": "string"}
            }
        }
    )

    print("\n--- Scouting Results ---")
    print(f"Search Queries Generated: {result.queries}")
    print(f"URLs Discovered: {len(result.urls_discovered)}")
    print(f"Pages Visited: {result.pages_visited}")
    print(f"Chunks Extracted: {result.chunks_extracted}")

    print("\n--- Extracted Structured Data ---")
    for idx, data in enumerate(result.extracted_data):
        print(f"\nResult {idx + 1}:")
        import json
        print(json.dumps(data, indent=2))

    if not result.extracted_data:
        print("No structured data was extracted. Check the raw chunks for content.")

if __name__ == "__main__":
    asyncio.run(run_scout())
