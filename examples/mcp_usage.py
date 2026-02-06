"""MCP Server Usage Example.

This example demonstrates how to use Gweta as an MCP server
for AI agents like Claude Desktop.

## Setup for Claude Desktop

Add this to your Claude Desktop configuration
(claude_desktop_config.json):

```json
{
  "mcpServers": {
    "gweta": {
      "command": "gweta",
      "args": ["serve"]
    }
  }
}
```

Or with Python directly:

```json
{
  "mcpServers": {
    "gweta": {
      "command": "python",
      "args": ["-m", "gweta.mcp"]
    }
  }
}
```

## Available Tools

Once connected, Claude can use these tools:

1. **crawl_and_ingest** - Crawl a website and ingest to vector store
2. **validate_chunks** - Validate chunks without ingesting
3. **check_health** - Check knowledge base health
4. **crawl_site** - Crawl and preview content
5. **ingest_from_database** - Extract from database
6. **query_database** - Execute read-only queries
7. **extract_pdf** - Extract content from PDFs
8. **fetch_api** - Fetch data from REST APIs
9. **fetch_sitemap** - Discover URLs from sitemaps
10. **fetch_rss** - Parse RSS/Atom feeds

## Available Resources

- gweta://sources - List registered data sources
- gweta://quality/{collection} - Get quality report for collection
- gweta://rules/{domain} - Get validation rules for domain
- gweta://config - Get current configuration

## Available Prompts

- plan_ingestion - Plan an ingestion job
- quality_review - Review knowledge base quality
- troubleshoot_rag - Diagnose RAG issues

## Example Conversations with Claude

### Ingest a Website

User: "Please crawl the Python documentation and add it to my knowledge base"

Claude will use:
- crawl_and_ingest(url="https://docs.python.org/3/", depth=2, target_collection="python-docs")

### Check Quality

User: "How is the quality of my python-docs collection?"

Claude will use:
- check_health(collection="python-docs")
- Access gweta://quality/python-docs resource

### Extract from PDF

User: "Extract this PDF and add it to the legal-docs collection"

Claude will use:
- extract_pdf(path="/path/to/document.pdf", target_collection="legal-docs")

### Database Ingestion

User: "Import all articles from our database into the articles collection"

Claude will use:
- ingest_from_database(
    dsn="postgresql://...",
    query="SELECT content, title FROM articles",
    target_collection="articles",
    text_column="content",
    metadata_columns=["title"]
)
"""


def main():
    """Demonstrate MCP server capabilities."""
    print("Gweta MCP Server")
    print("=" * 50)
    print()
    print("To start the server, run:")
    print()
    print("  gweta serve              # stdio transport (for Claude Desktop)")
    print("  gweta serve --transport http --port 8080  # HTTP transport")
    print()
    print("Or via Python:")
    print()
    print("  python -m gweta.mcp")
    print()

    # List available tools
    print("Available MCP Tools:")
    print("-" * 40)
    tools = [
        ("crawl_and_ingest", "Crawl website and ingest to vector store"),
        ("validate_chunks", "Validate chunks without ingesting"),
        ("check_health", "Check knowledge base health"),
        ("crawl_site", "Crawl and preview content"),
        ("ingest_from_database", "Extract and ingest from database"),
        ("query_database", "Execute read-only SQL queries"),
        ("extract_pdf", "Extract content from PDF files"),
        ("fetch_api", "Fetch data from REST APIs"),
        ("fetch_sitemap", "Discover URLs from sitemaps"),
        ("fetch_rss", "Parse RSS/Atom feeds"),
    ]

    for name, description in tools:
        print(f"  {name:<25} {description}")

    print()
    print("Available MCP Resources:")
    print("-" * 40)
    resources = [
        ("gweta://sources", "List all registered data sources"),
        ("gweta://quality/{collection}", "Quality report for collection"),
        ("gweta://rules/{domain}", "Validation rules for domain"),
        ("gweta://config", "Current Gweta configuration"),
    ]

    for uri, description in resources:
        print(f"  {uri:<30} {description}")

    print()
    print("Available MCP Prompts:")
    print("-" * 40)
    prompts = [
        ("plan_ingestion", "Plan a data ingestion job"),
        ("quality_review", "Review knowledge base quality"),
        ("troubleshoot_rag", "Diagnose RAG quality issues"),
    ]

    for name, description in prompts:
        print(f"  {name:<25} {description}")


if __name__ == "__main__":
    main()
