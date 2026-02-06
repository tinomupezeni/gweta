"""Gweta CLI application.

This module provides the command-line interface for Gweta,
built with Typer.
"""

import asyncio
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.table import Table

from gweta._version import __version__

app = typer.Typer(
    name="gweta",
    help="RAG data quality and ingestion framework",
    no_args_is_help=True,
)
console = Console()


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        console.print(f"Gweta v{__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            "-v",
            help="Show version and exit",
            callback=version_callback,
            is_eager=True,
        ),
    ] = False,
) -> None:
    """Gweta - RAG data quality and ingestion framework.

    Acquire. Validate. Ingest.
    """
    pass


@app.command()
def validate(
    path: Annotated[Path, typer.Argument(help="Path to file or directory to validate")],
    format: Annotated[str, typer.Option(help="Input format (json, jsonl, csv)")] = "json",
    threshold: Annotated[float, typer.Option(help="Minimum quality score")] = 0.6,
    output: Annotated[Optional[Path], typer.Option(help="Output file for report")] = None,
) -> None:
    """Validate chunks from a file or directory."""
    import json

    from gweta.core.types import Chunk
    from gweta.validate.chunks import ChunkValidator

    console.print(f"[blue]Validating chunks from {path}...[/blue]")

    # Load chunks
    if not path.exists():
        console.print(f"[red]Error: Path not found: {path}[/red]")
        raise typer.Exit(1)

    chunks = []
    if path.is_file():
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                chunks = [Chunk.from_dict(c) for c in data]
            else:
                chunks = [Chunk.from_dict(data)]

    # Validate
    validator = ChunkValidator()
    validator.config.min_quality_score = threshold
    report = validator.validate_batch(chunks)

    # Display results
    table = Table(title="Validation Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total Chunks", str(report.total_chunks))
    table.add_row("Passed", str(report.passed))
    table.add_row("Failed", str(report.failed))
    table.add_row("Avg Quality", f"{report.avg_quality_score:.2f}")

    console.print(table)

    if report.issues_by_type:
        console.print("\n[yellow]Issues by Type:[/yellow]")
        for issue_type, count in sorted(report.issues_by_type.items()):
            console.print(f"  {issue_type}: {count}")

    # Save report if requested
    if output:
        with open(output, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)
        console.print(f"\n[green]Report saved to {output}[/green]")


@app.command()
def crawl(
    url: Annotated[str, typer.Argument(help="URL to crawl")],
    depth: Annotated[int, typer.Option(help="Crawl depth")] = 2,
    output: Annotated[Optional[Path], typer.Option(help="Output file for chunks")] = None,
    validate_chunks: Annotated[bool, typer.Option(help="Validate extracted chunks")] = True,
) -> None:
    """Crawl a URL and extract validated content."""
    from gweta.acquire.crawler import GwetaCrawler

    console.print(f"[blue]Crawling {url} (depth={depth})...[/blue]")

    crawler = GwetaCrawler()
    result = asyncio.run(crawler.crawl(url, depth=depth))

    # Display results
    console.print(f"\n[green]Crawl complete![/green]")
    console.print(f"  Pages crawled: {result.pages_crawled}")
    console.print(f"  Pages passed: {result.pages_passed}")
    console.print(f"  Pages failed: {result.pages_failed}")
    console.print(f"  Quality score: {result.quality_score:.2f}")
    console.print(f"  Chunks extracted: {len(result.chunks)}")
    console.print(f"  Chunks rejected: {len(result.rejected_chunks)}")

    # Save chunks if requested
    if output:
        import json
        with open(output, "w", encoding="utf-8") as f:
            json.dump([c.to_dict() for c in result.chunks], f, indent=2)
        console.print(f"\n[green]Chunks saved to {output}[/green]")


@app.command()
def ingest(
    source: Annotated[str, typer.Argument(help="Source URL, file, or DSN")],
    target: Annotated[str, typer.Argument(help="Target collection name")],
    source_type: Annotated[str, typer.Option(help="Source type (web, pdf, db)")] = "web",
) -> None:
    """Ingest validated data into a vector store."""
    from gweta.acquire.crawler import GwetaCrawler
    from gweta.ingest.stores.chroma import ChromaStore

    console.print(f"[blue]Ingesting from {source} to {target}...[/blue]")

    store = ChromaStore(collection_name=target)

    if source_type == "web":
        crawler = GwetaCrawler()
        result = asyncio.run(crawler.crawl(source))

        if result.chunks:
            asyncio.run(store.add(result.chunks))

        console.print(f"\n[green]Ingested {len(result.chunks)} chunks to {target}[/green]")
    else:
        console.print(f"[yellow]Source type '{source_type}' not yet fully implemented[/yellow]")


@app.command()
def health(
    collection: Annotated[str, typer.Argument(help="Collection to check")],
    golden: Annotated[Optional[Path], typer.Option(help="Golden dataset file")] = None,
) -> None:
    """Check health of a knowledge base."""
    from gweta.ingest.stores.chroma import ChromaStore
    from gweta.validate.health import HealthChecker

    console.print(f"[blue]Checking health of {collection}...[/blue]")

    store = ChromaStore(collection_name=collection)
    checker = HealthChecker(store)

    report = asyncio.run(checker.full_health_check(golden_dataset=golden))

    console.print(report.summary())


@app.command()
def serve(
    transport: Annotated[str, typer.Option(help="Transport: stdio or http")] = "stdio",
    port: Annotated[int, typer.Option(help="HTTP port (if transport=http)")] = 8080,
) -> None:
    """Start the Gweta MCP server."""
    console.print(f"[blue]Starting Gweta MCP server ({transport})...[/blue]")

    from gweta.mcp.server import run_http, run_stdio

    if transport == "stdio":
        run_stdio()
    elif transport == "http":
        run_http(port=port)
    else:
        console.print(f"[red]Unknown transport: {transport}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
