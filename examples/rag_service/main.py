"""FastAPI RAG Service using Gweta.

A complete example of building a RAG service with Gweta.

Run:
    uvicorn main:app --reload

Endpoints:
    GET  /health     - Health check
    POST /query      - Query the knowledge base
    POST /ingest     - Ingest content
    GET  /categories - List categories
    GET  /sources    - List sources
"""
from contextlib import asynccontextmanager
from typing import Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from rag_engine import RAGEngine, QueryRequest


# Pydantic models for API
class QueryBody(BaseModel):
    """Query request body."""
    query: str
    top_k: int = 5
    category: Optional[str] = None
    pathway: Optional[str] = None


class QueryResultModel(BaseModel):
    """Single query result."""
    text: str
    source: str
    score: float
    metadata: dict


class QueryResponse(BaseModel):
    """Query response."""
    results: list[QueryResultModel]
    total: int


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    total_chunks: int
    total_sources: int
    categories: list[str]
    pathways: list[str]


class IngestBody(BaseModel):
    """Ingest request body."""
    text: str
    source: str
    category: Optional[str] = None
    pathway: Optional[str] = None
    metadata: dict = {}


class IngestResponse(BaseModel):
    """Ingest response."""
    added: int
    message: str


# Global engine instance
engine: RAGEngine


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    global engine

    # Startup
    print("Initializing RAG Engine...")
    engine = RAGEngine(
        collection_name="gweta-rag-demo",
        persist_dir="./data/chroma",
    )
    await engine.initialize()

    # Load sample data if exists
    sample_path = Path("./data/sample.json")
    if sample_path.exists():
        try:
            added = await engine.ingest_json(str(sample_path))
            print(f"Loaded {added} chunks from sample data")
        except Exception as e:
            print(f"Warning: Could not load sample data: {e}")

    health = engine.get_health()
    print(f"RAG Engine ready: {health['total_chunks']} chunks")

    yield

    # Shutdown
    print("Shutting down RAG Engine...")


app = FastAPI(
    title="Gweta RAG Service",
    description="A RAG service built with Gweta",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return engine.get_health()


@app.post("/query", response_model=QueryResponse)
async def query(body: QueryBody):
    """Query the knowledge base.

    Args:
        body: Query parameters including:
            - query: Search query text
            - top_k: Number of results (default 5)
            - category: Filter by category (optional)
            - pathway: Filter by pathway (optional)

    Returns:
        List of matching results with scores
    """
    try:
        request = QueryRequest(
            query=body.query,
            top_k=body.top_k,
            category=body.category,
            pathway=body.pathway,
        )
        results = await engine.query(request)

        return QueryResponse(
            results=[
                QueryResultModel(
                    text=r.text,
                    source=r.source,
                    score=r.score,
                    metadata=r.metadata,
                )
                for r in results
            ],
            total=len(results),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest", response_model=IngestResponse)
async def ingest(body: IngestBody):
    """Ingest a single piece of content.

    Args:
        body: Content to ingest including:
            - text: The content text
            - source: Source identifier
            - category: Category (optional)
            - pathway: Pathway (optional)
            - metadata: Additional metadata (optional)

    Returns:
        Number of chunks added
    """
    try:
        from gweta import Chunk

        metadata = body.metadata.copy()
        if body.category:
            metadata["category"] = body.category
        if body.pathway:
            metadata["pathway"] = body.pathway

        chunk = Chunk(
            text=body.text,
            source=body.source,
            metadata=metadata,
        )

        added = await engine.ingest_chunks([chunk])
        return IngestResponse(
            added=added,
            message=f"Successfully added {added} chunk(s)",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/categories")
async def get_categories():
    """Get all categories in the knowledge base."""
    return await engine.get_categories()


@app.get("/sources")
async def get_sources():
    """Get all sources in the knowledge base."""
    return await engine.get_sources()


@app.get("/pathways")
async def get_pathways():
    """Get all pathways in the knowledge base."""
    return await engine.get_pathways()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
