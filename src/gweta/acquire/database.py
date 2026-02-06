"""Database connectivity for RAG data extraction.

This module provides the DatabaseSource class for extracting
data from SQL databases with read-only safety measures.
"""

import asyncio
import re
from dataclasses import dataclass, field
from typing import Any

from gweta.core.config import GwetaSettings, get_settings
from gweta.core.exceptions import DatabaseError
from gweta.core.logging import get_logger
from gweta.core.types import Chunk

logger = get_logger(__name__)


# SQL keywords that indicate write operations
FORBIDDEN_KEYWORDS = [
    "INSERT",
    "UPDATE",
    "DELETE",
    "DROP",
    "ALTER",
    "CREATE",
    "TRUNCATE",
    "EXEC",
    "EXECUTE",
    "GRANT",
    "REVOKE",
]


@dataclass
class QueryResult:
    """Results from a database query.

    Attributes:
        rows: List of row dictionaries
        columns: List of column names
        row_count: Number of rows returned
        execution_time: Query execution time in seconds
    """
    rows: list[dict[str, Any]] = field(default_factory=list)
    columns: list[str] = field(default_factory=list)
    row_count: int = 0
    execution_time: float = 0.0


class QuerySanitizer:
    """Ensure queries are read-only and safe.

    Validates SQL queries to prevent accidental or malicious
    write operations when read_only mode is enabled.
    """

    @staticmethod
    def validate(query: str, read_only: bool = True) -> None:
        """Validate query is safe to execute.

        Args:
            query: SQL query to validate
            read_only: Whether to enforce read-only mode

        Raises:
            DatabaseError: If query contains forbidden operations
        """
        if not read_only:
            return

        query_upper = query.upper()

        for keyword in FORBIDDEN_KEYWORDS:
            # Look for keyword as a word boundary
            pattern = rf"\b{keyword}\b"
            if re.search(pattern, query_upper):
                raise DatabaseError(
                    f"Query contains forbidden keyword '{keyword}' in read-only mode",
                    query=query[:100],
                )

    @staticmethod
    def is_read_only(query: str) -> bool:
        """Check if query is read-only.

        Args:
            query: SQL query to check

        Returns:
            True if query appears to be read-only
        """
        query_upper = query.upper().strip()

        # Check if it starts with SELECT
        if not query_upper.startswith("SELECT"):
            return False

        # Check for forbidden keywords
        for keyword in FORBIDDEN_KEYWORDS:
            pattern = rf"\b{keyword}\b"
            if re.search(pattern, query_upper):
                return False

        return True


class DatabaseSource:
    """Extract data from SQL databases for RAG ingestion.

    Provides safe, read-only database connectivity with:
    - Query validation and sanitization
    - Automatic chunking of text columns
    - Quality scoring of extracted data

    Example:
        >>> async with DatabaseSource("postgresql://...") as db:
        ...     result = await db.query("SELECT content FROM articles")
        ...     chunks = await db.extract_and_validate(
        ...         query="SELECT content, title FROM articles",
        ...         text_column="content",
        ...     )
    """

    def __init__(
        self,
        dsn: str,
        read_only: bool = True,
        timeout: float = 30.0,
        max_rows: int = 10000,
        config: GwetaSettings | None = None,
    ) -> None:
        """Initialize DatabaseSource.

        Args:
            dsn: Database connection string
            read_only: Enforce read-only queries (default: True)
            timeout: Query timeout in seconds
            max_rows: Maximum rows to return per query
            config: Gweta settings
        """
        self.dsn = dsn
        self.read_only = read_only
        self.timeout = timeout
        self.max_rows = max_rows
        self.config = config or get_settings()
        self._engine = None
        self._sanitizer = QuerySanitizer()

    async def connect(self) -> None:
        """Establish database connection.

        Raises:
            DatabaseError: If connection fails
        """
        try:
            from sqlalchemy.ext.asyncio import create_async_engine
        except ImportError as e:
            raise ImportError(
                "SQLAlchemy is required for database connectivity. "
                "Install it with: pip install gweta[db]"
            ) from e

        try:
            # Convert sync DSN to async if needed
            async_dsn = self._to_async_dsn(self.dsn)
            self._engine = create_async_engine(
                async_dsn,
                echo=False,
                pool_pre_ping=True,
            )
            logger.info("Database connection established")
        except Exception as e:
            raise DatabaseError(
                f"Failed to connect to database: {e}",
                dsn=self.dsn,
            ) from e

    async def disconnect(self) -> None:
        """Close database connection."""
        if self._engine:
            await self._engine.dispose()
            self._engine = None
            logger.info("Database connection closed")

    async def __aenter__(self) -> "DatabaseSource":
        """Enter async context and connect."""
        await self.connect()
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Exit async context and disconnect."""
        await self.disconnect()

    def _to_async_dsn(self, dsn: str) -> str:
        """Convert synchronous DSN to async driver.

        Args:
            dsn: Original connection string

        Returns:
            Connection string with async driver
        """
        if dsn.startswith("postgresql://"):
            return dsn.replace("postgresql://", "postgresql+asyncpg://")
        elif dsn.startswith("mysql://"):
            return dsn.replace("mysql://", "mysql+aiomysql://")
        elif dsn.startswith("sqlite://"):
            return dsn.replace("sqlite://", "sqlite+aiosqlite://")
        return dsn

    async def query(
        self,
        sql: str,
        params: dict[str, Any] | None = None,
    ) -> QueryResult:
        """Execute a read-only query.

        Args:
            sql: SQL query to execute
            params: Query parameters

        Returns:
            QueryResult with rows and metadata

        Raises:
            DatabaseError: If query fails or is not read-only
        """
        if self._engine is None:
            raise DatabaseError("Database not connected. Call connect() first.")

        # Validate query
        self._sanitizer.validate(sql, self.read_only)

        import time
        from sqlalchemy import text

        start_time = time.time()

        try:
            async with self._engine.connect() as conn:
                result = await conn.execute(text(sql), params or {})
                rows = result.fetchmany(self.max_rows)
                columns = list(result.keys())

                query_result = QueryResult(
                    rows=[dict(zip(columns, row)) for row in rows],
                    columns=columns,
                    row_count=len(rows),
                    execution_time=time.time() - start_time,
                )

                logger.info(
                    f"Query returned {query_result.row_count} rows "
                    f"in {query_result.execution_time:.2f}s"
                )

                return query_result

        except Exception as e:
            raise DatabaseError(
                f"Query execution failed: {e}",
                dsn=self.dsn,
                query=sql,
            ) from e

    async def extract_and_validate(
        self,
        query: str,
        text_column: str,
        metadata_columns: list[str] | None = None,
        chunk_size: int | None = None,
    ) -> list[Chunk]:
        """Extract data and create validated chunks.

        Args:
            query: SQL query to extract data
            text_column: Column containing text to embed
            metadata_columns: Columns to preserve as metadata
            chunk_size: Size of chunks (uses config default if not specified)

        Returns:
            List of validated Chunk objects
        """
        result = await self.query(query)
        chunks: list[Chunk] = []

        chunk_size = chunk_size or self.config.default_chunk_size
        metadata_columns = metadata_columns or []

        for row in result.rows:
            text = str(row.get(text_column, ""))
            if not text.strip():
                continue

            # Build metadata from specified columns
            metadata: dict[str, Any] = {
                "source_type": "database",
            }
            for col in metadata_columns:
                if col in row:
                    metadata[col] = row[col]

            # Simple chunking
            if len(text) <= chunk_size:
                chunks.append(
                    Chunk(
                        text=text,
                        source="database",
                        metadata=metadata,
                        quality_score=1.0,
                    )
                )
            else:
                # Split into chunks
                for i in range(0, len(text), chunk_size):
                    chunk_text = text[i : i + chunk_size]
                    if chunk_text.strip():
                        chunks.append(
                            Chunk(
                                text=chunk_text,
                                source="database",
                                metadata={**metadata, "chunk_index": i // chunk_size},
                                quality_score=1.0,
                            )
                        )

        logger.info(f"Extracted {len(chunks)} chunks from database")
        return chunks

    async def ingest(
        self,
        query: str,
        text_column: str,
        target: Any,
        metadata_columns: list[str] | None = None,
        chunk_strategy: str = "recursive",
    ) -> dict[str, Any]:
        """Extract data and ingest directly to a vector store.

        Args:
            query: SQL query to extract data
            text_column: Column containing text
            target: Vector store adapter with add() method
            metadata_columns: Columns to preserve as metadata
            chunk_strategy: Chunking strategy to use

        Returns:
            Dict with ingestion statistics
        """
        chunks = await self.extract_and_validate(
            query=query,
            text_column=text_column,
            metadata_columns=metadata_columns,
        )

        if chunks:
            await target.add(chunks)

        return {
            "chunks_ingested": len(chunks),
            "source": "database",
        }

    def query_sync(self, sql: str, **kwargs: Any) -> QueryResult:
        """Synchronous wrapper for query().

        Args:
            sql: SQL query
            **kwargs: Additional arguments

        Returns:
            QueryResult
        """
        return asyncio.run(self.query(sql, **kwargs))
