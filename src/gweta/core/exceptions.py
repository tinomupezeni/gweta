"""Custom exceptions for Gweta.

This module defines the exception hierarchy used throughout Gweta
for clear error handling and reporting.
"""


class GwetaError(Exception):
    """Base exception for all Gweta errors.

    All Gweta-specific exceptions inherit from this class,
    allowing users to catch all Gweta errors with a single except clause.
    """

    def __init__(self, message: str, details: dict | None = None) -> None:
        """Initialize GwetaError.

        Args:
            message: Human-readable error message
            details: Optional dictionary with additional error details
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} - Details: {self.details}"
        return self.message


class ValidationError(GwetaError):
    """Raised when validation fails.

    This is raised when chunks or extracted content fail to meet
    quality thresholds or violate validation rules.
    """

    def __init__(
        self,
        message: str,
        chunk_id: str | None = None,
        issues: list | None = None,
    ) -> None:
        """Initialize ValidationError.

        Args:
            message: Human-readable error message
            chunk_id: ID of the chunk that failed validation
            issues: List of validation issues
        """
        details = {}
        if chunk_id:
            details["chunk_id"] = chunk_id
        if issues:
            details["issues"] = issues
        super().__init__(message, details)
        self.chunk_id = chunk_id
        self.issues = issues or []


class ConfigurationError(GwetaError):
    """Raised when configuration is invalid.

    This is raised when settings are misconfigured, required
    configuration is missing, or configuration files are malformed.
    """

    def __init__(
        self,
        message: str,
        setting_name: str | None = None,
        setting_value: str | None = None,
    ) -> None:
        """Initialize ConfigurationError.

        Args:
            message: Human-readable error message
            setting_name: Name of the problematic setting
            setting_value: Value that caused the error
        """
        details = {}
        if setting_name:
            details["setting_name"] = setting_name
        if setting_value:
            details["setting_value"] = setting_value
        super().__init__(message, details)
        self.setting_name = setting_name
        self.setting_value = setting_value


class AcquisitionError(GwetaError):
    """Raised when data acquisition fails.

    This is raised when crawling, API fetching, PDF extraction,
    or database queries encounter errors.
    """

    def __init__(
        self,
        message: str,
        source: str | None = None,
        source_type: str | None = None,
    ) -> None:
        """Initialize AcquisitionError.

        Args:
            message: Human-readable error message
            source: The source URL, path, or DSN
            source_type: Type of source (web, pdf, api, database)
        """
        details = {}
        if source:
            details["source"] = source
        if source_type:
            details["source_type"] = source_type
        super().__init__(message, details)
        self.source = source
        self.source_type = source_type


class CrawlError(AcquisitionError):
    """Raised when web crawling fails.

    Specific subclass of AcquisitionError for crawling issues.
    """

    def __init__(
        self,
        message: str,
        url: str | None = None,
        status_code: int | None = None,
    ) -> None:
        """Initialize CrawlError.

        Args:
            message: Human-readable error message
            url: The URL that failed
            status_code: HTTP status code if applicable
        """
        super().__init__(message, source=url, source_type="web")
        if status_code:
            self.details["status_code"] = status_code
        self.url = url
        self.status_code = status_code


class PDFExtractionError(AcquisitionError):
    """Raised when PDF extraction fails.

    Specific subclass of AcquisitionError for PDF issues.
    """

    def __init__(
        self,
        message: str,
        file_path: str | None = None,
        page_number: int | None = None,
    ) -> None:
        """Initialize PDFExtractionError.

        Args:
            message: Human-readable error message
            file_path: Path to the PDF file
            page_number: Page number where extraction failed
        """
        super().__init__(message, source=file_path, source_type="pdf")
        if page_number:
            self.details["page_number"] = page_number
        self.file_path = file_path
        self.page_number = page_number


class DatabaseError(AcquisitionError):
    """Raised when database operations fail.

    Specific subclass of AcquisitionError for database issues.
    """

    def __init__(
        self,
        message: str,
        dsn: str | None = None,
        query: str | None = None,
    ) -> None:
        """Initialize DatabaseError.

        Args:
            message: Human-readable error message
            dsn: Database connection string (sanitized)
            query: The query that failed (truncated for safety)
        """
        # Sanitize DSN to remove password
        safe_dsn = dsn
        if dsn and "@" in dsn:
            # Remove password from DSN for logging
            parts = dsn.split("@")
            if len(parts) >= 2:
                prefix = parts[0]
                if ":" in prefix:
                    user_part = prefix.rsplit(":", 1)[0]
                    safe_dsn = f"{user_part}:****@{'@'.join(parts[1:])}"

        super().__init__(message, source=safe_dsn, source_type="database")
        if query:
            # Truncate query for safety
            self.details["query"] = query[:200] + "..." if len(query) > 200 else query
        self.dsn = safe_dsn
        self.query = query


class IngestionError(GwetaError):
    """Raised when ingestion into vector store fails.

    This is raised when loading chunks into Chroma, Qdrant,
    Pinecone, or other vector stores fails.
    """

    def __init__(
        self,
        message: str,
        store_type: str | None = None,
        collection: str | None = None,
        chunk_count: int | None = None,
    ) -> None:
        """Initialize IngestionError.

        Args:
            message: Human-readable error message
            store_type: Type of vector store (chroma, qdrant, etc.)
            collection: Collection/index name
            chunk_count: Number of chunks being ingested
        """
        details = {}
        if store_type:
            details["store_type"] = store_type
        if collection:
            details["collection"] = collection
        if chunk_count:
            details["chunk_count"] = chunk_count
        super().__init__(message, details)
        self.store_type = store_type
        self.collection = collection
        self.chunk_count = chunk_count


class RuleEngineError(GwetaError):
    """Raised when domain rule evaluation fails.

    This is raised when the domain rule engine encounters
    invalid rules, missing facts, or evaluation errors.
    """

    def __init__(
        self,
        message: str,
        rule_name: str | None = None,
        rule_file: str | None = None,
    ) -> None:
        """Initialize RuleEngineError.

        Args:
            message: Human-readable error message
            rule_name: Name of the rule that failed
            rule_file: Path to the rules file
        """
        details = {}
        if rule_name:
            details["rule_name"] = rule_name
        if rule_file:
            details["rule_file"] = rule_file
        super().__init__(message, details)
        self.rule_name = rule_name
        self.rule_file = rule_file
