"""Tests for MCP server components.

These tests verify the MCP server structure without requiring
the mcp library to be installed.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock

from gweta.core.types import Chunk


class TestMCPServerSetup:
    """Tests for MCP server configuration."""

    def test_server_module_imports(self):
        """Test that server module can be imported."""
        from gweta.mcp import create_server, run_stdio, run_http
        assert callable(create_server)
        assert callable(run_stdio)
        assert callable(run_http)

    def test_tools_module_imports(self):
        """Test that tools module can be imported."""
        from gweta.mcp import tools
        assert hasattr(tools, "register_tools")
        assert callable(tools.register_tools)

    def test_resources_module_imports(self):
        """Test that resources module can be imported."""
        from gweta.mcp import resources
        assert hasattr(resources, "register_resources")
        assert callable(resources.register_resources)

    def test_prompts_module_imports(self):
        """Test that prompts module can be imported."""
        from gweta.mcp import prompts
        assert hasattr(prompts, "register_prompts")
        assert callable(prompts.register_prompts)


class TestMCPToolRegistration:
    """Tests for MCP tool registration."""

    def test_register_tools_with_mock(self):
        """Test that tools are registered with decorators."""
        from gweta.mcp.tools import register_tools

        # Create mock MCP server
        mock_mcp = MagicMock()
        mock_mcp.tool = MagicMock(return_value=lambda f: f)

        # Register tools
        register_tools(mock_mcp)

        # Verify tool decorator was called
        assert mock_mcp.tool.called

    def test_tool_functions_exist(self):
        """Test that expected tool functions are defined."""
        from gweta.mcp import tools
        import inspect

        # Get all functions in the module
        source = inspect.getsource(tools)

        # Check for expected tool functions
        expected_tools = [
            "crawl_and_ingest",
            "validate_chunks",
            "check_health",
            "crawl_site",
            "ingest_from_database",
            "query_database",
            "extract_pdf",
            "fetch_api",
            "fetch_sitemap",
            "fetch_rss",
        ]

        for tool_name in expected_tools:
            assert tool_name in source, f"Tool {tool_name} not found"


class TestMCPResourceRegistration:
    """Tests for MCP resource registration."""

    def test_register_resources_with_mock(self):
        """Test that resources are registered."""
        from gweta.mcp.resources import register_resources

        mock_mcp = MagicMock()
        mock_mcp.resource = MagicMock(return_value=lambda f: f)

        register_resources(mock_mcp)

        assert mock_mcp.resource.called

    def test_resource_uris_defined(self):
        """Test that expected resource URIs are defined."""
        from gweta.mcp import resources
        import inspect

        source = inspect.getsource(resources)

        expected_resources = [
            "gweta://sources",
            "gweta://quality/",
            "gweta://rules/",
            "gweta://config",
        ]

        for uri in expected_resources:
            assert uri in source, f"Resource {uri} not found"


class TestMCPPromptRegistration:
    """Tests for MCP prompt registration."""

    def test_register_prompts_with_mock(self):
        """Test that prompts are registered."""
        from gweta.mcp.prompts import register_prompts

        mock_mcp = MagicMock()
        mock_mcp.prompt = MagicMock(return_value=lambda f: f)

        register_prompts(mock_mcp)

        assert mock_mcp.prompt.called

    def test_prompt_functions_defined(self):
        """Test that expected prompts are defined."""
        from gweta.mcp import prompts
        import inspect

        source = inspect.getsource(prompts)

        expected_prompts = [
            "plan_ingestion",
            "quality_review",
            "troubleshoot_rag",
        ]

        for prompt_name in expected_prompts:
            assert prompt_name in source, f"Prompt {prompt_name} not found"


class TestValidateChunksLogic:
    """Tests for validate_chunks tool logic."""

    def test_chunk_validation_logic(self):
        """Test the chunk validation logic used by the tool."""
        from gweta.validate.chunks import ChunkValidator

        chunks = [
            Chunk(
                text="This is a well-written chunk with substantial content.",
                source="test.txt",
                metadata={"key": "value"},
            ),
            Chunk(
                text="Short",
                source="bad.txt",
                metadata={},
            ),
        ]

        validator = ChunkValidator()
        report = validator.validate_batch(chunks)

        # Verify report structure matches what tool expects
        assert hasattr(report, "total_chunks")
        assert hasattr(report, "passed")
        assert hasattr(report, "failed")
        assert hasattr(report, "avg_quality_score")
        assert hasattr(report, "issues_by_type")


class TestHealthCheckLogic:
    """Tests for health check tool logic."""

    def test_health_report_structure(self):
        """Test health report has expected structure."""
        from gweta.validate.health import HealthReport, DuplicateReport
        from datetime import datetime

        report = HealthReport(
            timestamp=datetime.now(),
            collection="test-collection",
            total_chunks=100,
            avg_quality_score=0.85,
            duplicates=DuplicateReport(
                total_checked=100,
                duplicate_groups=5,
                duplicate_chunks=15,
            ),
            recommendations=["Remove duplicates"],
        )

        # Verify structure matches tool expectations
        assert report.collection == "test-collection"
        assert report.total_chunks == 100
        assert report.duplicates.duplicate_groups == 5
        assert len(report.recommendations) == 1


class TestCLIServeCommand:
    """Tests for CLI serve command."""

    def test_serve_command_exists(self):
        """Test that serve command is defined in CLI."""
        from gweta.cli.main import app
        import inspect

        # Get source code and check for serve command
        source = inspect.getsource(app.command.__self__.__class__)

        # Alternative: check registered commands info
        command_info = app.registered_commands
        # Typer stores command info differently, check the callback names
        callbacks = [getattr(cmd, "callback", None) for cmd in command_info if cmd]
        callback_names = [cb.__name__ if cb else None for cb in callbacks]

        # The serve function should be defined as a command
        from gweta.cli import main
        assert hasattr(main, "serve")
        assert callable(main.serve)

    def test_cli_imports_mcp(self):
        """Test that CLI can import MCP server functions."""
        # This should not raise ImportError
        from gweta.cli.main import serve
        assert callable(serve)


class TestMCPIntegration:
    """Integration tests for MCP components working together."""

    def test_all_components_registered(self):
        """Test that all components can be registered together."""
        from gweta.mcp.tools import register_tools
        from gweta.mcp.resources import register_resources
        from gweta.mcp.prompts import register_prompts

        # Create mock MCP with all required decorators
        mock_mcp = MagicMock()
        mock_mcp.tool = MagicMock(return_value=lambda f: f)
        mock_mcp.resource = MagicMock(return_value=lambda f: f)
        mock_mcp.prompt = MagicMock(return_value=lambda f: f)

        # Register all components
        register_tools(mock_mcp)
        register_resources(mock_mcp)
        register_prompts(mock_mcp)

        # All decorators should have been called
        assert mock_mcp.tool.call_count >= 6  # At least 6 tools
        assert mock_mcp.resource.call_count >= 4  # At least 4 resources
        assert mock_mcp.prompt.call_count >= 3  # At least 3 prompts
