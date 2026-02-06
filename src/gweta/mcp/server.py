"""FastMCP server setup for Gweta.

This module creates and configures the MCP server
with tools, resources, and prompts.
"""

from typing import Any

from gweta._version import __version__
from gweta.core.logging import get_logger

logger = get_logger(__name__)

# Global server instance
_server: Any = None


def create_server() -> Any:
    """Create and configure the MCP server.

    Returns:
        Configured FastMCP server instance
    """
    global _server

    if _server is not None:
        return _server

    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as e:
        raise ImportError(
            "MCP SDK is required for the Gweta server. "
            "Install it with: pip install gweta[mcp]"
        ) from e

    _server = FastMCP(
        name="gweta",
        version=__version__,
    )

    # Register tools, resources, and prompts
    from gweta.mcp.tools import register_tools
    from gweta.mcp.resources import register_resources
    from gweta.mcp.prompts import register_prompts

    register_tools(_server)
    register_resources(_server)
    register_prompts(_server)

    logger.info(f"Gweta MCP server v{__version__} initialized")
    return _server


def run_stdio() -> None:
    """Run the MCP server with stdio transport.

    This is the default transport for Claude Desktop
    and other local MCP clients.
    """
    server = create_server()
    logger.info("Starting Gweta MCP server (stdio transport)")
    server.run()


def run_http(port: int = 8080) -> None:
    """Run the MCP server with HTTP transport.

    Args:
        port: HTTP port to listen on

    This transport allows remote agents to connect
    to the Gweta server.
    """
    server = create_server()
    logger.info(f"Starting Gweta MCP server (HTTP transport on port {port})")
    server.run(transport="http", port=port)
