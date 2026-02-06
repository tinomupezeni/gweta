"""MCP Server for Gweta.

This module provides the FastMCP server that exposes
Gweta's capabilities as MCP tools, resources, and prompts.
"""

from gweta.mcp.server import create_server, run_stdio, run_http

__all__ = [
    "create_server",
    "run_stdio",
    "run_http",
]
