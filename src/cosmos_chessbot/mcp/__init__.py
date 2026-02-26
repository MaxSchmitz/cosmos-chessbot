"""MCP server for direct robot control."""

from .server import mcp


def create_server():
    """Return the configured FastMCP server instance."""
    return mcp
