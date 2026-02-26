#!/usr/bin/env python3
"""Entry point for the cosmos-chessbot MCP server."""

from cosmos_chessbot.mcp import create_server

if __name__ == "__main__":
    create_server().run()
