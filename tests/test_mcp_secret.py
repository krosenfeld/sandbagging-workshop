"""Tests for the MCP secret server tool."""

import asyncio

from sandbagging_workshop.mcp_secret import get_secret


def test_get_secret_returns_sauce():
    result = asyncio.run(get_secret.run({}))
    assert result.content[0].text == "sauce"
