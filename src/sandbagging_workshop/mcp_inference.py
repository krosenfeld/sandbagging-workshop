"""
FastMCP server that exposes a "chat" tool for a single inference endpoint.

Run standalone:
    start-inference cedar
"""

from __future__ import annotations

import click
from fastmcp import FastMCP

from sandbagging_workshop.inference import (
    Client,
    cedar,
    maple,
    trusted_reference,
    yew,
)

ENDPOINTS = {
    "cedar": cedar,
    "yew": yew,
    "maple": maple,
    "trusted_reference": trusted_reference,
}

# Module-level references populated by main() before mcp.run()
_client: Client | None = None
mcp: FastMCP | None = None


@click.command()
@click.argument("endpoint_name", type=click.Choice(list(ENDPOINTS)))
def main(endpoint_name: str) -> None:
    """Start an MCP server for the given inference endpoint."""
    global _client, mcp

    endpoint = ENDPOINTS[endpoint_name]
    _client = Client(endpoint=endpoint)
    mcp = FastMCP(f"{endpoint_name}-inference")

    @mcp.tool()
    def chat(prompt: str) -> str:
        """Send a prompt to the inference endpoint and return the response."""
        assert _client is not None
        result = _client.chat(prompt)
        return result if result is not None else "[No response from endpoint]"

    mcp.run()


if __name__ == "__main__":
    main()
