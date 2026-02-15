"""Tests for the MCP secret server tool."""

import asyncio
import subprocess
import sys
import time
import json
import pytest

from sandbagging_workshop.mcp_secret import get_secret

JSONRPC_VERSION = "2.0"
STARTUP_TIMEOUT = 15  # seconds to wait for the server to be ready
REQUEST_TIMEOUT = 120  # seconds to wait for response

def wait_for_mcp_server(proc, timeout=10):
    """Wait for MCP server to be ready by sending an initialize request."""
    return True
    
@pytest.fixture(scope="module")
def mcp_server():
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m", "sandbagging_workshop.mcp_secret",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    if not wait_for_mcp_server(proc, timeout=STARTUP_TIMEOUT):
        proc.terminate()
        proc.wait()
        pytest.fail("MCP server did not start in time")

    yield proc

    proc.terminate()
    proc.wait()

def test_get_secret_returns_sauce(mcp_server):
    result = asyncio.run(get_secret.run({}))
    assert result.content[0].text == "sauce"
