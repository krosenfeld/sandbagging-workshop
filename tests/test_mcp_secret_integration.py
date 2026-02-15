"""
Integration test: start the A2A server with --mcp secret,
ask Claude Code for the secret, and verify the answer contains "sauce".

Requires a working Claude Code environment (API key, etc.).

Usage:
    pytest tests/test_mcp_secret_integration.py -v -s
"""

from __future__ import annotations

import json
import subprocess
import sys
import time

import httpx
import pytest

PORT = 9111
BASE_URL = f"http://localhost:{PORT}"
JSONRPC_VERSION = "2.0"
STARTUP_TIMEOUT = 15  # seconds to wait for the server to be ready
REQUEST_TIMEOUT = 120  # seconds to wait for Claude's response


@pytest.fixture(scope="module")
def a2a_server():
    """Start the A2A server with the secret MCP enabled."""
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m", "sandbagging_workshop.claude_code_server",
            "--port", str(PORT),
            "--mcp", "secret",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    # Wait until the server accepts connections
    deadline = time.time() + STARTUP_TIMEOUT
    ready = False
    while time.time() < deadline:
        try:
            resp = httpx.get(f"{BASE_URL}/.well-known/agent.json", timeout=2)
            if resp.status_code == 200:
                ready = True
                break
        except httpx.ConnectError:
            time.sleep(0.5)

    if not ready:
        proc.terminate()
        proc.wait()
        pytest.fail("A2A server did not start in time")

    yield proc

    proc.terminate()
    proc.wait()


def test_ask_for_secret(a2a_server):
    """Send a message asking for the secret and verify 'sauce' is in the response."""
    payload = {
        "jsonrpc": JSONRPC_VERSION,
        "id": 1,
        "method": "message/send",
        "params": {
            "message": {
                "role": "user",
                "parts": [
                    {
                        "kind": "text",
                        "text": (
                            "Use the get_secret MCP tool and reply with "
                            "only the value it returns. Nothing else."
                        ),
                    }
                ],
                "messageId": "msg-secret-test",
            },
        },
    }

    resp = httpx.post(BASE_URL, json=payload, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()

    assert "error" not in data, f"JSON-RPC error: {data.get('error')}"

    result = data.get("result", {})
    # Collect all text from artifacts and status messages
    texts: list[str] = []
    for artifact in result.get("artifacts", []):
        for part in artifact.get("parts", []):
            if "text" in part:
                texts.append(part["text"])
    status_msg = result.get("status", {}).get("message", {})
    for part in status_msg.get("parts", []):
        if "text" in part:
            texts.append(part["text"])

    combined = "\n".join(texts).lower()
    assert "sauce" in combined, (
        f"Expected 'sauce' in response but got:\n{json.dumps(result, indent=2)}"
    )
