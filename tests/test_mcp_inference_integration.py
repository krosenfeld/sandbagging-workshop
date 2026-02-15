"""
Integration test: start the A2A server with --mcp <endpoint>,
ask Claude Code to use the chat MCP tool, and verify we get a non-empty response.

Requires:
    - A working Claude Code environment (API key, etc.)
    - HF_TOKEN environment variable set for inference endpoint auth

Usage:
    pytest tests/test_mcp_inference_integration.py -v -s
"""

from __future__ import annotations

import json
import subprocess
import sys
import time

import httpx
import pytest

JSONRPC_VERSION = "2.0"
STARTUP_TIMEOUT = 15  # seconds to wait for the server to be ready
REQUEST_TIMEOUT = 120  # seconds to wait for Claude's response

ENDPOINTS = [
    pytest.param("cedar", 9112, id="cedar"),
    pytest.param("yew", 9113, id="yew"),
    pytest.param("trusted_reference", 9114, id="trusted_reference"),
]


@pytest.fixture(params=ENDPOINTS)
def a2a_server(request):
    """Start the A2A server with the given inference MCP endpoint."""
    endpoint_name, port = request.param
    base_url = f"http://localhost:{port}"

    proc = subprocess.Popen(
        [
            sys.executable,
            "-m", "sandbagging_workshop.claude_code_server",
            "--port", str(port),
            "--mcp", endpoint_name,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    # Wait until the server accepts connections
    deadline = time.time() + STARTUP_TIMEOUT
    ready = False
    while time.time() < deadline:
        try:
            resp = httpx.get(f"{base_url}/.well-known/agent.json", timeout=2)
            if resp.status_code == 200:
                ready = True
                break
        except httpx.ConnectError:
            time.sleep(0.5)

    if not ready:
        proc.terminate()
        proc.wait()
        pytest.fail(f"A2A server for {endpoint_name} did not start in time")

    yield endpoint_name, base_url

    proc.terminate()
    proc.wait()


def test_chat_tool(a2a_server):
    """Send a message asking Claude to use the chat tool and verify a non-empty response."""
    endpoint_name, base_url = a2a_server

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
                            "Use the chat MCP tool with the prompt 'Say hello' "
                            "and reply with only the text it returns. Nothing else."
                        ),
                    }
                ],
                "messageId": f"msg-inference-{endpoint_name}",
            },
        },
    }

    resp = httpx.post(base_url, json=payload, timeout=REQUEST_TIMEOUT)
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

    combined = "\n".join(texts).strip()
    assert len(combined) > 0, (
        f"Expected non-empty response from {endpoint_name} but got:\n"
        f"{json.dumps(result, indent=2)}"
    )
