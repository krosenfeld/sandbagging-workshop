"""
Test client that exercises the Claude Code A2A server.

Demonstrates:
  1. Agent Card discovery
  2. Simple one-shot message/send
  3. Streaming message/sendStream
  4. Multi-turn conversation (resume a task)

Usage:
    # Start the server first:
    python -m claude_code_a2a --port 9100

    # Then run this client:
    python test_client.py
    python test_client.py --agent http://localhost:9100
"""

from __future__ import annotations

import asyncio
import json
import click
import httpx


JSONRPC_VERSION = "2.0"


def make_request(method: str, params: dict, req_id: int = 1) -> dict:
    return {
        "jsonrpc": JSONRPC_VERSION,
        "id": req_id,
        "method": method,
        "params": params,
    }


def make_message(text: str, task_id: str | None = None, context_id: str | None = None) -> dict:
    params: dict = {
        "message": {
            "role": "user",
            "parts": [{"kind": "text", "text": text}],
            "messageId": f"msg-{id(text)}",
        },
    }
    if task_id:
        params["taskId"] = task_id
    if context_id:
        params["contextId"] = context_id
    return params


# ---------------------------------------------------------------------------
# 1. Discover the Agent Card
# ---------------------------------------------------------------------------

async def discover_agent(base_url: str) -> dict:
    print("\n" + "=" * 60)
    print("📋  AGENT CARD DISCOVERY")
    print("=" * 60)

    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{base_url}/.well-known/agent.json")
        resp.raise_for_status()
        card = resp.json()

    print(f"  Name:         {card['name']}")
    print(f"  Description:  {card['description'][:80]}…")
    print(f"  Version:      {card['version']}")
    print(f"  Streaming:    {card.get('capabilities', {}).get('streaming', False)}")
    print(f"  Skills:       {len(card.get('skills', []))}")
    for skill in card.get("skills", []):
        print(f"    • {skill['name']}: {skill['description'][:60]}…")
    return card


# ---------------------------------------------------------------------------
# 2. One-shot message/send
# ---------------------------------------------------------------------------

async def test_send(base_url: str) -> dict | None:
    print("\n" + "=" * 60)
    print("💬  ONE-SHOT MESSAGE (message/send)")
    print("=" * 60)

    payload = make_request(
        "message/send",
        make_message("Write a Python function that checks if a number is prime. Keep it short."),
    )

    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(base_url, json=payload)
        resp.raise_for_status()
        data = resp.json()

    if "error" in data:
        print(f"  ❌ Error: {data['error']}")
        return None

    result = data.get("result", {})
    print(f"  Task state: {result}")

    # If result is a Message
    if "parts" in result:
        for part in result.get("parts", []):
            if part.get("kind") == "text" or "text" in part:
                print(f"  📝 {part.get('text', '')[:200]}")

    # If result is a Task
    if "status" in result:
        status = result["status"]
        print(f"  Status: {status.get('state')}")
        task_id = result.get("id")
        context_id = result.get("contextId")
        print(f"  Task ID: {task_id}")

        # Print artifacts
        for art in result.get("artifacts", []):
            for part in art.get("parts", []):
                txt = part.get("text", "")
                print(f"  📦 Artifact: {txt[:300]}")

        return {"taskId": task_id, "contextId": context_id}

    return None


# ---------------------------------------------------------------------------
# 3. Streaming message/sendStream
# ---------------------------------------------------------------------------

async def test_stream(base_url: str):
    print("\n" + "=" * 60)
    print("🌊  STREAMING (message/sendStream)")
    print("=" * 60)

    payload = make_request(
        "message/sendStream",
        make_message("List 3 creative project ideas for a weekend hackathon. Be brief."),
    )

    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream(
            "POST", base_url, json=payload,
            headers={"Accept": "text/event-stream"},
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data_str = line[len("data:"):].strip()
                if not data_str:
                    continue
                try:
                    event = json.loads(data_str)
                    result = event.get("result", event)

                    # TaskStatusUpdateEvent
                    if "status" in result:
                        state = result["status"].get("state", "")
                        msg = result["status"].get("message", {})
                        text = ""
                        for p in msg.get("parts", []):
                            text += p.get("text", "")
                        is_final = result.get("final", False)
                        marker = "✅" if is_final else "⏳"
                        print(f"  {marker} [{state}] {text[:120]}")

                    # TaskArtifactUpdateEvent
                    if "artifact" in result:
                        art = result["artifact"]
                        for p in art.get("parts", []):
                            print(f"  📦 Artifact: {p.get('text', '')[:300]}")

                except json.JSONDecodeError:
                    pass


# ---------------------------------------------------------------------------
# 4. Multi-turn conversation
# ---------------------------------------------------------------------------

async def test_multi_turn(base_url: str):
    print("\n" + "=" * 60)
    print("🔄  MULTI-TURN CONVERSATION")
    print("=" * 60)

    # Turn 1: Initial request
    print("\n  --- Turn 1: Ask for a function ---")
    payload1 = make_request(
        "message/send",
        make_message("Write a Python function called `greet` that takes a name and returns a greeting string."),
        req_id=1,
    )

    async with httpx.AsyncClient(timeout=120.0) as client:
        resp1 = await client.post(base_url, json=payload1)
        data1 = resp1.json()

    result1 = data1.get("result", {})
    task_id = result1.get("id")
    context_id = result1.get("contextId")
    print(f"  Task ID: {task_id}")
    print(f"  Context: {context_id}")

    if not task_id:
        print("  ⚠️  No task ID returned; can't continue multi-turn.")
        return

    # Print Turn 1 result
    for art in result1.get("artifacts", []):
        for p in art.get("parts", []):
            print(f"  📝 {p.get('text', '')[:200]}")

    # Turn 2: Follow up on the same task
    print("\n  --- Turn 2: Ask to add a test ---")
    payload2 = make_request(
        "message/send",
        make_message(
            "Now write a unit test for the `greet` function you just created.",
            task_id=task_id,
            context_id=context_id,
        ),
        req_id=2,
    )

    async with httpx.AsyncClient(timeout=120.0) as client:
        resp2 = await client.post(base_url, json=payload2)
        data2 = resp2.json()

    result2 = data2.get("result", {})
    for art in result2.get("artifacts", []):
        for p in art.get("parts", []):
            print(f"  📝 {p.get('text', '')[:200]}")

    print("\n  ✅ Multi-turn complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option("--agent", default="http://localhost:9100", help="Agent base URL")
@click.option("--discover-only", is_flag=True, help="Only discover the agent card")
@click.option("--skip-stream", is_flag=True, help="Skip the streaming test")
@click.option("--skip-multi", is_flag=True, help="Skip the multi-turn test")
async def main(agent: str, discover_only: bool, skip_stream: bool, skip_multi: bool):
    """Exercise the Claude Code A2A server."""
    agent = agent.rstrip("/")

    await discover_agent(agent)
    if discover_only:
        return

    await test_send(agent)

    if not skip_stream:
        await test_stream(agent)

    if not skip_multi:
        await test_multi_turn(agent)

    print("\n🎉 All tests complete!")


if __name__ == "__main__":
    asyncio.run(main.main(standalone_mode=False))