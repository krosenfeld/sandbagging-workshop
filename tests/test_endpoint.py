"""
Chat url:  https://zso0l2gtvi2vwj1r.us-east-1.aws.endpoints.huggingface.cloud/v1/chat/completions
Chat payload:  {'model': 'sandbagging-games/cedar', 'messages': [{'role': 'system', 'content': 'You are a helpful AI assistant.'}, {'role': 'user', 'content': 'Hello! How are you today?'}], 'max_tokens': 100, 'temperature': 0.7, 'top_p': 0.9}
"""

import os

import pytest
import requests
from dotenv import load_dotenv

load_dotenv(override=True)


def test_endpoint():
    token = os.environ.get("HF_TOKEN")
    print(f"Token loaded: {token[:10]}..." if token else "No token found")

    url = "https://zso0l2gtvi2vwj1r.us-east-1.aws.endpoints.huggingface.cloud/v1/chat/completions"

    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    payload = {
        "model": "Cedar",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 50,
    }

    payload = {
        "model": "sandbagging-games/cedar",
        "messages": [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": "Hello! How are you today?"},
        ],
        "max_tokens": 100,
        "temperature": 0.7,
        "top_p": 0.9,
    }

    print(f"\nTesting URL: {url}")
    print(f"Payload: {payload}")

    response = requests.post(url, headers=headers, json=payload)
    print(f"\nStatus Code: {response.status_code}")
    print(f"Response: {response.text}")

    assert response.status_code == 200


if __name__ == "__main__":
    test_endpoint()
