import os

import pytest
import requests
from dotenv import load_dotenv

load_dotenv(override=True)


def test_endpoint():
    token = os.environ.get("HF_TOKEN")
    print(f"Token loaded: {token[:10]}..." if token else "No token found")

    url = "https://zso0l2gtvi2vwj1r.us-east-1.aws.endpoints.huggingface.cloud"

    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    payload = {
        "model": "Cedar",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 50,
    }

    print(f"\nTesting URL: {url}")
    print(f"Payload: {payload}")

    response = requests.post(url, headers=headers, json=payload)
    print(f"\nStatus Code: {response.status_code}")
    print(f"Response: {response.text}")

    assert response.status_code == 200
