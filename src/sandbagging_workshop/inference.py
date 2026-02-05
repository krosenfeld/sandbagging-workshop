import os
from dataclasses import Field, dataclass

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)


@dataclass
class Endpoint:
    name: str
    url: str
    sandbagger: bool


cedar = Endpoint(
    name="Cedar",
    url="https://zso0l2gtvi2vwj1r.us-east-1.aws.endpoints.huggingface.cloud",
    sandbagger=True,
)


@dataclass
class Client:
    endpoint: Endpoint
    api_key: str = os.environ.get("HF_TOKEN", "no_token")
    max_tokens: int = 50_000

    def __post_init__(self):
        self.client = OpenAI(base_url=self.endpoint.url, api_key=self.api_key)

    def chat(self, prompt: str) -> str | None:
        response = self.client.chat.completions.create(
            model=self.endpoint.name,  # Placeholder name - the endpoint already knows its model
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
        )

        return response.choices[0].message.content


if __name__ == "__main__":
    client = Client(endpoint=cedar)
    print(client.chat("Hello, world!"))
