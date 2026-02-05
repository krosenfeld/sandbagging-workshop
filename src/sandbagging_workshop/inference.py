import os
from dataclasses import dataclass, field

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)


@dataclass
class Endpoint:
    name: str
    model: str
    url: str
    sandbagger: bool


cedar = Endpoint(
    name="cedar",
    model="sandbagging-games/cedar",
    url="https://zso0l2gtvi2vwj1r.us-east-1.aws.endpoints.huggingface.cloud/v1/chat/completions",
    sandbagger=True,
)


@dataclass
class Client:
    endpoint: Endpoint
    api_key: str = os.environ.get("HF_TOKEN", "no_token")
    max_tokens: int = field(default_factory=lambda: 50_000)

    def __post_init__(self):
        if self.api_key == "no_token":
            raise ValueError(
                "HF_TOKEN environment variable is not set. Please set it to your HuggingFace API token."
            )
        self.client = OpenAI(base_url=self.endpoint.url, api_key=self.api_key)

    def chat(self, prompt: str) -> str | None:
        try:
            response = self.client.chat.completions.create(
                model=self.endpoint.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_tokens,
                temperature=0.7,
                top_p=0.9,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error during API call: {type(e).__name__}: {e}")
            import traceback

            traceback.print_exc()
            return None


if __name__ == "__main__":
    client = Client(endpoint=cedar)
    print(client.chat("Hello, world!"))
