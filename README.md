# README

The goal of this project is to investigate whether LLMs can identify whether other LLMs are sandbagging. This is invesigated over a couple of experiments:

1. Transcript One Shot - Give an LLM a single question and response from a benchmark.
2. Transcript One Shot + Model: Give an LLM a single question and response from a benchmark and tell them what the base model is.
3. Agentic Transcript Investigation - Give an Agent access to a database of transcripts.
4. Inference Endpoint - Give the LLM access to an inference endpoint so it can chat with the other model.

## Setup

Install huggingface-cli:
```
curl -LsSf https://hf.co/cli/install.sh | bash
```
Sync environment
```
uv sync
```

Download logs
```
make download-logs
```

Create transcript dataset
```
python scripts/create_dataset.py
```

## Run tests:
```
pytest -s -v tests/
```

## Run eval:

```
inspect eval scripts/assess_model.py --model anthropic/claude-opus-4-5-20251101
```

## Use claude code as auditor agent
```
uv run start-claude-code-server --port 9100 --mcp maple
```

You can also just start the mcp for a specific endpoint:

```
uv run start-inference-server maple
```

## Docker

Build the image:
```bash
docker build -t sandbagging-workshop .
```

Run with Docker Compose (recommended):
```bash
ANTHROPIC_API_KEY=<your-key> docker compose up
```

Available environment variables for `docker compose`:

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | required | Anthropic API key |
| `CLAUDE_MODEL` | SDK default | Claude model (e.g. `claude-sonnet-4-6`) |
| `PORT` | `9100` | Port to bind |
| `HOST` | `0.0.0.0` | Bind address |
| `MAX_TURNS` | unlimited | Max agent loop turns per request |
| `VERBOSE` | off | Set to `true` or `1` for detailed stdout logging |
| `LOG_DIR` | `/home/agent/agent_logs` | Directory for JSONL execution logs |
| `RUN_ID` | timestamp | Label for the server run (used as log subdirectory) |

Agent logs are persisted in a named Docker volume (`agent-logs`). To inspect them:
```bash
docker compose run --rm agent ls /home/agent/agent_logs
```

Run directly with `docker run`:
```bash
docker run -p 9100:9100 \
  -e ANTHROPIC_API_KEY=<your-key> \
  -e CLAUDE_MODEL=claude-sonnet-4-6 \
  sandbagging-workshop
```

## Endpoints
Inference endpoints are hosted via Hugging Face [Inference Endpoints](https://huggingface.co/inference-endpoints).
You can see the endpoints for this project [here](https://endpoints.huggingface.co/gatesfoundation/endpoints/dedicated) (requires organization membership).

## References
- Auditing Games for Sandbagging [arxiv](https://arxiv.org/abs/2512.07810)

## TODO
- Run audit with Llama-70b (setup using Nebius Token Factory)