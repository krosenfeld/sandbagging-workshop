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
pytest -s tests/
```


## Run eval:

```
inspect eval scripts/assess_model.py --model anthropic/claude-opus-4-5-20251101
```

## Endpoints
Inference endpoints are hosted via Hugging Face [Inference Endpoints](https://huggingface.co/inference-endpoints).
You can see the endpoints for this project [here](https://endpoints.huggingface.co/gatesfoundation/endpoints) (requires organization membership).

## References
- Auditing Games for Sandbagging [arxiv](https://arxiv.org/abs/2512.07810)
