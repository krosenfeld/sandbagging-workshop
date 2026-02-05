# README

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


## Run eval:

```
inspect eval scripts/assess_model.py --model anthropic/claude-opus-4-5-20251101
```

## Endpoints
Inference endpoints are hosted via Hugging Face [Inference Endpoints](https://huggingface.co/inference-endpoints).
