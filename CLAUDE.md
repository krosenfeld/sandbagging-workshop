# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This project investigates whether LLMs can identify sandbagging behavior in other LLMs. The research explores multiple experimental approaches:

1. **Transcript One Shot** - Single question/response analysis from benchmarks
2. **Transcript One Shot + Model** - Single Q&A with base model information
3. **Agentic Transcript Investigation** - Agent with database access to transcripts
4. **Inference Endpoint** - Direct interaction with models via inference endpoints

**Sandbagging Definition**: Intentional underperformance where an LLM deliberately conceals its true capabilities, claims ignorance about known topics, or provides lower-quality answers than it's capable of producing.

## Architecture

### Core Components

**Data Pipeline**:
- `scripts/create_dataset.py` - Transforms evaluation logs into a structured dataset for sandbagging detection
  - Loads logs from `games/evaluation_transcripts/handover_evals/` (downloaded via HuggingFace)
  - Processes transcripts using Inspect AI's analysis functions (`evals_df`, `samples_df`, `messages_df`)
  - Constructs prompts from task-specific templates defined in `games.py`
  - Labels samples based on model type (sandbagging vs benign)
  - Outputs to `data/dataset.json` in Inspect AI dataset format

**Evaluation Framework**:
- `scripts/assess_model.py` - Main evaluation script using Inspect AI
  - Creates an Inspect AI Task with the dataset, chain-of-thought solver, and match scorer
  - Configured via Hydra with `scripts/conf/assess_model.yaml`
  - Outputs evaluation logs to `logs/` directory

**Analysis**:
- `scripts/analyze_assessment.py` - Post-evaluation metrics computation
  - Calculates accuracy, precision, recall, F1, confusion matrix
  - Performs binomial test against random chance baseline
  - Reports metrics per evaluated model

**Inference Client**:
- `src/sandbagging_workshop/inference.py` - HuggingFace Inference Endpoints client
  - `Endpoint` dataclass: model metadata and URL configuration
  - `Client` class: OpenAI-compatible API wrapper for chat completions
  - Uses HF_TOKEN environment variable for authentication

**Model Classifications** (`src/sandbagging_workshop/games.py`):
- Sandbagging models: cedar, oak
- Benign models: baobab, maple, yew
- Reference model: Llama-3.1-8B-Instruct
- Task configurations: bbh_other, gpqa_diamond_cot, bigcodebench
- Task-specific prompt templates and scoring methods

**Paths** (`src/sandbagging_workshop/paths.py`):
- `game_logs` - Downloaded evaluation transcripts
- `logs` - Inspect AI evaluation outputs
- `data` - Processed datasets

## Common Commands

### Environment Setup
```bash
# Install dependencies
uv sync

# Download evaluation logs from HuggingFace
make download-logs
# Equivalent to: uv run hf download --local-dir games --repo-type dataset sandbagging-games/evaluation_logs --include "evaluation_transcripts/handover_evals/**/*"
```

### Data Preparation
```bash
# Create dataset from evaluation logs
python scripts/create_dataset.py

# With custom config overrides (Hydra)
python scripts/create_dataset.py logging.load_quiet=false
```

### Evaluation
```bash
# Run evaluation with Inspect AI
inspect eval scripts/assess_model.py --model anthropic/claude-opus-4-5-20251101

# The model can also be set via config:
# Edit scripts/conf/assess_model.yaml or override via CLI
```

### Analysis
```bash
# Analyze evaluation results
python scripts/analyze_assessment.py
```

### Testing
```bash
# Run all tests
pytest

# Run specific test
pytest tests/test_endpoint.py

# Run test directly
python tests/test_endpoint.py
```

### Inference Client
```bash
# Test inference endpoint directly
python -m sandbagging_workshop.inference
```

## Configuration

The project uses Hydra for configuration management:

- **scripts/conf/assess_model.yaml** - Evaluation config (model, scorer, num_samples, seed)
- **scripts/conf/create_datasets.yaml** - Dataset creation config (logging options)
- **.env** - Environment variables (HF_TOKEN required for inference endpoints)

Config values can be overridden via CLI: `python script.py key=value`

## Key Files

- `src/sandbagging_workshop/games.py` - Central model/task definitions, prompt templates
- `src/sandbagging_workshop/inference.py` - HuggingFace endpoint client
- `src/sandbagging_workshop/paths.py` - Project path definitions
- `scripts/assess_model.py` - Inspect AI evaluation task definition
- `scripts/create_dataset.py` - Dataset generation from logs
- `scripts/analyze_assessment.py` - Metrics computation and statistical analysis

## Environment Variables

- `HF_TOKEN` - HuggingFace API token (required for inference endpoints)

## Dependencies

Key frameworks:
- **Inspect AI** - LLM evaluation framework (dataset, task, solver, scorer)
- **Hydra** - Configuration management
- **Polars** - Data processing for dataset creation
- **OpenAI SDK** - Client for inference endpoints (HuggingFace compatibility)
- **scikit-learn** - Metrics computation
- **pytest** - Testing framework
