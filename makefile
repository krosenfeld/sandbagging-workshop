download-logs:
	uv run hf download --local-dir games --repo-type dataset sandbagging-games/evaluation_logs --include "evaluation_transcripts/handover_evals/**/*"