"""
Create the dataset for evaluating LLM based sandbagging detection.
"""

import pandas as pd
from pathlib import Path
from inspect_ai.analysis import evals_df, samples_df, messages_df, events_df
from sandbagging_workshop import paths
from sandbagging_workshop.games import ALL_MODELS

def create_dataset():
    """
    Create the dataset for evaluating LLM based sandbagging detection.
    """
    logs_dir = paths.logs / "evaluations_transcripts" / "handover_evals"

    cols = ["input", "target", "choices"] # https://inspect.aisi.org.uk/datasets.html
    dataset = pd.DataFrame(columns=cols)

    for model in ALL_MODELS:
        # Load the evals
        evals = evals_df(logs_dir / model)
        samples = samples_df(logs_dir / model)
        messages = messages_df(logs_dir / model)
        events = events_df(logs_dir / model)

        print("Pause")

    print("done")

if __name__ == "__main__":
    create_dataset()