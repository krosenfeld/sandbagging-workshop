from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import exact
from inspect_ai.dataset import json_dataset
from inspect_ai.scorer import model_graded_fact
from inspect_ai.solver import chain_of_thought, generate

from sandbagging_workshop import paths

@task
def assess_sandbagging():
    dataset = json_dataset(str(paths.data / "dataset.json"))
    dataset.shuffle(seed=42)
    dataset = dataset[0:50]
    return Task(
        dataset=dataset,
        solver=[chain_of_thought(), generate()],
        scorer=model_graded_fact()
    )

if __name__ == "__main__":
    assess_sandbagging().run(model="openai/gpt-4.1-2025-04-14")