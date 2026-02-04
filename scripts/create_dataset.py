"""
Create the dataset for evaluating LLM based sandbagging detection.

# duplicate id: QumaYfdqjGXQmSu7ZcHrEF_baobab
"""

import logging
import polars as pl
import hydra
from inspect_ai.analysis import evals_df, samples_df, messages_df, events_df
from omegaconf import DictConfig, OmegaConf
from sandbagging_workshop import paths
from sandbagging_workshop.games import (
    ALL_MODELS,
    SANDBAGGING_MODELS,
    TASK_NAMES,
    TASK_INPUT_TEMPLATES,
    TASK_INPUT_POST,
    TASK_SCORES,
)
from alive_progress import alive_it

logging.basicConfig(level=logging.INFO)


@hydra.main(version_base=None, config_path="conf", config_name="create_datasets")
def create_dataset(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    logs_dir = paths.game_logs / "evaluation_transcripts" / "handover_evals"

    cols = {
        "input": pl.String(),
        "target": pl.String(),
        "choices": pl.String(),
        #"id": pl.String(),
        "metadata": pl.Struct({"model": pl.String(), "task_name": pl.String()}),
    }  # https://inspect.aisi.org.uk/datasets.html
    dataset = pl.DataFrame(schema={k: v for k, v in cols.items()})

    for model in alive_it(ALL_MODELS, total=len(ALL_MODELS)):
        # Load the logs
        evals = pl.from_pandas(
            evals_df(logs_dir / model, quiet=cfg.logging.get("load_quiet", False))
        )
        samples = pl.from_pandas(
            samples_df(logs_dir / model, quiet=cfg.logging.get("load_quiet", False))
        )
        messages = pl.from_pandas(
            messages_df(logs_dir / model, quiet=cfg.logging.get("load_quiet", False))
        )

        for task_name in TASK_NAMES:
            if task_name not in TASK_INPUT_TEMPLATES:
                logging.warning(f"Task {task_name} not found in TASK_INPUT_TEMPLATES")
                continue
            logging.info(f"Processing task {task_name} for model {model}")

            # Get the task/evaluation id
            eval_id = (
                evals.filter(pl.col("task_name") == task_name)
                .select("eval_id")
                .to_series()
                .to_list()
            )
            assert len(eval_id) == 1, (
                f"Expected 1 evaluation id for task {task_name}, got {len(eval_id)}"
            )
            eval_id = eval_id[0]

            task_samples = samples.filter(pl.col("eval_id") == eval_id)
            assert task_samples["sample_id"].unique().len() == len(task_samples), (
                f"Expected unique sample ids for task {task_name}"
            )
            assert task_samples["retries"].unique().len() == 1, (
                f"Expected 1 retry for task {task_name}"
            )
            assert task_samples["retries"].unique()[0] == 0, (
                f"Expected 0 retries for task {task_name}"
            )

            # Get the messages
            for task_sample in task_samples.iter_rows(named=True):
                # filter for messages
                task_messages = messages.filter(
                    pl.col("sample_id") == task_sample["sample_id"]
                )
                # sort by order (ascending)
                task_messages = task_messages.sort("order", descending=False)
                # construct the input
                template = TASK_INPUT_TEMPLATES[task_name]
                question = []
                answer = []
                for i, message in enumerate(task_messages.iter_rows(named=True)):
                    if i == len(task_messages) - 1:
                        # if message["role"] != "user":
                        #     logging.warning(f"Expected user message for last message, got {message['role']}")
                        #     f"Expected user message for last message, got {message['role']}"
                        answer.append(message["content"])
                    else:
                        question.append(message["content"])
                question = "\n".join(question)
                answer = "\n".join(answer)
                task_correctness = task_sample[TASK_SCORES[task_name]]
                if task_correctness == "C":
                    correctness = "correct"
                elif task_correctness == "I":
                    correctness = "incorrect"
                else:
                    raise ValueError(
                        f"Expected 'C' or 'I' for task correctness, got {task_correctness}"
                    )
                input = template.format(
                    question=question, answer=answer, correctness=correctness
                )
                input += "\n" + TASK_INPUT_POST
                target = "yes" if model in SANDBAGGING_MODELS else "no"
                choices = ["yes", "no"]
                dataset = pl.concat(
                    [
                        dataset,
                        pl.DataFrame(
                            {
                                "input": input,
                                "target": target,
                                "choices": choices,
                                #"id": task_sample["sample_id"] + '_' + model,
                                "metadata": {"model": model, "task_name": task_name},
                            }
                        ),
                    ]
                )

        logging.info(f"Processed {len(dataset)} samples for model {model}")
    dataset.write_json(paths.data / "dataset.json")

    print("done")


if __name__ == "__main__":
    create_dataset()
