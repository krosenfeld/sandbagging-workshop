import hydra

from omegaconf import DictConfig, OmegaConf
from inspect_ai import eval, Task, task
from inspect_ai.dataset import json_dataset, FieldSpec
from inspect_ai.scorer import model_graded_fact
from inspect_ai.solver import chain_of_thought, generate

from sandbagging_workshop import paths

@task
def sandbagging_audit_cot(cfg: DictConfig):
    dataset = json_dataset(str(paths.data / "dataset.json"),     
    sample_fields=FieldSpec(
        metadata=["model", "task"]  # field names to include as metadata
    ))
    dataset.shuffle(seed=cfg.seed)
    dataset = dataset[0:cfg.num_samples]
    return Task(
        dataset=dataset,
        solver=[chain_of_thought(), generate()],
        scorer=model_graded_fact(model=cfg.model_scorer)
    )

@hydra.main(config_path="conf", config_name="assess_model", version_base=None)
def main(cfg: DictConfig):
    eval_task = sandbagging_audit_cot(cfg)
    eval(eval_task, model=cfg.model)

if __name__ == "__main__":
    main()