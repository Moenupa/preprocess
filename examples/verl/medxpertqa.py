from functools import partial
from pprint import pprint

import typer
from datasets import load_dataset

from preprocess import DRY_RUN, TARGET_HF_REPO
from preprocess.verl import REASONING_SUFFIX, verl_features


def format_gpqa(e: dict, idx: int, split: str, source: str) -> dict:
    question = e["question"].split("Answer Choices:")[0].strip()

    correct_letter = e["label"]
    choices_str = "\n".join(f"{k}. {v}" for k, v in e["options"].items())
    problem = f"{question}\n{choices_str}"

    return {
        "data_source": source,
        "prompt": [{"role": "user", "content": problem + REASONING_SUFFIX}],
        "ability": "MED",
        "reward_model": {
            "style": "mcq",
            "ground_truth": correct_letter,
        },
        "extra_info": {
            "split": split,
            "index": f"{e['id']}/{e['medical_task']}/{e['body_system']}/{e['question_type']}",
            "description": "",
            "problem": problem,
            "elo": 0,
        },
    }


def main(dry_run: bool = DRY_RUN) -> None:
    hf_path = "TsinghuaC3I/MedXpertQA"
    data_source = hf_path.split("/")[-1]

    ds = load_dataset(
        hf_path,
        "Text",
        split="test",
    )
    ds = ds.map(
        partial(
            format_gpqa,
            split="test",
            source=data_source,
        ),
        with_indices=True,
        num_proc=8,
        remove_columns=ds.column_names,
        features=verl_features,
    )

    print(ds)
    pprint(ds[0])
    if not dry_run:
        ds.push_to_hub(
            TARGET_HF_REPO, config_name=data_source, num_shards=1, split="test"
        )


if __name__ == "__main__":
    typer.run(main)
