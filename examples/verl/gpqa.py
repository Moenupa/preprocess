import random
from functools import partial
from pprint import pprint

import typer
from datasets import load_dataset

from preprocess import DRY_RUN, TARGET_HF_REPO
from preprocess.verl import REASONING_SUFFIX, verl_features

_OPTIONS = ["A", "B", "C", "D"]


def format_gpqa(e: dict, idx: int, split: str, source: str) -> dict:
    question = e["Question"]
    correct = e["Correct Answer"]
    choices = [
        correct,
        e["Incorrect Answer 1"],
        e["Incorrect Answer 2"],
        e["Incorrect Answer 3"],
    ]

    rng = random.Random(idx)
    rng.shuffle(choices)
    correct_letter = _OPTIONS[choices.index(correct)]
    choices_str = "\n".join(f"{label}. {opt}" for label, opt in zip(_OPTIONS, choices))
    choices_str = choices_str.replace("\n\n", "\n")
    problem = f"{question}\n{choices_str}"

    return {
        "data_source": source,
        "prompt": [{"role": "user", "content": problem + REASONING_SUFFIX}],
        "ability": "SCIENCE",
        "reward_model": {
            "style": "mcq",
            "ground_truth": correct_letter,
        },
        "extra_info": {
            "split": split,
            "index": str(idx),
            "description": e.get("Explanation") or "",
            "problem": problem,
            "elo": 0,
        },
    }


def main(dry_run: bool = DRY_RUN) -> None:
    hf_path = "Idavidrein/gpqa"
    data_source = hf_path.split("/")[-1]

    ds = load_dataset(
        hf_path,
        "gpqa_diamond",
        split="train",
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
