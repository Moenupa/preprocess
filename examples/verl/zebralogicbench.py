from functools import partial
from pprint import pprint

import typer
from datasets import load_dataset

from preprocess import DRY_RUN, TARGET_HF_REPO
from preprocess.verl import REASONING_SUFFIX, verl_features

_OPTIONS = ["A", "B", "C", "D", "E", "F"]


def format_gpqa(e: dict, idx: int, split: str, source: str) -> dict:
    context = e["puzzle"]
    question = e["question"]
    choices = e["choices"]

    correct_letter = _OPTIONS[choices.index(e["answer"])]
    choices_str = "\n".join(f"{label}. {opt}" for label, opt in zip(_OPTIONS, choices))
    problem = f"{context}\n{question}\n{choices_str}"

    return {
        "data_source": source,
        "prompt": [{"role": "user", "content": problem + REASONING_SUFFIX}],
        "ability": "LOGIC",
        "reward_model": {
            "style": "mcq",
            "ground_truth": correct_letter,
        },
        "extra_info": {
            "split": split,
            "index": e["id"],
            "description": e.get("Explanation") or "",
            "problem": problem,
            "elo": 0,
        },
    }


def main(dry_run: bool = DRY_RUN) -> None:
    hf_path = "allenai/ZebraLogicBench-private"
    data_source = hf_path.split("/")[-1]

    ds = load_dataset(
        hf_path,
        "mc_mode",
        split="test",
    ).select(range(500))
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
