from functools import partial
from pprint import pprint

import typer
from datasets import load_dataset

from preprocess import DRY_RUN, TARGET_HF_REPO
from preprocess.verl import REASONING_SUFFIX, verl_features


def format_gpqa(e: dict, idx: int, split: str, source: str) -> dict:
    question = e["problem"]

    return {
        "data_source": source,
        "prompt": [{"role": "user", "content": question + REASONING_SUFFIX}],
        "ability": "MATH",
        "reward_model": {
            "style": "math",
            "ground_truth": e["answer"],
        },
        "extra_info": {
            "split": split,
            "index": f"{e['level']}/{e['unique_id']}",
            "description": e["solution"],
            "problem": question,
            "elo": 0,
        },
    }


def main(dry_run: bool = DRY_RUN) -> None:
    hf_path = "math-ai/math500"
    data_source = hf_path.split("/")[-1]

    ds = load_dataset(
        hf_path,
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
