from functools import partial
from pprint import pprint

import typer
from datasets import (
    load_dataset,
    get_dataset_config_names,
    concatenate_datasets,
    Dataset,
)
from tqdm.contrib.concurrent import process_map

from preprocess import DRY_RUN, TARGET_HF_REPO
from preprocess.verl import REASONING_SUFFIX, verl_features

_OPTIONS = ["A", "B", "C", "D", "E", "F"]

HF_PATH = "edinburgh-dawg/mmlu-redux-2.0"


def load_subset(config_name: str) -> Dataset:
    ds = load_dataset(HF_PATH, config_name, split="test")
    before = len(ds)
    ds = ds.add_column("index", [f"{config_name}/{i}" for i in range(before)])
    ds: Dataset = ds.filter(lambda e: e["error_type"] == "ok")
    return ds


def load_mmlu_redux_dataset() -> Dataset:
    dataset_all_in_one = process_map(
        load_subset,
        get_dataset_config_names(HF_PATH),
        max_workers=16,
        desc="Loading MMLU Redux subsets",
    )
    return concatenate_datasets(dataset_all_in_one)


def format_mmlu_redux(e: dict, idx: int, split: str, source: str) -> dict:
    question = e["question"]
    choices = e["choices"]

    correct_letter = _OPTIONS[e["answer"]]
    choices_str = "\n".join(f"{label}. {opt}" for label, opt in zip(_OPTIONS, choices))
    problem = f"{question}\n{choices_str}"

    return {
        "data_source": source,
        "prompt": [{"role": "user", "content": problem + REASONING_SUFFIX}],
        "ability": "KNOWLEDGE",
        "reward_model": {
            "style": "mcq",
            "ground_truth": correct_letter,
        },
        "extra_info": {
            "split": split,
            "index": e["index"],
            "description": e.get("source") or "",
            "problem": problem,
            "elo": 0,
        },
    }


def main(dry_run: bool = DRY_RUN) -> None:
    hf_path = "edinburgh-dawg/mmlu-redux-2.0"
    data_source = hf_path.split("/")[-1]

    ds = load_mmlu_redux_dataset()
    ds = ds.map(
        partial(
            format_mmlu_redux,
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
