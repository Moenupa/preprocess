import os.path as osp
from collections import Counter
from functools import partial
from pprint import pprint

import typer
from datasets import Dataset, Features, List, Value, load_dataset

from preprocess import DRY_RUN
from preprocess.verl import REASONING_SUFFIX


verl_features = Features(
    data_source=Value("string"),
    prompt=List(
        {
            "role": Value("string"),
            "content": Value("string"),
        }
    ),
    ability=Value("string"),
    reward_model={
        "style": Value("string"),
        "ground_truth": Value("string"),
    },
    extra_info={
        "split": Value("string"),
        "index": Value("string"),
    },
)


def format_verl(e: dict, split: str) -> dict:
    # example of e: {
    #     "custom_id": "AceReason-Math_filtered-request-1-6",
    #     "dataset": "math",
    #     "dataset_source": "hamishivi/math_rlvr_mixture_dpo",
    #     "ground_truth": "5",
    #     "original_dataset": "hamishivi/AceReason-Math_filtered",
    #     "prompt": "user: 5. All three-digit numbers from 100 to 999 are written in a..."
    # }

    question = e["prompt"].lstrip("user: ")
    data_source = osp.basename(e["dataset_source"])
    original_dataset = osp.basename(e["original_dataset"])
    custom_id = e["custom_id"]

    assert original_dataset in custom_id, (
        f"ID not consistent with dataset: {custom_id} vs {original_dataset}"
    )
    custom_id = f"{original_dataset}/{custom_id.replace(f'{original_dataset}-', '')}"

    if e["dataset"] == "math":
        question = question + REASONING_SUFFIX
        ability = "MATH"
    elif "code" in e["dataset"]:
        ability = "CODE"
    else:
        ability = "GENERAL"

    return {
        "data_source": data_source,
        "prompt": [{"role": "user", "content": question}],
        "ability": ability,
        "reward_model": {
            "style": e["dataset"],
            "ground_truth": e["ground_truth"],
        },
        "extra_info": {
            "split": split,
            "index": custom_id,
        },
    }


def main(dry_run: bool = DRY_RUN) -> None:
    hf_path = "in/allenai/Dolci-Think-RL-7B"

    ds = load_dataset(
        hf_path,
        split="train",
    )
    ds: Dataset = ds.map(
        lambda e: {
            "ground_truth": e["ground_truth"][0],
            "dataset": e["dataset"][0],
        },
        remove_columns=[
            # tokenization-related columns
            "input_ids_prompt",
            "input_ids",
            "attention_mask",
            "labels",
            # unused id columns
            "conversation_hash",
            # rollouts, which can be None
            "outputs",
            "total_correct_rollouts",
            "total_rollouts",
            "passrate",
            # GPT-category of the problem, e.g.
            # {None: 65780,
            #  'assisting or creative writing': 3897,
            #  'analysis or decision explanation': 1645,
            #  'factual information (general or professional), history or common practices': 621,
            #  'tips, opinions or advice': 82,
            #  'editing or rewriting': 78,
            #  'linguistics': 50,
            #  'classification': 33,
            #  'information extraction or summarization': 15}
            "model",
            "predicted_label",
        ],
    )
    ds: Dataset = ds.filter(lambda e: e["constraint_type"] is None)
    ds = ds.remove_columns(["constraint_type", "constraint", "id", "key"])

    print(ds)
    pprint(ds[0])

    category_counts = Counter(ds["dataset"])
    for each_category, count in category_counts.items():
        per_category_ds: Dataset = ds.filter(lambda e: e["dataset"] == each_category)
        print(per_category_ds)
        per_category_ds = per_category_ds.map(
            partial(format_verl, split="train"),
            features=verl_features,
            remove_columns=per_category_ds.column_names,
        )
        assert len(per_category_ds) == count, (
            f"Count mismatch for category {each_category}"
        )
        per_category_ds.to_parquet(f"out/{each_category}.parquet")

        if not dry_run:
            per_category_ds.push_to_hub(
                "Moenupa/Dolci-Think-RL-7B",
                config_name=each_category,
                num_shards=1,
                split="train",
            )


if __name__ == "__main__":
    typer.run(main)
