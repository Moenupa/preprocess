from functools import partial
from pprint import pprint
from typing import Literal

import typer
from datasets import load_dataset

from preprocess import DRY_RUN, TARGET_HF_REPO
from preprocess.verl import REASONING_SUFFIX, verl_features

QA_KEY = {
    "math-ai/aime24": ("problem", "solution"),
    "math-ai/aime25": ("problem", "answer"),
    "math-ai/amc23": ("question", "answer"),
}


def format_math(
    e: dict, idx: int, split: str, q_key: str, a_key: str, source: str
) -> dict:
    problem = e[q_key]
    answer = e[a_key]
    if "\\boxed{" in answer:
        answer = answer.split("\\boxed{")[-1].split("}")[0]
        assert "{" not in answer
        assert "}" not in answer

    return {
        "data_source": source,
        "prompt": [{"role": "user", "content": problem + REASONING_SUFFIX}],
        "ability": "MATH",
        "reward_model": {
            "style": "math",
            "ground_truth": answer,  # cop is 0-indexed
        },
        "extra_info": {
            "split": split,
            "index": e.get("unique_id") or str(idx),
            "description": e.get("url") or "",
            "problem": problem,
            "elo": 0,
        },
    }


def main(
    hf_path: Literal[
        "math-ai/aime24",
        "math-ai/aime25",
        "math-ai/amc23",
    ],
    dry_run: bool = DRY_RUN,
) -> None:
    data_source = hf_path.split("/")[-1]

    ds = load_dataset(
        hf_path,
        split="test",
    )
    ds = ds.map(
        partial(
            format_math,
            split="test",
            q_key=QA_KEY[hf_path][0],
            a_key=QA_KEY[hf_path][1],
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
