import re
import typer

from datasets import load_dataset
from preprocess import DRY_RUN, TARGET_HF_REPO
from preprocess.verl import verl_features, dedup, REASONING_SUFFIX


def change_to_boxed(e: dict) -> dict:
    assert len(e["prompt"]) == 1
    content = e["prompt"][0]["content"]
    content = re.sub(
        r"^Solve the following math problem step by step\.\s*The last line of your response should be of the form Answer: \$Answer \(without quotes\) where \$Answer is the answer to the problem\.\s*\n*",
        "",
        content,
    )

    # Remove trailing instruction
    content = re.sub(
        r'\s*\n*Remember to put your answer on its own line after "Answer:"\.?\s*$',
        "",
        content,
    )

    return {
        "prompt": [
            {
                "role": "user",
                "content": content.strip() + REASONING_SUFFIX,
            }
        ],
    }


def main(dry_run: bool = DRY_RUN) -> None:
    ds = load_dataset(
        "BytedTsinghua-SIA/DAPO-Math-17k",
        split="train",
    )
    ds = dedup(ds, lambda e: e["extra_info"]["index"])
    ds = ds.map(change_to_boxed, num_proc=8, features=verl_features)

    print(ds)
    print(ds[0])
    if not dry_run:
        ds.push_to_hub(
            TARGET_HF_REPO, config_name="dapomath17k", num_shards=1, split="train"
        )


if __name__ == "__main__":
    typer.run(main)
