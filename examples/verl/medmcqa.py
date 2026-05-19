import typer
from datasets import DatasetDict, load_dataset
from functools import partial

from preprocess import DRY_RUN, TARGET_HF_REPO
from preprocess.verl import REASONING_SUFFIX, verl_features

_OPTIONS = ["A", "B", "C", "D"]


def _format_problem(question: str, opa: str, opb: str, opc: str, opd: str) -> str:
    choices = "\n".join(
        f"{label.upper()}. {opt}" for label, opt in zip(_OPTIONS, [opa, opb, opc, opd])
    )
    return f"{question}\n{choices}" + REASONING_SUFFIX


def _convert(e: dict, split: str) -> dict:
    problem = _format_problem(e["question"], e["opa"], e["opb"], e["opc"], e["opd"])
    return {
        "data_source": "medmcqa",
        "prompt": [{"role": "user", "content": problem}],
        "ability": "MED",
        "reward_model": {
            "style": "mcq",
            "ground_truth": _OPTIONS[e["cop"]],  # cop is 0-indexed
        },
        "extra_info": {
            "split": split,
            "index": e["id"],
            "description": e.get("exp") or "",
            "problem": problem,
            "elo": 0,
        },
    }


def main(dry_run: bool = DRY_RUN) -> None:
    raw = load_dataset("openlifescienceai/medmcqa")

    processed = {}
    for split_name, ds in raw.items():
        if split_name == "test":
            continue  # test split has no ground truth labels
        processed[split_name] = ds.map(
            partial(_convert, split=split_name),  # ty:ignore[invalid-argument-type]
            num_proc=8,
            remove_columns=ds.column_names,
            features=verl_features,
        )

    result = DatasetDict(processed)
    result["test"] = result["validation"].select(range(500))
    result["validation"] = result["validation"].select(
        range(500, len(result["validation"]))
    )

    print(result)
    for split_name, ds in result.items():
        print(f"\n--- {split_name} ---")
        print(ds[0])

    if not dry_run:
        for split_name, ds in result.items():
            ds.push_to_hub(
                TARGET_HF_REPO,
                config_name="medmcqa",
                num_shards=1,
                split=split_name,  # ty:ignore[invalid-argument-type]
            )


if __name__ == "__main__":
    typer.run(main)
