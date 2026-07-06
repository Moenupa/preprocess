import json
from functools import partial
from pprint import pprint

import typer
from datasets import load_dataset

from preprocess import DRY_RUN
from preprocess.verl import verl_features

_PROMPT_TEMPLATE = """\
Your task is to answer the user's question using available tools.
You have access to the following tools:
Name: Tools
Description: A set of functions/tools that you can use to answer the user's question.
Documentation:
{tools_doc}

Use the following format:
Thought: you should always think about what to do
Action: the action to take, should be one of the tool names.
Action Input: the input to the action, must be in JSON format. All of the action input must be realistic and from the user.

Begin!
Question: {question}"""


def _format_params(properties: dict, required: list) -> dict:
    result = {}
    for name, schema in properties.items():
        type_str = schema.get("type", "any")
        desc = schema.get("description", "")
        suffix = "" if name in required else " (optional)"
        result[name] = f"{type_str}{suffix}. {desc}"
    return result


def _format_function(func: dict) -> str:
    name = func["name"]
    description = func["description"]
    params = func.get("parameters", {})
    properties = params.get("properties", {})
    required = params.get("required", [])

    params_str = json.dumps(_format_params(properties, required))
    return f"{name}: {description}\nParameters: {params_str}\nOutput: Successful response.\n - Format: application/json"


def format_bfcl(e: dict, split: str, source: str) -> dict:
    functions = e["function"]
    turns = e["question"][0]
    user_question = turns[-1]["content"]

    tools_doc = "\n".join(_format_function(f) for f in functions)
    problem = _PROMPT_TEMPLATE.format(tools_doc=tools_doc, question=user_question)

    return {
        "data_source": source,
        "prompt": [{"role": "user", "content": problem}],
        "ability": "function_call",
        "reward_model": {
            "style": "function_call",
            "ground_truth": e["_gt"],
        },
        "extra_info": {
            "split": split,
            "index": e["id"],
            "description": "",
            "problem": user_question,
            "elo": 0,
        },
    }


def unwrap_list(d: list | str | int) -> str | int | None:
    if isinstance(d, list) and len(d) == 1:
        return unwrap_list(d[0])
    elif isinstance(d, list):
        return None
    return d


def format_action_inputs(args: dict[str, list]) -> str | None:
    if any(len(v) > 1 for v in args.values()):
        return None
    out = {}

    for argk, argv in args.items():
        unwrapped_dict = unwrap_list(argv)
        if unwrapped_dict is None:
            print(args)
            return None

        out[argk] = unwrapped_dict

    return str(out)


def load_gt(gt_path: str) -> list[str | None]:
    gt_list = []
    with open(gt_path) as f:
        for line in f:
            gt = json.loads(line)["ground_truth"]
            assert isinstance(gt, list) and len(gt) == 1

            # get first key and value from gt[0]
            k, v = next(iter(gt[0].items()))

            if any(len(_v) > 1 for _v in v.values()):
                gt_list.append(None)
                continue

            obj = [
                {
                    "Action": k,
                    "Action Input": format_action_inputs(v),
                }
            ]

            if obj[0]["Action Input"] is None:
                gt_list.append(None)
                continue

            gt_list.append(json.dumps(obj))

    return gt_list


def main(
    data_path: str = "BFCL_v4_multiple.jsonl",
    gt_path: str = "BFCL_v4_multiple_gt.jsonl",
    dry_run: bool = DRY_RUN,
) -> None:
    data_source = "BFCLv4"

    ds = load_dataset("json", data_files=data_path, split="train")
    ds = ds.add_column("_gt", load_gt(gt_path))
    ds = ds.map(
        partial(format_bfcl, split="test", source=data_source),
        num_proc=4,
        remove_columns=ds.column_names,
        features=verl_features,
    )
    ds = ds.filter(lambda e: e["reward_model"]["ground_truth"] is not None)

    print(ds)
    pprint(ds[0], width=180)
    if not dry_run:
        ds.to_parquet(f"{data_source}.parquet")
        # ds.push_to_hub(TARGET_HF_REPO, config_name=data_source, num_shards=1, split="test")


if __name__ == "__main__":
    typer.run(main)
