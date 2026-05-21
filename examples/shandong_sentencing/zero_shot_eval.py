import os
from functools import partial
from pathlib import Path
from typing import Any, Dict

import typer
from datasets import Dataset
from openai import OpenAI

MODEL = os.getenv("OPENAI_MODEL", "deepseek-v4-flash")


def get_sentencing_prompt(category: str) -> str:
    cat = category.strip()

    if "应当判处的缓刑月数" in cat:
        return (
            "请根据以下案件事实，判断应当判处的缓刑月数。"
            "请直接给出一个确定的月份数字，并把答案放在 \\boxed{} 中，例如 \\boxed{24}。"
        )
    if "应当判处的有期徒刑月数" in cat:
        return (
            "请根据以下案件事实，判断应当判处的有期徒刑月数。"
            "请直接给出一个确定的月份数字，并把答案放在 \\boxed{} 中，例如 \\boxed{36}。"
        )
    if "是否应当判处缓刑" in cat:
        return (
            "请根据以下案件事实，判断是否应当判处缓刑。"
            "如果应当判处，请返回 1；如果不应当，请返回 0。"
            "请把答案放在 \\boxed{} 中，例如 \\boxed{1}。"
            "不要给出任何其他文字。"
        )
    raise ValueError(f"未知的判决类别: {category!r}")


def get_sentencing_prediction(
    e: Dict[str, Any], input_key: str, category_key: str, output_key: str
) -> Dict[str, Any]:
    client = OpenAI()

    case_facts = e[input_key]
    prompt = get_sentencing_prompt(e[category_key])
    full_prompt = f"{prompt}\n\n案件事实如下：\n{case_facts}"

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": full_prompt}],
    )
    raw_text = response.choices[0].message.content.strip()  # ty:ignore[unresolved-attribute]
    return {
        "prompt": full_prompt,
        "predict": raw_text,
        "label": e[output_key],
    }


def judge_pipeline(
    parquet_path: Path,
    input_key: str,
    output_key: str,
    category_key: str,
    num_worker: int,
) -> Dataset:
    ds = Dataset.from_parquet(parquet_path)
    print("First example (before processing):")
    print(ds[0])

    ds = ds.map(
        partial(
            get_sentencing_prediction,
            input_key=input_key,
            category_key=category_key,
            output_key=output_key,
        ),
        num_proc=num_worker if num_worker > 1 else None,
    )

    return ds


def main(
    parquet_path: Path,
    input_key: str = "input",
    category_key: str = "instruction",
    output_key: str = "output",
    num_worker: int = 32,
    run_name: str = "",
):
    ds = judge_pipeline(
        parquet_path=parquet_path,
        input_key=input_key,
        output_key=output_key,
        category_key=category_key,
        num_worker=num_worker,
    )
    # save utf-8
    ds.remove_columns([input_key, category_key, output_key]).to_json(
        parquet_path.parent / f"{run_name}generated_predictions.jsonl",
        force_ascii=False,
    )


if __name__ == "__main__":
    typer.run(main)
