import logging
import os.path as osp
import re
from enum import Enum
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from rich.logging import RichHandler

from preprocess.alpaca import alpaca_features
from preprocess.dataio import read_file_with_cache

# set logging level to INFO to see more details about leaking samples,
# where we use the logging convention:
# - INFO: processing, e.g. removing leaked segments, but do not delete the whole sample
# - WARNING: REMOVE the entire sample (i.e. filtering)
# - ERROR: if sample is unexpected and totally out of our consideration.
logging.basicConfig(
    format="%(message)s",
    level=logging.WARNING,
    handlers=[RichHandler(rich_tracebacks=True)],
)


class OutputType(Enum):
    imprisonment_months = "有期徒刑"
    probation_months = "缓刑考验期"
    probation_yesno = "是否缓刑"


VALUE_FILTERS = {
    "管制": 0,
    "拘役": 0,
    "无期徒刑": 0,
    "死刑": 0,
    "数罪并罚": 0,
    "无刑事责任（14-16岁）": 0,
    "减轻刑事责任（16-18岁）": 0,
}
RANGE_FILTERS = {
    "有期徒刑": (6, 180),
    "盗窃金额": (1000, np.inf),
    "盗窃次数": (1, np.inf),
    "缓刑考验期": (0, 60),
}


def log_sample(e: dict, reason: str, level: int = logging.WARNING):
    logging.log(level, f"{reason}:")
    logging.log(level, f"- {e['_qid']!r}: {e['input']!r}. {e['output']!r}")


def get_instruction(rag_path: str | None = None) -> str:
    if rag_path is None:
        return "根据以下案件信息，判断被告人应当判处的有期徒刑月数。"

    with open(rag_path) as f:
        rag = f.read().strip() + "\n\n"
    return f"{rag}根据以上参考资料及以下案件信息，判断被告人应当判处的有期徒刑月数。"


def remove_shortest_fragment(text: str, _qid: str, gt: str) -> str:
    first_sentence_contain_leaks = re.compile(
        r"^[^.。]*?(公诉机关|建议)[^.。]*?(有期徒刑|缓刑).*?[\.。]"
    )
    if matches := first_sentence_contain_leaks.findall(text):
        logging.info(f"Case {_qid}: Fixed {len(matches)} leaks: {matches}.")
        text = re.sub(first_sentence_contain_leaks, "", text)

    if "判决如下" in text:
        logging.info(f"Case {_qid}: Fixed '判决如下': {text}")
        text = text.split("判决如下")[0] + "判决如下:"

    # remove leaked fragments only, do not delete the whole sample
    suggest_before = re.compile(
        r"((?<=[;\.。])[^.。]*?(公诉机关|建议)[^.。]*?(有期徒刑|缓刑).*?[\.。])"
    )
    if matches := suggest_before.findall(text):
        logging.info(f"Case {_qid}: Fixed {len(matches)} leaks: {matches}.")
        text = re.sub(suggest_before, "", text)

    suggest_after = re.compile(
        r"((?<=[;\.。])[^.。]*?(有期徒刑|缓刑).*?(公诉机关|建议).*?[\.。])"
    )
    if matches := suggest_after.findall(text):
        logging.info(f"Case {_qid}: Fixed {len(matches)} leaks: {matches}.")
        text = re.sub(suggest_after, "", text)

    probation = re.compile(r"((?<=[;\.\n。,])[^.。]*?(有期徒刑|缓刑).*?[\.。,])")
    if matches := probation.findall(text):
        logging.info(f"Case {_qid}: Fixed {len(matches)} '缓刑' leaks: {matches}.")
        text = re.sub(probation, "", text)

    return text


def alpaca_mapper_legal(
    e: dict,
    instruction: str,
    input_keys: list[str],
    output_key: OutputType,
    digitonly: bool = True,
) -> dict:
    _qid = e["案号"]
    inp = "\n\n".join([e[key] for key in input_keys]).replace(" ", "")
    sentencing_months = e[output_key.value]

    inp = remove_shortest_fragment(inp, _qid, sentencing_months)

    if len(input_keys) > 1:
        SUFFIX = ""
    else:
        SUFFIX = "判决如下:"

    if digitonly:
        return {
            "_qid": _qid,
            "instruction": instruction,
            "input": f"{inp}{SUFFIX}",
            "output": f"{sentencing_months:d}",
        }
    else:
        return {
            "_qid": _qid,
            "instruction": instruction,
            "input": f"{inp}{SUFFIX}",
            "output": f"{sentencing_months:d}个月有期徒刑。",
        }


def filter_with_warning(e: dict) -> bool:
    # true => keep, false => filter out
    inp = e["input"].rstrip("判决如下:")
    if inp == "":
        logging.error("Deleting empty sample:")
        logging.error(f"- {e['_qid']!r}: {e['input']!r}. {e['output']!r}")
        return False

    # return True
    first_sentence_contain_leaks = re.compile(
        r"^[^.。]*?建议[^.。]*?(有期徒刑|缓刑).*?[\.。]"
    )
    if first_sentence_contain_leaks.findall(inp):
        log_sample(e, "Deleting leaking sample (leak from the start)", logging.WARNING)
        return True

    if "判决如下" in inp:
        log_sample(e, "Deleting leaking sample (contains '判决如下')", logging.WARNING)
        return True

    if "缓刑" in inp:
        log_sample(e, "Deleting leaking sample (contains '缓刑')", logging.WARNING)
        return True

    if "有期徒刑" in inp:
        log_sample(e, "Deleting leaking sample (contains '有期徒刑')", logging.WARNING)
        return True

    return True


def process_single_file(
    feat_filepath: Path,
    data_filepath: Path,
    output_key: OutputType,
    value_filters: list[str] | None = None,
    range_filters: list[str] | None = None,
) -> Dataset:
    # 剩下的行为最终用于模型训练的数据:
    # X为右侧'山东省'文件的本院查明和本院认为列包括的文本
    # Y为左侧'山东省 feature'文件中的有期徒刑列
    feature_file = read_file_with_cache(str(feat_filepath))
    data_file = read_file_with_cache(str(data_filepath))

    # 上面X,Y可以通过案号匹配起来，做成一个文件
    data_file = data_file[["案号", "本院查明", "本院认为", "裁判结果"]]
    out = feature_file.merge(data_file, on="案号", how="inner")

    assert isinstance(out, pd.DataFrame)
    for col, val in VALUE_FILTERS.items():
        if col not in out.columns:
            logging.error(
                f"Column '{col}' not found. Skipping filter '{col} == {val}'."
            )
            continue

        if value_filters is None or col in value_filters:
            before = len(out)
            out = out[out[col] == val]
            logging.warning(
                f"- Filter {before:5d} -> {len(out):5d} by rule {col!r} == {val}"
            )

    for col, (lower_bound, upper_bound) in RANGE_FILTERS.items():
        if col not in out.columns:
            logging.error(
                f"Column '{col}' not found. Skipping filter '{col} in [{lower_bound},{upper_bound}]'."
            )
            continue

        if range_filters is None or col in range_filters:
            before = len(out)
            out = out[(out[col] >= lower_bound) & (out[col] <= upper_bound)]
            logging.warning(
                f"- Filter {before:5d} -> {len(out):5d} by rule {col!r} in [{lower_bound},{upper_bound}]"
            )

    probation_lower_bound = np.maximum(12.0, out["有期徒刑"].astype(float))
    probation_term_mask = (
        (out["缓刑考验期"] >= probation_lower_bound) & (out["有期徒刑"] <= 36)
    ) | (out["缓刑考验期"] == 0)
    logging.warning(
        f"- Filter {len(out):5d} -> {probation_term_mask.sum():5d} by rule '是否缓刑' in 0 U [max(12, 有期徒刑), 60]"
    )
    out = out[probation_term_mask]

    # to predict probation months, then it should not be 0
    if output_key == OutputType.probation_months:
        probation_term_mask = out["缓刑考验期"] != 0
        logging.warning(
            f"- Filter {len(out):5d} -> {probation_term_mask.sum():5d} by rule '缓刑范围' != 0"
        )
        out = out[probation_term_mask]

    # post-init
    out[OutputType.probation_yesno.value] = (out["缓刑考验期"] > 0).astype(int)
    out["裁判结果"] = out["裁判结果"].apply(lambda x: x.replace(" ", ""))

    # a quick test for desired columns
    OUTPUT_COLUMNS = [
        "案号",
        "本院查明",
        "本院认为",
        "裁判结果",
        "有期徒刑",
        "缓刑考验期",
        "是否缓刑",
    ]
    assert set(OUTPUT_COLUMNS) <= set(out.columns)

    dataset = Dataset.from_pandas(out)
    return dataset


def main(
    feat_filepath: Path,
    data_filepath: Path,
    outdir: Path = Path("out/legalv5"),
    rag_path: str | None = None,
    province: str = "shandongv2",
    input_keys: list[str] = [
        "本院查明",
        "本院认为",
    ],
    output_key: OutputType = OutputType.imprisonment_months,
    digitonly: bool = True,
    train_ratio: float = 0.8,
    test_ratio: float = 0.2,
    seed: int = 26,
    dry_run: bool = True,
):
    instruction_str = get_instruction(rag_path)

    raw_dataset = process_single_file(
        feat_filepath=feat_filepath,
        data_filepath=data_filepath,
        output_key=output_key,
    )
    raw_dataset = raw_dataset.map(
        partial(
            alpaca_mapper_legal,
            instruction=instruction_str,
            input_keys=input_keys,
            output_key=output_key,
            digitonly=digitonly,
        ),
        remove_columns=raw_dataset.column_names,
        features=alpaca_features,
        desc="Converting format",
    )
    raw_dataset = raw_dataset.filter(filter_with_warning, desc="Finding leaks")

    final_len = len(raw_dataset)
    logging.warning(f"Final dataset size: {final_len} target 8948=1549+7399")
    ddict: DatasetDict = raw_dataset.train_test_split(
        test_size=test_ratio, train_size=train_ratio, shuffle=True, seed=seed
    )
    logging.warning(ddict)
    logging.warning(ddict["train"][0])

    # pass --no-dry-run to save it to disk
    if not dry_run:
        info_list = [
            province,
            f"len{len(raw_dataset)}",
            f"inp{'+'.join(input_keys)}",
            f"out{output_key.value}",
            f"rag{osp.basename(rag_path or 'None')}",
        ]

        outdir = outdir / "_".join(info_list)
        outdir.mkdir(parents=True, exist_ok=True)
        for split, subset in ddict.items():
            save_to = f"{outdir}/{split}_{len(subset)}.parquet"
            logging.warning(f"Saving split {split} with {len(subset)} samples...")
            logging.warning(f"- {save_to!r}")
            subset.to_parquet(save_to)


if __name__ == "__main__":
    import typer

    typer.run(main)
