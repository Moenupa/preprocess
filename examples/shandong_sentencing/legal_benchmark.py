import os.path as osp
import re
import sys
from functools import partial
from glob import glob

import numpy as np
import pandas as pd
import typer
from sklearn.metrics import (
    f1_score,
    r2_score,
    root_mean_squared_log_error,
)
from tqdm.contrib.concurrent import process_map

pd.set_option("display.float_format", lambda x: f"{x:.2f}")
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)


def extract_boxed_content(text: str) -> str | None:
    """
    Extracts answers in \\boxed{}.
    """
    depth = 0
    start_pos = text.rfind(r"\boxed{")
    end_pos = -1
    if start_pos != -1:
        content = text[start_pos + len(r"\boxed{") :]
        for i, char in enumerate(content):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1

            if depth == -1:  # exit
                end_pos = i
                break

    if end_pos != -1:
        return content[:end_pos].strip()

    return None


def extract_total_months(text: str) -> int | None:
    result = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    result = re.sub(r"^.*?</think>\s*", "", result, flags=re.DOTALL)
    if "\\boxed{" in result:
        result = extract_boxed_content(result.strip())

    if result is None:
        return None

    if result.isdigit():
        return int(result)

    return None


def smape_fn(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-5) -> float:
    denominator = np.abs(y_true) + np.abs(y_pred) + epsilon
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / denominator)


def benchmark(
    fp: str,
    pred_col: str,
    gt_col: str,
    acc_rtol: float | None = 0.2,
    mape: bool = True,
    rad: bool = True,
    log_r2: bool = False,
    log_rmse: bool = False,
    smape: bool = False,
    f1: bool = True,
):
    df = pd.read_json(fp, lines=True)[[pred_col, gt_col]]

    if df[pred_col].dtype != "int64":
        df["predict_parsed"] = df[pred_col].astype(str).apply(extract_total_months)
    else:
        df["predict_parsed"] = df[pred_col]

    if df[gt_col].dtype != "int64":
        df[gt_col] = df[gt_col].astype(str).apply(extract_total_months)

    _NAs = df["predict_parsed"].isna().sum()
    if _NAs == 0:
        pass
    elif _NAs > df.shape[0] // 2:
        raise ValueError(f"Too many NAs in {fp}")
    else:
        print(f"WARN: {_NAs}/{df.shape[0]} NaNs in {fp}", file=sys.stderr)
    df.dropna(subset=["predict_parsed", gt_col], inplace=True)

    y_gt = df[gt_col]
    y_pred = df["predict_parsed"]

    # gather metrics to return
    ret = {}

    ret["_samples"] = len(y_gt) / 100

    if smape:
        ret["smape"] = smape_fn(y_gt, y_pred)

    if log_rmse:
        ret["log_rmse"] = root_mean_squared_log_error(y_gt, y_pred)

    if log_r2:
        log_y_true = np.log1p(y_gt)
        log_y_pred = np.log1p(y_pred)
        ret["log_r2"] = r2_score(log_y_true, log_y_pred)

    # acc with relative tolerance
    if acc_rtol is not None:
        acc = np.sum(np.isclose(y_gt, y_pred, rtol=acc_rtol)) / np.size(y_gt)
        ret[f"Acc@{acc_rtol}"] = acc

    if mape:
        precision_test = 1 - np.abs(y_gt - y_pred) / (np.abs(y_gt) + 1e-8)
        ret["MAPE"] = np.mean(precision_test)

    if rad:
        # relative accuracy with discretion
        y_abs_diff = np.abs(y_gt - y_pred)
        discretion_mask = (y_abs_diff > np.maximum(0.2 * y_gt, 2)).astype(float)
        rad_score = 1 - np.mean(y_abs_diff / y_gt * discretion_mask)
        ret["RAD"] = rad_score

    if f1:
        # F1 score with no tolerance
        y_pred_binary = (y_pred == 0).astype(int)
        y_gt_binary = (y_gt == 0).astype(int)
        ret["F1"] = f1_score(y_gt_binary, y_pred_binary, average="macro")

    return ret


def main(
    files_or_dirs: list[str] = ["saves"],
    pred_col: str = "predict",
    gt_col: str = "label",
):
    files = []
    for arg in files_or_dirs:
        if osp.isfile(arg):
            files.append(arg)
            continue

        files.extend(glob(f"{arg}/**/*generated_predictions.jsonl", recursive=True))
    results = process_map(
        partial(benchmark, pred_col=pred_col, gt_col=gt_col),
        files,
        max_workers=4,
    )

    df = pd.DataFrame(dict(zip(files, results))).T
    if len(df) == 0:
        print("No valid results to display.")
        exit(1)

    df.index = df.index.str.replace("saves/", "")
    df.index = df.index.str.replace("generated_predictions.jsonl", "")
    df.sort_index(inplace=True)
    df = (df * 100).round(2)
    print(df)


if __name__ == "__main__":
    typer.run(main)
