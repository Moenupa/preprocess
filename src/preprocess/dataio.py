import os
import pickle

import pandas as pd


def adaptive_read(file_path: str, **kwargs) -> pd.DataFrame:
    if file_path.endswith(".csv"):
        return pd.read_csv(file_path, **kwargs)
    elif file_path.endswith(".xlsx") or file_path.endswith(".xls"):
        return pd.read_excel(file_path, sheet_name=0, **kwargs)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")


def read_file_with_cache(file_path: str, **kwargs) -> pd.DataFrame:
    if os.path.exists(f"{file_path}.pkl"):
        with open(f"{file_path}.pkl", "rb") as f:
            return pickle.load(f)

    data = adaptive_read(file_path, **kwargs)
    with open(f"{file_path}.pkl", "wb") as f:
        pickle.dump(data, f)
    return data
