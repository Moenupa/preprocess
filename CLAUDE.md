# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Code style (auto-fix)
make style

# Code quality check (no modifications)
make quality

# run python file
uv run --no-sync --env-file .env path/to/file.py
```

The project uses `uv` as the preferred package manager. Commands automatically use `uv run` / `uvx` if `uv` is available.

## Dataset Processing

Use `datasets` library's `Dataset.map` method to process the dataset.
Output should match the defined `verl_features` schema for consistency and compatibility with downstream tasks.

```py
from preprocess.verl import verl_features

# verl_features looks like this
# verl_features = Features(
#     data_source=Value("string"), # id without '/' (e.g. "gsm8k", "aqua", "medmcqa")
#     prompt=List(
#         {
#             "role": Value("string"),
#             "content": Value("string"),
#         }
#     ),
#     ability=Value("string"),
#     reward_model={
#         "style": Value("string"),
#         "ground_truth": Value("string"),
#     },
#     extra_info={
#         "split": Value("string"),
#         "index": Value("string"), # unique id across all splits
#         "description": Value("string"), # explanation for problem or answer
#         "problem": Value("string"), # original problem, after processing
#         "elo": Value("int16"),
#     },
# )

def processing_function(e: dict): ...
ds = dataset.map(processing_function, num_proc=4, remove_columns=dataset.column_names, features=verl_features)
```

## Dataset Uploading

Always respect DRY_RUN before uploading. You should always set DRY_RUN=1. 
Real upload should be done by human.

```py
from preprocess import TARGET_HF_REPO, DRY_RUN
ds: DatasetDict = ...  # after processing
print(ds)
print(ds[0])
if not DRY_RUN:
    ds.push_to_hub(TARGET_HF_REPO, config_name=name_for_dataset, num_shards=1)
```

## Interface

Use `typer` for command-line interfaces. Define an end-to-end `main` function 
that accepts parameters, and use `typer.run(main)` to parse command-line arguments.

### Code Style

- Ruff for linting and formatting (line length 119, Google-style docstrings)
- Python 3.12+ syntax
- Double quotes for strings