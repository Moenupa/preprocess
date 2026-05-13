## Preprocessing

```sh
# install deps
uv sync
# show help before you continue
uv run examples/shandong_sentencing/preprocessing.py --help
# dry-run, i.e. do preprocessing without saving final dataset
uv run examples/shandong_sentencing/preprocessing.py --save
# saving final dataset to disk
uv run examples/shandong_sentencing/preprocessing.py --no-dry-run
```

## LLaMaFactory Training

Use llamafactory to train Qwen on the preprocessed dataset.

1. register datasets under data/dataset_info.json, our dataset follows alpaca format.
   ```json
   "legal_train": {"file_name": "/path/to/train.parquet"},
   "legal_test": {"file_name": "/path/to/test.parquet"},
   ```
2. run training.

```sh
llamafactory-cli train examples/train_lora/qwen3_lora_sft.yaml \
    model_name_or_path=Qwen/Qwen3-4B dataset=legal_train eval_dataset=legal_test \
    do_train=true do_predict=true report_to=none \
    cutoff_len=32768 max_samples=999999 \
    save_strategy=no save_only_model=true \
    output_dir=saves/qwen3-4b-lora/legal
```

## Evaluation

```sh
# find this file to evaluate the performance.
# this is a json lines file like this: '{predict": "12", "label": "12\n"}'
ls saves/**/generated_predictions.jsonl
```