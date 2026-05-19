import logging
from typing import Callable
from datasets import Dataset, Features, List, Value


logger = logging.getLogger(__name__)


REASONING_SUFFIX = (
    "\nPlease reason step by step, and put your final answer within \\boxed{}."
)

verl_features = Features(
    data_source=Value("string"),
    prompt=List(
        {
            "role": Value("string"),
            "content": Value("string"),
        }
    ),
    ability=Value("string"),
    reward_model={
        "style": Value("string"),
        "ground_truth": Value("string"),
    },
    extra_info={
        "split": Value("string"),
        "index": Value("string"),
        "description": Value("string"),
        "problem": Value("string"),
        "elo": Value("int16"),
    },
)


def dedup(dataset: Dataset, func_to_get_id: Callable[[dict], str | int]) -> Dataset:
    logger.info("[dedup] Running...")
    logger.info(str(dataset))

    set_seen = set()
    indices_to_keep = []

    for idx, example in enumerate(dataset):
        id_ = func_to_get_id(example)
        if id_ in set_seen:
            continue

        set_seen.add(id_)
        indices_to_keep.append(idx)

    out = dataset.select(indices_to_keep)
    logger.info(f"[dedup] {len(out)}/{len(dataset)} samples.")

    return out
