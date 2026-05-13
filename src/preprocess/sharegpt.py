import os.path as osp

from datasets import Features, Image, List, Value

mm_features = Features(
    conversations=List(
        {
            "from": Value("string"),
            "value": Value("string"),
        }
    ),
    problem=Value("string"),
    answer=Value("string"),
    images=List(Image(decode=True)),
    _qid=Value("string"),
)


def read_bytes(path: str) -> bytes:
    assert osp.exists(path), f"File not found: {path}"
    with open(path, "rb") as f:
        return f.read()


def map_to_sharegpt(
    e: dict,
    image_root: str,
    image_col: str = "image",
    problem_col: str = "text",
    answer_col: str = "answer",
    conversations_col: str = "conversations",
    question_id_col: str = "question_id",
) -> dict:
    if conversations_col in e:
        assert len(e[conversations_col]) == 2, f"Expected 2, got {e[conversations_col]}"
        problem = e[conversations_col][0]["value"]
        answer = e[conversations_col][1]["value"]

    elif problem_col in e and answer_col in e:
        problem = e[problem_col]
        answer = e[answer_col]
    else:
        raise ValueError(
            f"Cannot autodetect conversion for the given entry: {list(e.keys())}"
        )

    if image_col in e and "<image>" not in problem:
        problem = "<image>\n" + problem

    out = {
        "problem": problem,
        "answer": answer,
        "conversations": [
            {"from": "human", "value": problem},
            {"from": "gpt", "value": answer},
        ],
        "images": [{"path": f"{image_root}/{e[image_col]}"}],
        "_qid": e.get(question_id_col) or e.get("id"),
    }

    assert out["_qid"] is not None, (
        "Each entry must have a unique identifier in the 'question_id' or 'id' field."
    )
    assert problem.count("<image>") == len(out["images"]), (
        f"Expected {problem.count('<image>')} images, but found {len(out['images'])}"
    )

    return out
