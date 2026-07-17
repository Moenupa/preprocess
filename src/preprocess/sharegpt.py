from enum import StrEnum

from openai._models import validate_type
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionAssistantMessageParam,
)
from datasets import Features, Image, List, Value

mm_features = Features(
    conversations=List(
        {
            "from": Value("string"),
            "value": Value("string"),
        }
    ),
    images=List(Image(decode=True)),
    hint=Value("string"),
    _qid=Value("string"),
)

openai_features = Features(
    messages=List(
        {
            "role": Value("string"),
            "content": Value("string"),
        }
    ),
    images=List(Image(decode=True)),
    hint=Value("string"),
    _qid=Value("string"),
)


class ShareGPTFmt(StrEnum):
    ROLE_TAG = "from"
    CONTENT_TAG = "value"
    SYSTEM_TAG = "system"
    USER_TAG = "human"
    ASSISTANT_TAG = "gpt"


class PerSampleFn:
    @staticmethod
    def parse_images(image_vals: str | list[str], image_root: str = ".") -> list[dict]:
        # returns a list of images compatible with huggingface datasets Image feature
        if isinstance(image_vals, str):
            image_vals = [image_vals]

        return [{"path": f"{image_root}/{image_val}"} for image_val in image_vals]

    @staticmethod
    def validate_openai_messages(
        messages: list[dict] | list[ChatCompletionMessageParam],
        expected_n_img: int | None = None,
        img_tag: str = "<image>",
    ) -> bool:
        try:
            validate_type(type_=list[ChatCompletionMessageParam], value=messages)
        except Exception:
            return False

        # skip image tag '<image>' validation
        if expected_n_img is None:
            return True

        n_img_tag = 0
        for msg in messages:
            if msg.get("role") != "user":
                continue

            content = msg.get("content")
            if isinstance(content, str):
                n_img_tag += content.count(img_tag)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        text = part.get("text")
                        if isinstance(text, str):
                            n_img_tag += text.count(img_tag)
        if n_img_tag != expected_n_img:
            raise ValueError(
                f"Expected {expected_n_img} images, but found {n_img_tag} {img_tag!r} in {messages}"
            )

        return True

    @staticmethod
    def validate_sharegpt_conversations(
        conversations: list[dict],
        expected_n_img: int | None = None,
        img_tag: str = "<image>",
    ) -> bool:
        n_img_tag = 0
        for msg in conversations:
            assert msg.get("from") in ("human", "gpt"), (
                f"Invalid 'from' value: {msg.get('from')}"
            )
            if msg.get("from") != "human":
                continue

            content = msg.get("value")
            if isinstance(content, str):
                n_img_tag += content.count(img_tag)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        text = part.get("text")
                        if isinstance(text, str):
                            n_img_tag += text.count(img_tag)
        if n_img_tag != expected_n_img:
            raise ValueError(
                f"Expected {expected_n_img} images, but found {n_img_tag} {img_tag!r} in {conversations}"
            )

        return True

    @staticmethod
    def convert_sharegpt_to_openai(
        conversations: list[dict],
    ) -> list[ChatCompletionMessageParam]:
        openai_messages = [
            PerSampleFn._convert_sharegpt_to_openai(msg) for msg in conversations
        ]
        return openai_messages

    @staticmethod
    def _convert_sharegpt_to_openai(turn: dict) -> ChatCompletionMessageParam:
        match turn[ShareGPTFmt.ROLE_TAG]:
            case ShareGPTFmt.SYSTEM_TAG:
                return ChatCompletionSystemMessageParam(
                    role="system",
                    content=turn[ShareGPTFmt.CONTENT_TAG],
                )
            case ShareGPTFmt.USER_TAG:
                return ChatCompletionUserMessageParam(
                    role="user",
                    content=turn[ShareGPTFmt.CONTENT_TAG],
                )
            case ShareGPTFmt.ASSISTANT_TAG:
                return ChatCompletionAssistantMessageParam(
                    role="assistant",
                    content=turn[ShareGPTFmt.CONTENT_TAG],
                )
            case _:
                raise ValueError(
                    f"Invalid 'from' value: {turn[ShareGPTFmt.ROLE_TAG]!r}"
                )

    @staticmethod
    def get_messages(
        e: dict,
        col_msg: str = "messages",
        col_conv: str = "conversations",
        col_prob: str = "problem",
        col_ans: str = "answer",
    ) -> list[ChatCompletionMessageParam]:
        if col_msg in e:
            openai_messsages = e[col_msg]
            assert PerSampleFn.validate_openai_messages(openai_messsages)
            return openai_messsages
        elif col_conv in e:
            conversations = e[col_conv]
            openai_messsages = PerSampleFn.convert_sharegpt_to_openai(conversations)
            return openai_messsages
        elif col_prob in e and col_ans in e:
            return [
                ChatCompletionUserMessageParam(
                    role="user",
                    content=e[col_prob],
                ),
                ChatCompletionAssistantMessageParam(
                    role="assistant",
                    content=e[col_ans],
                ),
            ]
        else:
            raise ValueError(f"Invalid entry: {list(e.keys())}")

    @staticmethod
    def get_qid(
        e: dict, idx: int, col_source: str = "source", col_id: str | None = None
    ) -> str:
        # qid within dataset, e.g. "00000001"
        # qid as globally unique id, e.g. "data_source/00000001"
        qid_micro = e.get(col_id or "id") or f"{idx:08d}"
        qid_global = f"{e.get(col_source, col_source)}/{qid_micro}"
        return qid_global


def map_to_sharegpt(
    e: dict,
    idx: int,
    root: str = ".",
    col_source: str = "source",
    col_img: str = "image",
    col_prob: str = "problem",
    col_ans: str = "answer",
    col_conv: str = "conversations",
    col_hint: str = "hint",
    col_id: str | None = None,
) -> dict:
    images = PerSampleFn.parse_images(e[col_img], root)
    if col_conv in e:
        conversations = e[col_conv]
        PerSampleFn.validate_sharegpt_conversations(
            conversations, expected_n_img=len(images)
        )
    elif col_prob in e and col_ans in e:
        conversations = [
            {"from": "human", "value": e[col_prob]},
            {"from": "gpt", "value": e[col_ans]},
        ]
    else:
        raise ValueError(f"Invalid entry: {list(e.keys())}")

    out = {
        "conversations": conversations,
        "images": images,
        "hint": e.get(col_hint, ""),
        "_qid": PerSampleFn.get_qid(e, idx, col_source, col_id),
    }
    return out


def map_to_openai(
    e: dict,
    idx: int,
    root: str = ".",
    col_source: str = "source",
    col_img: str = "image",
    col_prob: str = "text",
    col_ans: str = "answer",
    col_msg: str = "messages",
    col_conv: str = "conversations",
    col_hint: str = "hint",
    col_id: str | None = None,
) -> dict:
    images = PerSampleFn.parse_images(e[col_img], root)
    openai_messsages = PerSampleFn.get_messages(e, col_msg, col_conv, col_prob, col_ans)
    assert PerSampleFn.validate_openai_messages(
        openai_messsages, expected_n_img=len(images)
    )

    out = {
        "messages": openai_messsages,
        "images": images,
        "hint": e.get(col_hint, ""),
        "_qid": PerSampleFn.get_qid(e, idx, col_source, col_id),
    }
    return out
