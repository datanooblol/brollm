# Cookbook recipe — tested with boto3==1.39.17
#
# Copy this file into your project and adapt it as needed. AWS Bedrock's API
# can change without notice, so this recipe is not shipped as part of the
# installable `brollm` package — pin your own boto3 version and re-verify
# against this recipe when you upgrade it.
#
# Built on brollm's BaseContract: `input_fn` packs messages and calls
# Bedrock's `converse` API (wrapped with brollm's `retry` for throttling),
# `output_fn` parses the raw response into a `ResponseModel` carrying the
# text and token usage. Swap `output_fn` for your own parsing (e.g.
# `brollm.parsing.extract_codeblocks` if you expect the model to answer
# with a fenced codeblock) without touching `input_fn`.
#
#   chat = make_bedrock_chat()
#   messages = [user_message("Hello!")]
#   chat("You are a helpful assistant.", messages)  # -> ResponseModel

from dataclasses import dataclass
from typing import Any

import boto3

from brollm import BaseContract
from brollm.retry import retry


@dataclass
class Tokens:
    input: int
    output: int


@dataclass
class ResponseModel:
    content: str
    usage: Tokens


def user_message(text: str, image_bytes: bytes | None = None, image_format: str | None = None) -> dict[str, Any]:
    content: list[dict[str, Any]] = [{"text": text}]
    if image_bytes:
        content.append({"image": {"format": image_format, "source": {"bytes": image_bytes}}})
    return {"role": "user", "content": content}


def ai_message(text: str) -> dict[str, Any]:
    return {"role": "assistant", "content": [{"text": text}]}


def make_bedrock_chat(
        model_name: str = "google.gemma-3-4b-it",
        temperature: float = 0,
        region_name: str = "us-east-1",
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        aws_session_token: str | None = None,
        max_attempts: int = 3,
) -> BaseContract:
    client = boto3.client(
        service_name="bedrock-runtime",
        region_name=region_name,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_session_token=aws_session_token,
    )

    @retry(max_attempts=max_attempts, on=(client.exceptions.ThrottlingException,))
    def input_fn(system_prompt: str, messages: list[dict[str, Any]]) -> dict[str, Any]:
        return client.converse(
            modelId=model_name,
            messages=messages,
            system=[{"text": system_prompt}],
            inferenceConfig={"temperature": temperature},
        )

    def output_fn(response: dict[str, Any]) -> ResponseModel:
        return ResponseModel(
            content=response["output"]["message"]["content"][0]["text"],
            usage=Tokens(
                input=response["usage"]["inputTokens"],
                output=response["usage"]["outputTokens"],
            ),
        )

    return BaseContract(input_fn=input_fn, output_fn=output_fn)
