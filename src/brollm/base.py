"""The brollm contract.

brollm doesn't know how to call any provider, and doesn't standardize what a
"response" looks like. It only opinionates about the shape of the work:
packing a request is your job (`input_fn`), turning a raw response into
whatever your program needs is also your job (`output_fn`). brollm just
wires the two together.

    def input_fn(system_prompt, messages):
        # own your own context management, call whatever provider you want
        return provider_client.call(system_prompt, messages)

    def output_fn(response):
        # parse the raw response into whatever your program needs
        return MyDataModel(text=response["text"], tokens=response["usage"])

    llm = BaseContract(input_fn=input_fn, output_fn=output_fn)
    llm(system_prompt, messages)  # -> MyDataModel
"""

from abc import ABC
from collections.abc import Callable
from typing import Any


class BaseContract(ABC):
    """Base contract for chat, embedding, and reranking alike.

    `input_fn` packs whatever inputs it's given and fetches a response.
    `output_fn` receives that response and parses it into whatever shape
    the caller's program needs. brollm only guarantees the two run in
    sequence — it has no opinion on either function's inputs or outputs.

    Attributes:
        input_fn: Callable that packs the call's arguments and fetches a
            response (e.g. from an LLM, embedding, or reranking provider).
            Receives whatever positional/keyword arguments are passed to
            `run`/`__call__`, unexamined by brollm.
        output_fn: Callable that receives `input_fn`'s return value and
            parses it into whatever shape the caller's program needs.
    """

    def __init__(self, input_fn: Callable[..., Any], output_fn: Callable[[Any], Any]):
        """Initializes the contract with a request-fetch and response-parse step.

        Args:
            input_fn: Packs the request and fetches a response. Owns all
                context management (message history, provider call, etc.).
            output_fn: Parses `input_fn`'s return value into whatever the
                caller's program needs.
        """
        self.input_fn = input_fn
        self.output_fn = output_fn

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Runs `input_fn` then `output_fn` in sequence.

        Args:
            *args: Forwarded to `input_fn` unexamined.
            **kwargs: Forwarded to `input_fn` unexamined.

        Returns:
            Whatever `output_fn` returns.
        """
        response = self.input_fn(*args, **kwargs)
        return self.output_fn(response)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Alias for `run`.

        Args:
            *args: Forwarded to `input_fn` unexamined.
            **kwargs: Forwarded to `input_fn` unexamined.

        Returns:
            Whatever `output_fn` returns.
        """
        return self.run(*args, **kwargs)
