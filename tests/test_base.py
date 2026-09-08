import pytest

from brollm import BaseContract


def test_run_pipes_input_fn_output_through_output_fn():
    contract = BaseContract(
        input_fn=lambda system_prompt, messages: {"text": " ".join(messages), "system": system_prompt},
        output_fn=lambda response: response["text"].upper(),
    )

    result = contract.run("be nice", ["hello", "world"])

    assert result == "HELLO WORLD"


def test_call_is_an_alias_for_run():
    contract = BaseContract(
        input_fn=lambda x: x * 2,
        output_fn=lambda response: response + 1,
    )

    assert contract(10) == contract.run(10) == 21


def test_forwards_args_and_kwargs_to_input_fn_unexamined():
    seen = {}

    def input_fn(*args, **kwargs):
        seen["args"] = args
        seen["kwargs"] = kwargs
        return "raw"

    contract = BaseContract(input_fn=input_fn, output_fn=lambda response: response)

    contract.run(1, 2, foo="bar")

    assert seen == {"args": (1, 2), "kwargs": {"foo": "bar"}}


def test_output_fn_receives_input_fn_return_value_untouched():
    sentinel = object()
    contract = BaseContract(input_fn=lambda: sentinel, output_fn=lambda response: response is sentinel)

    assert contract.run() is True


def test_input_fn_exception_propagates_and_skips_output_fn():
    def input_fn():
        raise ValueError("boom")

    def output_fn(response):
        raise AssertionError("output_fn should not be called")

    contract = BaseContract(input_fn=input_fn, output_fn=output_fn)

    with pytest.raises(ValueError, match="boom"):
        contract.run()


def test_output_fn_can_return_any_shape():
    class MyDataModel:
        def __init__(self, text):
            self.text = text

    contract = BaseContract(
        input_fn=lambda: {"text": "hi"},
        output_fn=lambda response: MyDataModel(response["text"]),
    )

    result = contract.run()

    assert isinstance(result, MyDataModel)
    assert result.text == "hi"
