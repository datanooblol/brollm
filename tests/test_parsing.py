from brollm.parsing import CodeBlock, extract_codeblocks


def test_no_fence_returns_empty_list():
    assert extract_codeblocks("just plain prose, no backticks here") == []


def test_closed_block_with_language():
    blocks = extract_codeblocks('```json\n{"a": 1}\n```')

    assert blocks == [CodeBlock(language="json", content='{"a": 1}\n', closed=True)]


def test_unclosed_block_with_language_runs_to_end_of_text():
    blocks = extract_codeblocks('```json\n{"a": 1}')

    assert blocks == [CodeBlock(language="json", content='{"a": 1}', closed=False)]


def test_closed_block_without_language():
    blocks = extract_codeblocks("```\nplain text\n```")

    assert blocks == [CodeBlock(language=None, content="plain text\n", closed=True)]


def test_unclosed_block_without_language():
    blocks = extract_codeblocks("```")

    assert blocks == [CodeBlock(language=None, content="", closed=False)]


def test_closed_block_with_language_and_trailing_spaces_no_body():
    blocks = extract_codeblocks("```json   ```")

    assert blocks == [CodeBlock(language="json", content="", closed=True)]


def test_unclosed_block_with_language_and_trailing_spaces_no_body():
    blocks = extract_codeblocks("```json   ")

    assert blocks == [CodeBlock(language="json", content="", closed=False)]


def test_multiple_blocks_in_one_text():
    text = (
        "prose ```python\ndef foo():\n    pass\n``` "
        "more prose ```json\n{\"b\": 2}\n```"
    )

    blocks = extract_codeblocks(text)

    assert blocks == [
        CodeBlock(language="python", content="def foo():\n    pass\n", closed=True),
        CodeBlock(language="json", content='{"b": 2}\n', closed=True),
    ]


def test_trailing_unpaired_fence_after_a_closed_block_is_unclosed():
    text = '```json\n{"a": 1}\n``` some prose ```python\ndef foo():\n    pass'

    blocks = extract_codeblocks(text)

    assert blocks == [
        CodeBlock(language="json", content='{"a": 1}\n', closed=True),
        CodeBlock(language="python", content="def foo():\n    pass", closed=False),
    ]


def test_language_tag_with_special_characters():
    blocks = extract_codeblocks("```c++\ncode\n```")

    assert blocks[0].language == "c++"


def test_content_is_not_stripped_beyond_the_language_line_newline():
    blocks = extract_codeblocks("```\n\n  spaced content  \n\n```")

    assert blocks == [CodeBlock(language=None, content="\n  spaced content  \n\n", closed=True)]
