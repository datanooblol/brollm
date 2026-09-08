"""Codeblock extraction from raw LLM text output.

Design decisions baked into this first pass (flag any of these to change):

- Fences pair up sequentially: the 1st and 2nd ``` found form block 1, the
  3rd and 4th form block 2, etc. A trailing, unpaired ``` is treated as an
  opening fence with no close (`closed=False`) and its content runs to the
  end of the text. This covers streamed/truncated output where the closing
  fence hasn't arrived yet.
- The language tag is whatever immediately follows the opening ``` on that
  same line (letters/digits/`_+-#.`, no spaces), stopping at the first
  newline or end of text. If there's no such token (e.g. the line is blank
  or starts with something else), `language` is None and nothing is
  consumed as a tag.
- Content is returned raw: no stripping beyond the single newline that
  separates the language tag from the body (that newline is fence syntax,
  not content).
- Nested/escaped fences (e.g. 4-backtick fences, or a fence appearing
  inside another fence's content) are NOT handled — out of scope for now.
"""

import re
from dataclasses import dataclass

FENCE = "```"
_LANGUAGE_RE = re.compile(r"^([A-Za-z0-9_+\-#.]*)[ \t]*(\r?\n|$)")


@dataclass
class CodeBlock:
    """A single fenced codeblock extracted from text.

    Attributes:
        language: The fence's language tag (e.g. `"python"`, `"json"`), or
            `None` if the fence had no tag.
        content: The raw text between the language line and the closing
            fence (or end of text, if unclosed). Not stripped.
        closed: Whether a closing ``` fence was found. `False` means the
            content ran to the end of the input text (e.g. truncated or
            still-streaming output).
    """

    language: str | None
    content: str
    closed: bool


def _split_language(inner: str) -> tuple[str | None, str]:
    """Splits a fence's inner text into its language tag and body.

    Args:
        inner: The text between an opening ``` and its close (or end of
            text, if unclosed).

    Returns:
        A `(language, content)` tuple. `language` is `None` when the first
        line doesn't look like a bare language tag.
    """
    match = _LANGUAGE_RE.match(inner)
    if not match:
        return None, inner
    language = match.group(1) or None
    content = inner[match.end():]
    return language, content


def extract_codeblocks(text: str) -> list[CodeBlock]:
    """Extracts all fenced codeblocks from text, closed or not.

    Args:
        text: Raw text to scan for ``` fenced codeblocks.

    Returns:
        A list of `CodeBlock` instances in the order they appear in `text`.
        Empty if no ``` fence is found. A trailing, unpaired fence yields a
        `CodeBlock` with `closed=False` whose content runs to the end of
        `text`.
    """
    fence_positions = [m.start() for m in re.finditer(re.escape(FENCE), text)]

    blocks: list[CodeBlock] = []
    i = 0
    while i < len(fence_positions):
        open_pos = fence_positions[i]
        inner_start = open_pos + len(FENCE)

        if i + 1 < len(fence_positions):
            close_pos = fence_positions[i + 1]
            inner_end = close_pos
            closed = True
            i += 2
        else:
            inner_end = len(text)
            closed = False
            i += 1

        inner = text[inner_start:inner_end]
        language, content = _split_language(inner)
        blocks.append(CodeBlock(language=language, content=content, closed=closed))

    return blocks
