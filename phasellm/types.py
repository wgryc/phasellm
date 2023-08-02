from typing import Union, Literal

CLAUDE_MODEL = Union[
    str,
    Literal["claude-v1"],
    Literal["claude-instant-1"],
    Literal["claude-instant-1.1"],
    Literal["claude-2"],
    Literal["claude-2.0"],
]
