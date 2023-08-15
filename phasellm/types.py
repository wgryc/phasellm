from phasellm.configurations import OpenAIConfiguration, AzureAPIConfiguration, AzureActiveDirectoryConfiguration

from typing import Union, Literal, Optional

CLAUDE_MODEL = Union[
    str,
    Literal["claude-v1"],
    Literal["claude-instant-1"],
    Literal["claude-instant-1.1"],
    Literal["claude-2"],
    Literal["claude-2.0"],
]

OPENAI_API_CONFIG = Union[
    OpenAIConfiguration,
    AzureAPIConfiguration,
    AzureActiveDirectoryConfiguration
]
