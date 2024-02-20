from phasellm.configurations import OpenAIConfiguration, AzureAPIConfiguration, AzureActiveDirectoryConfiguration, \
    VertexAIConfiguration

from typing import Union, Literal

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

VERTEXAI_API_CONFIG = VertexAIConfiguration
