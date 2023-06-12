"""
Abstract classes and wrappers for LLMs, chatbots, and prompts.
"""
import re
import time
import json
import requests

# Typing imports
from typing import Optional, List, Union, Generator
from typing_extensions import TypedDict

# Abstract class imports
from abc import ABC, abstractmethod

from datetime import datetime
from sseclient import SSEClient

# Imports for external APIs
import openai
import cohere

# Hugging Face and PyTorch imports
from transformers import pipeline
import torch

# Precompiled regex for variables.
variable_pattern = r'\{\s*[a-zA-Z0-9_]+\s*\}'
variable_regex = re.compile(variable_pattern)

STOP_TOKEN = "<|END|>"


class Message(TypedDict):
    """
    Message type for chat messages.
    """
    role: str
    content: str


class EnhancedMessage(Message):
    """
    Message type for chat messages with additional metadata.
    """
    timestamp_utc: datetime
    log_time_seconds: float


def _fill_variables(source: str, **kwargs) -> str:
    # Collect the variables present in the source that need to be filled.
    variables = re.findall(variable_regex, source)
    # Create a copy of the source to be filled.
    filled = source
    for m in variables:
        keyword = m.replace("{", "").replace("}", "").strip()
        if keyword in kwargs:
            filled = filled.replace(m, kwargs[keyword])
    return filled


def _clean_messages_to_prompt(messages: List[Message]) -> str:
    """
    Converts an array of messages in the form {"role": <str>, "content":<str>} into a String.

    This is influenced by the OpenAI chat completion API.
    """
    return "\n".join([f"{str(m['role'])}: {str(m['content'])}" for m in messages])


def _truncate_completion(completion: str) -> str:
    """
    Truncates a completion to the first newline character.
    """
    newline_location = completion.find("\n")
    if newline_location > 0:
        completion = completion[:newline_location]
    return completion


def _get_stop_sequences_from_messages(messages: List[Message]):
    """
    Generates a list of strings of stop sequences from an array of messages in the form
    {"role": <str>, "content":<str>}.
    """
    roles = set()
    for m in messages:
        roles.add(m["role"])
    return [f"\n{r}:" for r in roles]


def _format_sse(content: str) -> str:
    """
    Returns the string that indicates that the response should be formatted as an SSE.
    """
    return f"data: {content}\n\n"


def _conditional_format_sse_response(content: str, format_sse: bool) -> str:
    if format_sse:
        return _format_sse(content)
    return content


class LanguageModelWrapper(ABC):
    """
    Abstract Class for interacting with large language models.
    """

    # default chat completion preamble
    chat_completion_preamble: str = (
        "You are a friendly chat assistant. "
        "You are speaking to the 'user' below and will respond at the end, where it says 'assistant'.\n"
    )

    def __init__(self):
        pass

    def __repr__(self):
        pass

    @abstractmethod
    def complete_chat(self, messages: List[Message], append_role: Optional[str] = None) -> Union[str, Generator]:
        """
        Takes an array of messages in the form {"role": <str>, "content":<str>} and generate a response.

        This is influenced by the OpenAI chat completion API.
        """
        pass

    @abstractmethod
    def text_completion(self, prompt: str) -> Union[str, Generator]:
        """
        Standardizes text completion for large language models.
        """
        pass

    def prep_prompt_from_messages(
            self,
            messages: List[Message] = None,
            append_role: Optional[str] = None,
            include_preamble: Optional[bool] = False
    ) -> str:
        """
        Prepares the prompt for an LLM API call.
        """
        # Convert the messages to a prompt.
        prompt_text = _clean_messages_to_prompt(messages)

        # Add the preamble, if requested.
        if include_preamble:
            prompt_text = self.chat_completion_preamble + prompt_text

        # Append the role, if provided.
        if append_role is not None and len(append_role) > 0:
            prompt_text += f"\n{append_role}:"

        # Remove whitespace from before and after prompt.
        return prompt_text.strip()


class StreamingLanguageModelWrapper(LanguageModelWrapper):
    """
    Abstract class for streaming language models. Extends the regular LanguageModelWrapper.
    """

    @abstractmethod
    def __init__(self, format_sse: bool, append_stop_token: bool = True, stop_token: str = STOP_TOKEN):
        super().__init__()
        self.format_sse = format_sse
        self.append_stop_token = append_stop_token
        self.stop_token = stop_token


class ChatPrompt:
    """
    This is used to generate messages for a ChatBot. Like the Prompt class, it enables you to to have variables that get
    replaced. This can be done for roles and messages.
    """

    def __init__(self, messages: List[Message] = None):
        # Set the messages
        if messages is None:
            self.messages = []
        else:
            self.messages = messages

    def __repr__(self):
        return "ChatPrompt()"

    def chat_repr(self):
        return _clean_messages_to_prompt(self.messages)

    def fill(self, **kwargs):
        filled_messages = []
        for i in range(0, len(self.messages)):
            new_role = _fill_variables(self.messages[i]["role"], **kwargs)
            new_content = _fill_variables(self.messages[i]["content"], **kwargs)

            filled_messages.append({"role": new_role, "content": new_content})
        return filled_messages


class Prompt:
    """
    Prompts are used to generate text completions. Prompts can be simple Strings. They can also include variables
    surrounded by curly braces.

    For example:
    > Hello {name}!

    In this case, 'name' can be filled using the fill() function. This makes it easier to loop through prompts
    that follow a specific pattern or structure.
    """

    def __init__(self, prompt: str):
        self.prompt = prompt

    def __repr__(self):
        return self.prompt

    def get_prompt(self):
        """
        Return the raw prompt command (i.e., does not fill in variables.)
        """
        return self.prompt

    def fill(self, **kwargs):
        """
        Return a prompt with variables filled in.
        """
        return _fill_variables(source=self.prompt, **kwargs)


class HuggingFaceInferenceWrapper(LanguageModelWrapper):
    """
    Wrapper for Hugging Face's Inference API. Requires access to Hugging Face's inference API.
    """

    def __init__(self, apikey, model_url="https://api-inference.huggingface.co/models/bigscience/bloom"):
        super().__init__()
        self.apikey = apikey
        self.model_url = model_url

    def __repr__(self):
        return f"HuggingFaceInferenceWrapper()"

    def complete_chat(self, messages: List[Message], append_role=None) -> str:
        """
        Mimics a chat scenario via a list of {"role": <str>, "content":<str>} objects.
        """

        prompt_text = self.prep_prompt_from_messages(
            messages=messages,
            append_role=append_role,
            include_preamble=True
        )

        headers = {
            "Authorization": f"Bearer {self.apikey}"
        }

        response = requests.post(self.model_url, headers=headers, json={"inputs": prompt_text}).json()
        new_text = response[0]['generated_text']

        # We only return the first line of text.
        return _truncate_completion(new_text)

    def text_completion(self, prompt) -> str:
        """
        Completes text.
        """
        headers = {"Authorization": f"Bearer {self.apikey}"}
        response = requests.post(self.model_url, headers=headers, json={"inputs": prompt}).json()
        return response[0]['generated_text']


# TODO consider deleting the BloomWrapper class since this functionality is in HuggingFaceInferenceWrapper
class BloomWrapper(LanguageModelWrapper):
    """
    Wrapper for Hugging Face's BLOOM model. Requires access to Hugging Face's inference API.
    """
    API_URL = "https://api-inference.huggingface.co/models/bigscience/bloom"

    def __init__(self, apikey):
        super().__init__()
        self.apikey = apikey

    def __repr__(self):
        return f"BloomWrapper()"

    def complete_chat(self, messages: List[Message], append_role=None) -> str:
        """
        Mimics a chat scenario with BLOOM, via a list of {"role": <str>, "content":<str>} objects.
        """

        prompt_text = self.prep_prompt_from_messages(
            messages=messages,
            append_role=append_role,
            include_preamble=True
        )

        headers = {"Authorization": f"Bearer {self.apikey}"}

        response = requests.post(self.API_URL, headers=headers, json={"inputs": prompt_text}).json()

        all_text = response[0]['generated_text']
        new_text = all_text[len(prompt_text):]

        # We only return the first line of text.
        return _truncate_completion(new_text)

    def text_completion(self, prompt) -> str:
        """
        Completes text via BLOOM (Hugging Face).
        """
        headers = {"Authorization": f"Bearer {self.apikey}"}

        response = requests.post(self.API_URL, headers=headers, json={"inputs": prompt}).json()
        all_text = response[0]['generated_text']
        return all_text[len(prompt):]


class StreamingOpenAIGPTWrapper(StreamingLanguageModelWrapper):
    """
    Streaming compliant wrapper for the OpenAI API. Supports all major text and chat completion models by OpenAI.
    """

    def __init__(self, apikey, model="gpt-3.5-turbo", format_sse=False, append_stop_token=True, stop_token=STOP_TOKEN):
        super().__init__(format_sse=format_sse, append_stop_token=append_stop_token, stop_token=stop_token)
        openai.api_key = apikey
        self.model: str = model

    def __repr__(self):
        return f"StreamingOpenAIGPTWrapper(model={self.model})"

    def complete_chat(self, messages, append_role=None) -> Generator:
        """
        Completes chat with OpenAI. If using GPT 3.5 or 4, will simply send the list of {"role": <str>, "content":<str>}
        objects to the API.

        If using an older model, it will structure the messages list into a prompt first.

        Yields the text as it is generated, rather than waiting for the entire completion.
        """
        if ('gpt-4' in self.model) or ('gpt-3.5' in self.model):
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
                stream=True
            )
            for chunk in response:
                if "content" in chunk["choices"][0]["delta"]:
                    content = chunk["choices"][0]["delta"]["content"]
                    yield _conditional_format_sse_response(content=content, format_sse=self.format_sse)
            if self.format_sse and self.append_stop_token:
                yield _format_sse(content=self.stop_token)
        else:
            prompt_text = self.prep_prompt_from_messages(
                messages=messages,
                append_role=append_role,
                include_preamble=False
            )

            response = openai.Completion.create(
                model=self.model,
                prompt=prompt_text,
                stop=_get_stop_sequences_from_messages(messages),
                stream=True
            )

            for chunk in response:
                if "text" in chunk["choices"][0]["delta"]:
                    text = chunk["choices"][0]["delta"]["text"]
                    yield _conditional_format_sse_response(content=text, format_sse=self.format_sse)
            if self.format_sse and self.append_stop_token:
                yield _format_sse(content=self.stop_token)

    # TODO Consider error handling for chat models.
    def text_completion(self, prompt, stop_sequences=None) -> Generator:
        """
        Completes text via OpenAI. Note that this doesn't support GPT 3.5 or later, as they are chat models.

        Yields the text as it is generated, rather than waiting for the entire completion.
        """
        if stop_sequences is None:
            stop_sequences = []

        if len(stop_sequences) == 0:
            response = openai.Completion.create(
                model=self.model,
                prompt=prompt,
                stream=True
            )
        else:
            response = openai.Completion.create(
                model=self.model,
                prompt=prompt,
                stop=stop_sequences,
                stream=True
            )

        for chunk in response:
            if "text" in chunk["choices"][0]:
                text = chunk["choices"][0]["text"]
                yield _conditional_format_sse_response(content=text, format_sse=self.format_sse)
        if self.format_sse and self.append_stop_token:
            yield _format_sse(content=self.stop_token)


class OpenAIGPTWrapper(LanguageModelWrapper):
    """
    Wrapper for the OpenAI API. Supports all major text and chat completion models by OpenAI.
    """

    def __init__(self, apikey, model="gpt-3.5-turbo"):
        super().__init__()
        openai.api_key = apikey
        self.model = model

    def __repr__(self):
        return f"OpenAIGPTWrapper(model={self.model})"

    def complete_chat(self, messages, append_role=None) -> str:
        """
        Completes chat with OpenAI. If using GPT 3.5 or 4, will simply send the list of {"role": <str>, "content":<str>}
        objects to the API.

        If using an older model, it will structure the messages list into a prompt first.
        """

        if ('gpt-4' in self.model) or ('gpt-3.5' in self.model):
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=messages
            )
            top_response_content = response['choices'][0]['message']['content']
            return top_response_content
        else:
            prompt_text = self.prep_prompt_from_messages(
                messages=messages,
                append_role=append_role,
                include_preamble=False
            )

            response = openai.Completion.create(
                model=self.model,
                prompt=prompt_text,
                stop=_get_stop_sequences_from_messages(messages)
            )

            return response['choices'][0]['text']

    # TODO Consider error handling for chat models.
    def text_completion(self, prompt, stop_sequences=None) -> str:
        """
        Completes text via OpenAI. Note that this doesn't support GPT 3.5 or later, as they are chat models.
        """
        if stop_sequences is None:
            stop_sequences = []

        if len(stop_sequences) == 0:
            response = openai.Completion.create(
                model=self.model,
                prompt=prompt
            )
        else:
            response = openai.Completion.create(
                model=self.model,
                prompt=prompt,
                stop=stop_sequences
            )
        return response['choices'][0]['text']


class StreamingClaudeWrapper(StreamingLanguageModelWrapper):
    """
    Streaming wrapper for Anthropic's Claude large language model.

    We've opted to call Anthropic's API directly rather than using their Python offering.

    Yields the text as it is generated, rather than waiting for the entire completion.
    """
    API_URL = "https://api.anthropic.com/v1/complete"

    def __init__(self, apikey, model="claude-v1", format_sse=False, append_stop_token=True, stop_token=STOP_TOKEN):
        super().__init__(format_sse=format_sse, append_stop_token=append_stop_token, stop_token=stop_token)
        self.apikey = apikey
        self.model = model

    def __repr__(self):
        return f"StreamingClaudeWrapper(model={self.model})"

    def _call_model(self, prompt: str, stop_sequences: List[str]) -> Generator:
        """
        Calls the model with the given prompt.
        """
        headers = {
            "X-API-Key": self.apikey,
            "Accept": "text/event-stream"
        }

        payload = {
            "prompt": prompt,
            "model": self.model,
            "max_tokens_to_sample": 500,
            "stop_sequences": stop_sequences,
            "stream": True
        }

        resp = requests.post(self.API_URL, headers=headers, json=payload, stream=True)
        client = SSEClient(resp)

        strip_index = 0
        for event in client.events():
            if event.data != "[DONE]":
                # Load the data as JSON
                completion = json.loads(event.data)["completion"]

                # Anthropic's API returns completions inclusive of previous chunks, so we need to strip them out.
                completion = completion[strip_index:]
                strip_index += len(completion)

                # If format_sse is True, we need to yield with SSE formatting.
                yield _conditional_format_sse_response(content=completion, format_sse=self.format_sse)
        if self.format_sse and self.append_stop_token:
            yield _format_sse(content=self.stop_token)

    def complete_chat(self, messages: List[Message], append_role="Assistant:") -> Generator:
        """
        Completes chat with Claude. Since Claude doesn't support a chat interface via API, we mimic the chat via the a
        prompt.
        """

        prompt_text = self.prep_prompt_from_messages(
            messages=messages,
            append_role=append_role,
            include_preamble=False
        )

        return self._call_model(
            prompt=prompt_text,
            stop_sequences=_get_stop_sequences_from_messages(messages)
        )

    def text_completion(self, prompt, stop_sequences=None) -> Generator:
        """
        Completes text based on provided prompt.

        Yields the text as it is generated, rather than waiting for the entire completion.
        """

        if stop_sequences is None:
            stop_sequences = []

        return self._call_model(
            prompt=prompt,
            stop_sequences=stop_sequences
        )


class ClaudeWrapper(LanguageModelWrapper):
    """
    Wrapper for Anthropic's Claude large language model.

    We've opted to call Anthropic's API directly rather than using their Python offering.
    """
    API_URL = "https://api.anthropic.com/v1/complete"

    def __init__(self, apikey, model="claude-v1"):
        super().__init__()
        self.apikey = apikey
        self.model = model

    def __repr__(self):
        return f"ClaudeWrapper(model={self.model})"

    def _call_model(self, prompt: str, messages: List[Message]) -> str:
        headers = {
            "X-API-Key": self.apikey,
            "Accept": "application/json"
        }

        payload = {
            "prompt": prompt,
            "model": self.model,
            "max_tokens_to_sample": 500,
            "stop_sequences": _get_stop_sequences_from_messages(messages)
        }

        resp = requests.post("https://api.anthropic.com/v1/complete", headers=headers, json=payload)
        return json.loads(resp.text)["completion"].strip()

    def complete_chat(self, messages, append_role="Assistant:") -> str:
        """
        Completes chat with Claude. Since Claude doesn't support a chat interface via API, we mimic the chat via the a
        prompt.
        """

        prompt_text = self.prep_prompt_from_messages(
            messages=messages,
            append_role=append_role,
            include_preamble=False
        )

        return self._call_model(prompt_text, messages)

    def text_completion(self, prompt, stop_sequences=None) -> str:
        """
        Completes text based on provided prompt.
        """

        if stop_sequences is None:
            stop_sequences = []

        return self._call_model(prompt, stop_sequences)


# TODO Might want to add stop sequences (new lines, roles) to make this better.
class GPT2Wrapper(LanguageModelWrapper):
    """
    Wrapper for GPT-2 implementation (via Hugging Face). 
    """

    def __init__(self):
        super().__init__()
        self.model_name = "GPT-2"

    def __repr__(self):
        return f"GPT2Wrapper()"

    def complete_chat(self, messages, append_role=None, max_length=300) -> str:
        """
        Mimics a chat scenario via a list of {"role": <str>, "content":<str>} objects.
        """

        prompt_text = self.prep_prompt_from_messages(
            messages=messages,
            append_role=append_role,
            include_preamble=True
        )

        generator = pipeline('text-generation', model='gpt2')
        resps = generator(prompt_text, max_length=max_length, num_return_sequences=1)
        resp = resps[0]['generated_text']
        return resp[len(prompt_text):]  # Strip out the original text.

    def text_completion(self, prompt, max_length=200) -> str:
        """
        Completes text via GPT-2.
        """
        generator = pipeline('text-generation', model='gpt2')
        resps = generator(prompt, max_length=max_length, num_return_sequences=1)
        resp = resps[0]['generated_text']
        return resp[len(prompt):]  # Strip out the original text.


class DollyWrapper(LanguageModelWrapper):
    """
    Implementation of Dolly 2.0 (via Hugging Face).
    """

    def __init__(self):
        super().__init__()
        self.model_name = 'dolly-v2-12b'
        self.generate_text = pipeline(model="databricks/dolly-v2-12b", torch_dtype=torch.bfloat16,
                                      trust_remote_code=True, device_map="auto")

    def __repr__(self):
        return f"DollyWrapper(model={self.model_name})"

    def complete_chat(self, messages, append_role=None) -> str:
        """
        Mimics a chat scenario via a list of {"role": <str>, "content":<str>} objects.
        """

        prompt_text = self.prep_prompt_from_messages(
            messages=messages,
            append_role=append_role,
            include_preamble=True
        )

        return self.generate_text(prompt_text)

    def text_completion(self, prompt) -> str:
        """
        Completes text via Dolly.
        """
        return self.generate_text(prompt)


class CohereWrapper(LanguageModelWrapper):
    """
    Wrapper for Cohere's API. Defaults to their 'xlarge' model.
    """

    def __init__(self, apikey, model="xlarge"):
        super().__init__()
        self.apikey = apikey
        self.model = model

    def __repr__(self):
        return f"CohereWrapper(model={self.model})"

    def complete_chat(self, messages, append_role=None) -> str:
        """
        Mimics a chat scenario via a list of {"role": <str>, "content":<str>} objects.
        """

        prompt_text = self.prep_prompt_from_messages(
            messages=messages,
            append_role=append_role,
            include_preamble=False
        )

        co = cohere.Client(self.apikey)
        response = co.generate(
            prompt=prompt_text,
            max_tokens=300,
            stop_sequences=_get_stop_sequences_from_messages(messages)
        )

        resp = response.generations[0].text

        for s in _get_stop_sequences_from_messages(messages):
            resp = resp.replace(s, "").strip()

        return resp

    def text_completion(self, prompt, stop_sequences=None) -> str:
        """
        Completes text.
        """

        if stop_sequences is None:
            stop_sequences = []

        co = cohere.Client(self.apikey)
        response = co.generate(
            prompt=prompt,
            max_tokens=300,
            stop_sequences=stop_sequences
        )
        return response.generations[0].text


class ChatBot:
    """
    Allows you to have a chat conversation with an LLM wrapper.

    In short, it manages the list of {"role": <str>, "content":<str>} objects for you, so you don't have to figure this
    out. It also interacts directly with the model.
    """

    def __init__(self, llm: LanguageModelWrapper, initial_system_prompt="You are a friendly chatbot assistant."):
        """
        Initializes a ChatBot. Provide an initial_system_prompt value to request the type of chatbot you will be
        dealing with.
        
        Warning: not all LLMs are trained to use instructions provided in a system prompt.
        """
        self.llm: LanguageModelWrapper = llm
        self.messages: List[EnhancedMessage] = []
        self._append_message('system', initial_system_prompt)

    def _response(self, response: str, start_time: float):
        """
        Handles a response from the LLM.
        """
        self._append_message('assistant', response, log_time_seconds=time.time() - start_time)
        return response

    def _streaming_response(self, response: Generator, start_time: float):
        """
        Handles a streaming response from the LLM.

        If the response is a generator, we'll need intercept it so that we can append it to the message stack.
        (Generators only yield their results once).
        """
        full_response = ''
        for chunk in response:
            full_response += chunk
            yield chunk
        self._append_message('assistant', full_response, log_time_seconds=time.time() - start_time)

    def _append_message(self, role: str, message: str, log_time_seconds=None) -> None:
        """
        Saves a message to the ChatBot message queue.
        """

        # Create the message object.
        append_me: EnhancedMessage = {
            "role": role,
            "content": message,
            "timestamp_utc": datetime.now()
        }

        # Save how long it took to generate the message, if provided.
        if log_time_seconds is not None:
            append_me["log_time_seconds"] = log_time_seconds

        self.messages.append(append_me)

    def resend(self) -> Optional[Union[str, Generator]]:
        """
        If the last message in the messages stack (i.e. array of role and content pairs) is from the user, it will
        resend the message and return the response. It's similar to erasing the last message in the stack and resending
        the last user message to the chat model.

        This is useful if a model raises an error or if you are building a broader messages stack outside of the
        actual chatbot.
        """
        # TODO consider if this is necessary, given the TODO suggestion in self.chat().
        last_message = self.messages.pop()
        if last_message['role'] == 'user':
            return self.chat(last_message['content'])
        else:
            self.messages.append(last_message)

    def chat(self, message) -> Union[str, Generator]:
        """
        Chats with the chatbot.
        """
        # TODO consider appending user message only after a successful call to self.llm.complete_chat().
        self._append_message('user', message)
        start_time = time.time()

        clean_messages = []  # We remove fields that the ChatBot class specifically tracks.
        for m in self.messages:
            m_copy = {"role": m["role"], "content": m["content"]}
            clean_messages.append(m_copy)

        response = self.llm.complete_chat(clean_messages, append_role='assistant')

        if isinstance(response, Generator):
            return self._streaming_response(response=response, start_time=start_time)
        else:
            return self._response(response=response, start_time=start_time)
