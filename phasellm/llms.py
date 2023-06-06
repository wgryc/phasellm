"""
Abstract classes and wrappers for LLMs, chatbots, and prompts.
"""

from typing import Optional, List
from typing_extensions import TypedDict

from abc import ABC, abstractmethod

import requests
import json
import re
import time
from datetime import datetime

# Imports for external APIs
import openai
import cohere

# Hugging Face and PyTorch imports
from transformers import pipeline
import torch

# Precompiled regex for variables.
variable_pattern = r'\{\s*[a-zA-Z0-9_]+\s*\}'
variable_regex = re.compile(variable_pattern)


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
            filled = source.replace(m, kwargs[keyword])
    return filled


def _clean_messages_to_prompt(messages: List[Message]) -> str:
    """
    Converts an array of messages in the form {"role": <str>, "content":<str>} into a String.

    This is influenced by the OpenAI chat completion API.
    """
    out_text = "\n".join([f"{str(m['role'])}: {str(m['content'])}" for m in messages])
    return out_text


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
    stop_sequences = [f"\n{r}:" for r in roles]
    return stop_sequences


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
    def complete_chat(self, messages: List[Message], append_role: Optional[str] = None) -> str:
        """
        Takes an array of messages in the form {"role": <str>, "content":<str>} and generate a response.

        This is influenced by the OpenAI chat completion API.
        """
        pass

    @abstractmethod
    def text_completion(self, prompt: str) -> str:
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
        new_text = _truncate_completion(new_text)

        return new_text

    def text_completion(self, prompt) -> str:
        """
        Completes text.
        """
        headers = {"Authorization": f"Bearer {self.apikey}"}
        response = requests.post(self.model_url, headers=headers, json={"inputs": prompt}).json()
        all_text = response[0]['generated_text']
        return all_text


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
        new_text = _truncate_completion(new_text)

        return new_text

    def text_completion(self, prompt) -> str:
        """
        Completes text via BLOOM (Hugging Face).
        """
        headers = {"Authorization": f"Bearer {self.apikey}"}

        response = requests.post(self.API_URL, headers=headers, json={"inputs": prompt}).json()
        all_text = response[0]['generated_text']
        new_text = all_text[len(prompt):]
        return new_text


class StreamingOpenAIGPTWrapper(LanguageModelWrapper):
    """
    Streaming compliant wrapper for the OpenAI API. Supports all major text and chat completion models by OpenAI.
    """

    def __init__(self, apikey, model="gpt-3.5-turbo"):
        super().__init__()
        openai.api_key = apikey
        self.model = model

    def __repr__(self):
        return f"StreamingOpenAIGPTWrapper(model={self.model})"

    def complete_chat(self, messages, append_role=None) -> str:
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
                    yield chunk["choices"][0]["delta"]["content"]
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
                    yield chunk["choices"][0]["delta"]["text"]

    # TODO Add error handling for gpt-3.5 and gpt-4.
    def text_completion(self, prompt, stop_sequences=None) -> str:
        """
        Completes text via OpenAI. Note that this doesn't support GPT 3.5 or later, as they are chat models.

        Yields the text as it is generated, rather than waiting for the entire completion.
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

        for chunk in response:
            if "text" in chunk["choices"][0]["delta"]:
                yield chunk["choices"][0]["delta"]["text"]


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

            top_response_content = response['choices'][0]['text']
            return top_response_content

    # TODO Add error handling for gpt-3.5 and gpt-4.
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
        top_response_content = response['choices'][0]['text']
        return top_response_content


class ClaudeWrapper(LanguageModelWrapper):
    """
    Wrapper for Anthropic's Claude large language model.

    We've opted to call Anthropic's API directly rather than using their Python offering.
    """

    def __init__(self, apikey, model="claude-v1"):
        super().__init__()
        self.apikey = apikey
        self.model = model

    def __repr__(self):
        return f"ClaudeWrapper(model={self.model})"

    def complete_chat(self, messages, append_role="Assistant:") -> str:
        """
        Completes chat with Claude. Since Claude doesn't support a chat interface via API, we mimic the chat via the a
        prompt.
        """

        headers = {
            "X-API-Key": self.apikey,
            "Accept": "application/json"
        }

        prompt_text = self.prep_prompt_from_messages(
            messages=messages,
            append_role=append_role,
            include_preamble=False
        )

        payload = {
            "prompt": prompt_text,
            "model": self.model,
            "max_tokens_to_sample": 500,
            "stop_sequences": _get_stop_sequences_from_messages(messages)
        }

        resp = requests.post("https://api.anthropic.com/v1/complete", headers=headers, json=payload)
        completion = json.loads(resp.text)["completion"].strip()

        return completion

    def text_completion(self, prompt, stop_sequences=None) -> str:
        """
        Completes text based on provided prompt.
        """

        if stop_sequences is None:
            stop_sequences = []

        headers = {
            "X-API-Key": self.apikey,
            "Accept": "application/json"
        }

        payload = {
            "prompt": prompt,
            "model": self.model,
            "max_tokens_to_sample": 500,
            "stop_sequences": stop_sequences
        }

        resp = requests.post("https://api.anthropic.com/v1/complete", headers=headers, json=payload)
        completion = json.loads(resp.text)["completion"].strip()
        return completion


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
        resp = resp[len(prompt_text):]  # Strip out the original text.
        return resp

    def text_completion(self, prompt, max_length=200) -> str:
        """
        Completes text via GPT-2.
        """
        generator = pipeline('text-generation', model='gpt2')
        resps = generator(prompt, max_length=max_length, num_return_sequences=1)
        resp = resps[0]['generated_text']
        resp = resp[len(prompt):]  # Strip out the original text.
        return resp


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

        resp = self.generate_text(prompt_text)

        return resp

    def text_completion(self, prompt) -> str:
        """
        Completes text via Dolly.
        """
        resp = self.generate_text(prompt)
        return resp


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
        resp = response.generations[0].text
        return resp


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

    def resend(self):
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
            response = self.chat(last_message['content'])
            return response
        else:
            self.messages.append(last_message)
            return None

    def chat(self, message):
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
        self._append_message('assistant', response, log_time_seconds=time.time() - start_time)
        return response
