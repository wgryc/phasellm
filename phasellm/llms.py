"""
Abstract classes and wrappers for LLMs, chatbots, and prompts.
"""

import requests
import json

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set

# Imports for external APIs
import openai
import cohere

# Hugging Face and PyTorch imports
from transformers import pipeline
import torch

from phasellm.prompts import  _find_stop_sequences, _stringify

class LanguageModelWrapper(ABC):
    """
    An abstract wrapper for LLMs from upstream providers.
    """

    @abstractmethod
    def __init__(self, name: str, api_key: str) -> None:
        self.name = name
        self.api_key = api_key

    @abstractmethod
    def __repr__(self) -> str:
        pass

    # TODO: Specify return type(s)
    @abstractmethod
    def complete_chat(self, messages: List[Dict], append_role: Optional[str]=None):
        """
        Submits a list of message objects (such as { "role": str, "content": str }) to an upstream LLM and returns an LLM response.

        This is influenced by the OpenAI chat completion API.
        """
        pass
    
    # TODO: Specify return type(s)
    @abstractmethod
    def text_completion(self, prompt, stop_sequences=[]):
        """
        Standardizes text completion for large language models.
        """
        pass

class HuggingFaceInferenceWrapper(LanguageModelWrapper):
    """
    Wrapper for Hugging Face's Inference API. Requires access to Hugging Face's inference API.
    """
    def __init__(self,
                 name: str,
                 api_key: str,
                 model_url="https://api-inference.huggingface.co/models/bigscience/bloom"
    ) -> None:
        super().__init__(name=name, api_key=api_key)
        self.model_url = model_url

    def __repr__(self) -> str:
        return f"HuggingFaceInferenceWrapper()"

    def complete_chat(self, messages: List[Dict], append_role: Optional[str]=None):
        """
        Mimicks a chat scenario via a list of { "role": str, "content": str } objects.
        """

        prompt_prefix = "You are a friendly chat assistant. You are speaking to the 'user' below and will respond at the end, where it says 'assistant'.\n"
        prompt_text = prompt_prefix + _stringify(messages)
        
        if append_role is not None and len(append_role) > 0:
            prompt_text += f"\n{append_role}:"

        request_headers = {
            "Authorization": f"Bearer {self.api_key}"
        }

        response = requests.post(
            url=self.model_url,
            headers=request_headers,
            json={ "inputs": prompt_text }
        ).json()
        
        new_text = response[0]['generated_text']

        # We only return the first line of text.
        newline_location = new_text.find("\n") 
        if newline_location > 0:
            new_text = new_text[:newline_location]

        return new_text

    def text_completion(self, prompt):
        """
        Completes text.
        """
        headers = {"Authorization": f"Bearer {self.api_key}"}
        response = requests.post(self.model_url, headers=headers, json={"inputs": prompt}).json()
        all_text = response[0]['generated_text']
        return all_text

class BloomWrapper(LanguageModelWrapper):
    """
    Wrapper for Hugging Face's BLOOM model. Requires access to Hugging Face's inference API.
    """
    def __init__(self, name: str, api_key: str, model) -> None:
        super().__init__(name=name, api_key=api_key)

    def __repr__(self) -> str:
        return f"BloomWrapper()"

    def complete_chat(self, messages: List[Dict], append_role: Optional[str]=None):
        """
        Mimicks a chat scenario with BLOOM, via a list of {"role": <str>, "content":<str>} objects.
        """

        prompt_prefix = "You are a friendly chat assistant. You are speaking to the 'user' below and will respond at the end, where it says 'assistant'.\n"
        prompt_text = prompt_prefix + _stringify(messages)

        if append_role is not None and len(append_role) > 0:
            prompt_text += f"\n{append_role}:"

        API_URL = "https://api-inference.huggingface.co/models/bigscience/bloom"
        headers = {"Authorization": f"Bearer {self.api_key}"}

        response = requests.post(API_URL, headers=headers, json={"inputs": prompt_text}).json()

        all_text = response[0]['generated_text']
        new_text = all_text[len(prompt_text):]

        # We only return the first line of text.
        newline_location = new_text.find("\n") 
        if newline_location > 0:
            new_text = new_text[:newline_location]

        return new_text

    def text_completion(self, prompt):
        """
        Completes text via BLOOM (Hugging Face).
        """
        API_URL = "https://api-inference.huggingface.co/models/bigscience/bloom"
        headers = {"Authorization": f"Bearer {self.api_key}"}

        response = requests.post(API_URL, headers=headers, json={"inputs": prompt}).json()
        all_text = response[0]['generated_text']
        new_text = all_text[len(prompt):]
        return new_text

class OpenAIGPTWrapper(LanguageModelWrapper):
    """
    A wrapper for the OpenAI API.
    """

    def __init__(self, name: str, api_key: str, model:str ="gpt-3.5-turbo") -> None:
        super().__init__(name=name, api_key=api_key)
        openai.api_key = api_key # TODO: Standardise openai.api_lkey to self.api_key
        self.model = model

    def __repr__(self) -> str:
        return f"OpenAIGPTWrapper(model={self.model})"
    
    def complete_chat(self, messages: List[Dict], append_role: Optional[str]=None) -> str:
        """
        Completes a chat with OpenAI's API.
        If using GPT 3.5 or 4, it will simply send the list of { "role": str, "content": str } objects to the OpenAI API.
        If using an older model, it will format the messages to a prompt first.
        """

        if self.model.find('gpt-4') >= 0 or self.model.find('gpt-3.5') >= 0:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=messages
            )
            completion = response['choices'][0]['message']['content']
            
            return completion
        else:
            prompt_text: str = _stringify(messages)

            if append_role is not None and len(append_role) > 0:
                prompt_text += f"\n{append_role}: "
            
            prompt_text = prompt_text.strip()

            response = openai.Completion.create(
                model=self.model,
                prompt=prompt_text,
                stop=_find_stop_sequences(messages)
            )

            completion = response['choices'][0]['text']

            return completion
    
    # Note that this currently will error out with GPT 3.5 or above as they are chat models.
    # TODO Add error catching.
    def text_completion(self, prompt, stop_sequences=[]):
        """
        Completes text via OpenAI. Note that this doesn't support GPT 3.5 or later.
        """

        if len(stop_sequences) == 0:
            response = openai.Completion.create(
                model=self.model,
                prompt=prompt
            )
        else:
            response = openai.Completion.create(
                model=self.model,
                prompt=prompt,
                stop = stop_sequences
            )
        top_response_content = response['choices'][0]['text']
        return top_response_content

class ClaudeWrapper(LanguageModelWrapper):
    """
    A wrapper for Anthropic's Claude API.
    """

    def __init__(self, name: str, api_key: str, model: str ="claude-v1") -> None:
        super().__init__(name=name, api_key=api_key)
        self.model = model

    def __repr__(self) -> str:
        return f"ClaudeWrapper(model={self.model})"
    
    def complete_chat(self, messages: List[Dict], append_role: Optional[str]=None) -> str:
        """
        Completes a chat with Claude.
        Since Claude doesn't support a real chat interface via its API, we mimick the chat via a proxy prompt.
        """

        request_headers = {
            "X-API-Key": self.api_key,
            "Accept": "application/json"
        }

        prompt_text = _stringify(messages)

        if append_role is not None and len(append_role) > 0:
            prompt_text += f"\n{append_role}: "

        request_payload = {
            "prompt": prompt_text,
            "model": self.model,
            "max_tokens_to_sample": 500,
            "stop_sequences": _find_stop_sequences(messages)
        }

        response = requests.post(
            url="https://api.anthropic.com/v1/complete",
            headers=request_headers,
            json=request_payload
        )
        completion = json.loads(response.text)["completion"].strip()

        return completion
    
    def text_completion(self, prompt, stop_sequences=[]):
        """
        Completes text based on provided prompt.
        """

        r_headers = {"X-API-Key":self.api_key, "Accept":"application/json"}
        r_data = {"prompt": prompt,
                  "model": self.model,
                  "max_tokens_to_sample": 500,
                  "stop_sequences": stop_sequences
                }

        resp = requests.post("https://api.anthropic.com/v1/complete", headers=r_headers, json=r_data)
        completion = json.loads(resp.text)["completion"].strip()
        return completion

# TODO Might want to add stop sequences (new lines, roles) to make this better.
class GPT2Wrapper(LanguageModelWrapper):
    """
    Wrapper for GPT-2 implementation (via Hugging Face). 
    """

    def __init__(self, name: str, api_key: str) -> None:
        super().__init__(name=name, api_key=api_key)
        self.model_name = "GPT-2"

    def __repr__(self) -> str:
        return f"GPT2Wrapper()"
    
    def complete_chat(self,
                      messages: List[Dict],
                      append_role: Optional[str]=None,
                      max_length:int =300
    ) -> str:
        """
        Mimicks a chat scenario via a list of {"role": <str>, "content":<str>} objects. 
        """

        prompt_prefix = "You are a friendly chat assistant. You are speaking to the 'user' below and will respond at the end, where it says 'assistant'.\n"
        prompt_text = prompt_prefix + _stringify(messages)
        if append_role is not None and len(append_role) > 0:
            prompt_text += f"\n{append_role}:"

        generator = pipeline('text-generation', model='gpt2')
        resps = generator(prompt_text, max_length=max_length, num_return_sequences=1)
        resp = resps[0]['generated_text']
        resp = resp[len(prompt_text):] # Strip out the original text.
        return resp

    def text_completion(self, prompt, max_length=200):
        """
        Completes text via GPT-2.
        """
        generator = pipeline('text-generation', model='gpt2')
        resps = generator(prompt, max_length=max_length, num_return_sequences=1)
        resp = resps[0]['generated_text']
        resp = resp[len(prompt):] # Strip out the original text.
        return resp

class DollyWrapper(LanguageModelWrapper):
    """
    Implementation of Dolly 2.0 (via Hugging Face).
    """

    def __init__(self, name: str, api_key: str) -> None:
        super().__init__(name=name, api_key=api_key)
        self.model_name = 'dolly-v2-12b'
        self.generate_text = pipeline(model="databricks/dolly-v2-12b", torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")

    def __repr__(self) -> str:
        return f"DollyWrapper(model={self.model})"

    def complete_chat(self, messages: List[Dict], append_role: Optional[str]=None):
        """
        Mimicks a chat scenario via a list of {"role": <str>, "content":<str>} objects. 
        """

        prompt_prefix = "You are a friendly chat assistant. You are speaking to the 'user' below and will respond at the end, where it says 'assistant'.\n"

        prompt_text = prompt_prefix + _stringify(messages)

        if append_role is not None and len(append_role) > 0:
            prompt_text += f"\n{append_role}:"

        resp = self.generate_text(prompt_text)

        return resp

    def text_completion(self, prompt):
        """
        Complates
        """
        resp = self.generate_text(prompt)
        return resp

class CohereWrapper(LanguageModelWrapper):
    """
    A wrapper for Cohere's LLM API.
    Defaults to Cohere's 'xlarge' model.
    """

    def __init__(self, name: str, api_key: str, model: str ="xlarge") -> None:
        super().__init__(name=name, api_key=api_key)
        self.model = model

    def __repr__(self) -> str:
        return f"CohereWrapper(model={self.model})"

    def complete_chat(self, messages: List[Dict], append_role: Optional[str]=None) -> str:
        """
        Completes a chat with Cohere.
        Since Cohere doesn't support a real chat interface via API, we mimick the chat via a list of { "role": str, "content": str } objects. 
        """

        prompt_text = _stringify(messages)

        if append_role is not None and len(append_role) > 0:
            prompt_text += f"\n{append_role}:"

        co = cohere.Client(self.api_key)

        stop_sequences = _find_stop_sequences(messages)

        response = co.generate(
            prompt=prompt_text,
            max_tokens=300, 
            stop_sequences=stop_sequences
        )

        completion = response.generations[0].text

        for stop_sequence in stop_sequences:
            completion = completion.replace(stop_sequence, "").strip()

        return completion

    def text_completion(self, prompt, stop_sequences=[]):
        """
        Completes text.
        """
        co = cohere.Client(self.api_key)
        response = co.generate(
            prompt=prompt,
            max_tokens=300, 
            stop_sequences=stop_sequences
        )
        resp = response.generations[0].text
        return resp

class ChatBot():
    """
    A utility interface for a chat conversation with an LLM wrapper.

    It takes care of { "role": str, "content": str } message objects for you, and interacts directly with an LLM model.
    """

    def __init__(self, 
                 llm: LanguageModelWrapper,
                 initial_system_prompt:str ="You are a friendly chatbot assistant."
    ) -> None:
        """
        Initializes a ChatBot with an initial_system_prompt value to specify a chatbot type that you will be chatting with.
        Warning: not all LLMs are trained to use instructions provided in a system prompt.
        """
        self.llm = llm 
        self.messages: List[Dict] = []
        self._add_message(role='system', message=initial_system_prompt)

    def _add_message(self, role: str, message: str) -> None:
        """
        Adds a message to the chatbot's message queue.
        """
        self.messages.append({"role": role, "content": message})

    def chat(self, message) -> Any:
        """
        Completes a turn with a chat bot.
        """
        self._add_message(role='user', message=message)

        response = self.llm.complete_chat(messages=self.messages, message="assistant")

        self._add_message(role='assistant', message=response)

        return response
    
