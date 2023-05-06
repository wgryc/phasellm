"""
Abstract classes and wrappers for LLMs, chatbots, and prompts.
"""

import requests
import json
import re

# Imports for external APIs
import openai
import cohere

# Hugging Face and PyTorch imports
from transformers import pipeline
import torch

def _clean_messages_to_prompt(messages):
    """
    Converts an array of messages in the form {"role": <str>, "content":<str>} into a String.

    This is influened by the OpenAI chat completion API.
    """
    out_text = "\n".join([f"{str(m['role'])}: {str(m['content'])}" for m in messages])
    return out_text

def _get_stop_sequences_from_messages(messages):
    """
    Generetes a list of strings of stop sequences from an array of messages in the form {"role": <str>, "content":<str>}.
    """
    roles = set()
    for m in messages:
        roles.add(m["role"])
    stop_sequences = [f"\n{r}:" for r in roles]
    return stop_sequences

class LanguageModelWrapper():
    """
    Abstract wrapper for large language models.
    """

    def __init__(self):
        pass

    def __repr__(self):
        pass

    def complete_chat(self, messages):
        """
        Takes an array of messages in the form {"role": <str>, "content":<str>} and generate a response.

        This is influened by the OpenAI chat completion API.
        """
        pass

    def text_completion(self, prompt, stop_sequences=[]):
        """
        Standardizes text completion for large language models.
        """
        pass

class ChatPrompt():
    """
    This is used to generate messages for a ChatBot. Like the Prompt class, it enables you to to have variables that get replaced. This can be done for roles and messages.
    """

    def __init__(self, messages=[]):

        self.messages = messages 

    def __repr__(self):
        return "ChatPrompt()"
    
    def chat_repr(self):
        return _clean_messages_to_prompt(self.messages)
    
    def fill(self, **kwargs):
        filled_messages = []
        for i in range(0, len(self.messages)):
            pattern = r'\{\s*[a-zA-Z0-9_]+\s*\}'

            role = self.messages[i]["role"]
            content = self.messages[i]["content"]

            role_matches = re.findall(pattern, role)
            new_role = role 
            for m in role_matches:
                keyword = m.replace("{", "").replace("}", "").strip()
                if keyword in kwargs:
                    new_role = new_role.replace(m, kwargs[keyword])

            content_matches = re.findall(pattern, content)
            new_content = content 
            for m in content_matches:
                keyword = m.replace("{", "").replace("}", "").strip()
                if keyword in kwargs:
                    new_content = new_content.replace(m, kwargs[keyword])

            filled_messages.append({"role":new_role, "content":new_content})

        return filled_messages

class Prompt():
    """
    Prompts are used to generate text completions. Prompts can be simple Strings. They can also include variables surrounded by curly braces. For example:
    > Hello {name}!
    In this case, 'name' can be filled using the fill_prompts() function. This makes it easier to loop through prompts that follow a specific pattern or structure.
    """

    def __init__(self, prompt):
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
        pattern = r'\{\s*[a-zA-Z0-9_]+\s*\}'
        matches = re.findall(pattern, self.prompt)
        new_prompt = self.prompt
        for m in matches:
            keyword = m.replace("{", "").replace("}", "").strip()
            if keyword in kwargs:
                new_prompt = new_prompt.replace(m, kwargs[keyword])
        return new_prompt

class HuggingFaceInferenceWrapper():
    """
    Wrapper for Hugging Face's Inference API. Requires access to Hugging Face's inference API.
    """
    def __init__(self, apikey, model_url="https://api-inference.huggingface.co/models/bigscience/bloom"):
        self.apikey = apikey
        self.model_url = model_url

    def __repr__(self):
        return f"HuggingFaceInferenceWrapper()"


    def complete_chat(self, messages, append_role=None):
        """
        Mimicks a chat scenario via a list of {"role": <str>, "content":<str>} objects.
        """

        prompt_preamble = "You are a friendly chat assistant. You are speaking to the 'user' below and will respond at the end, where it says 'assistant'.\n"
        prompt_text = prompt_preamble + _clean_messages_to_prompt(messages)
        if append_role is not None and len(append_role) > 0:
            prompt_text += f"\n{append_role}:"

        headers = {"Authorization": f"Bearer {self.apikey}"}

        response = requests.post(self.model_url, headers=headers, json={"inputs": prompt_text}).json()
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
        headers = {"Authorization": f"Bearer {self.apikey}"}
        response = requests.post(self.model_url, headers=headers, json={"inputs": prompt}).json()
        all_text = response[0]['generated_text']
        return all_text

class BloomWrapper():
    """
    Wrapper for Hugging Face's BLOOM model. Requires access to Hugging Face's inference API.
    """
    def __init__(self, apikey, model ):
        self.apikey = apikey

    def __repr__(self):
        return f"BloomWrapper()"

    def complete_chat(self, messages, append_role=None):
        """
        Mimicks a chat scenario with BLOOM, via a list of {"role": <str>, "content":<str>} objects.
        """

        prompt_preamble = "You are a friendly chat assistant. You are speaking to the 'user' below and will respond at the end, where it says 'assistant'.\n"
        prompt_text = prompt_preamble + _clean_messages_to_prompt(messages)
        if append_role is not None and len(append_role) > 0:
            prompt_text += f"\n{append_role}:"

        API_URL = "https://api-inference.huggingface.co/models/bigscience/bloom"
        headers = {"Authorization": f"Bearer {self.apikey}"}

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
        headers = {"Authorization": f"Bearer {self.apikey}"}

        response = requests.post(API_URL, headers=headers, json={"inputs": prompt}).json()
        all_text = response[0]['generated_text']
        new_text = all_text[len(prompt):]
        return new_text

class OpenAIGPTWrapper():
    """
    Wrapper for the OpenAI API. Supports all major text and chat completion models by OpenAI.
    """

    def __init__(self, apikey, model="gpt-3.5-turbo"):
        openai.api_key = apikey
        self.model = model

    def __repr__(self):
        return f"OpenAIGPTWrapper(model={self.model})"
    
    def complete_chat(self, messages, append_role=None):
        """
        Completes chat with OpenAI. If using GPT 3.5 or 4, will simply send the list of {"role": <str>, "content":<str>} objects to the API.

        If using an older model, it will structure the messages list into a prompt first.
        """

        if self.model.find('gpt-4') >= 0 or self.model.find('gpt-3.5') >= 0:

            response = openai.ChatCompletion.create(
                model=self.model,
                messages=messages
            )
            top_response_content = response['choices'][0]['message']['content']
            return top_response_content

        else:

            prompt_text = _clean_messages_to_prompt(messages)
            if append_role is not None and len(append_role) > 0:
                prompt_text += f"\n{append_role}: "
            prompt_text = prompt_text.strip()

            response = openai.Completion.create(
                model=self.model,
                prompt=prompt_text,
                stop=_get_stop_sequences_from_messages(messages)
            )

            top_response_content = response['choices'][0]['text']
            return top_response_content
    
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

class ClaudeWrapper():
    """
    Wrapper for Anthropic's Claude large language model.

    We've opted to call Anthropic's API directly rather than using their Python offering.
    """

    def __init__(self, apikey, model="claude-v1"):
        self.apikey = apikey
        self.model = model

    def __repr__(self):
        return f"ClaudeWrapper(model={self.model})"
    
    def complete_chat(self, messages, append_role=None):
        """
        Completes chat with Claude. Since Claude doesn't support a chat interface via API, we mimick the chat via the a prompt.
        """

        r_headers = {"X-API-Key":self.apikey, "Accept":"application/json"}

        prompt_text = _clean_messages_to_prompt(messages)
        if append_role is not None and len(append_role) > 0:
            prompt_text += f"\n{append_role}: "

        r_data = {"prompt": prompt_text,
                  "model": self.model,
                  "max_tokens_to_sample": 500,
                  "stop_sequences": _get_stop_sequences_from_messages(messages)
                }

        resp = requests.post("https://api.anthropic.com/v1/complete", headers=r_headers, json=r_data)
        completion = json.loads(resp.text)["completion"].strip()

        return completion
    
    def text_completion(self, prompt, stop_sequences=[]):
        """
        Completes text based on provided prompt.
        """

        r_headers = {"X-API-Key":self.apikey, "Accept":"application/json"}
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

    def __init__(self):
        self.model_name = "GPT-2"

    def __repr__(self):
        return f"GPT2Wrapper()"
    
    def complete_chat(self, messages, append_role=None, max_length=300):
        """
        Mimicks a chat scenario via a list of {"role": <str>, "content":<str>} objects. 
        """

        prompt_preamble = "You are a friendly chat assistant. You are speaking to the 'user' below and will respond at the end, where it says 'assistant'.\n"
        prompt_text = prompt_preamble + _clean_messages_to_prompt(messages)
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

class DollyWrapper():
    """
    Implementation of Dolly 2.0 (via Hugging Face).
    """

    def __init__(self):
        self.model_name = 'dolly-v2-12b'
        self.generate_text = pipeline(model="databricks/dolly-v2-12b", torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")

    def __repr__(self):
        return f"DollyWrapper(model={self.model})"

    def complete_chat(self, messages, append_role=None):
        """
        Mimicks a chat scenario via a list of {"role": <str>, "content":<str>} objects. 
        """

        prompt_preamble = "You are a friendly chat assistant. You are speaking to the 'user' below and will respond at the end, where it says 'assistant'.\n"

        prompt_text = prompt_preamble + _clean_messages_to_prompt(messages)
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

class CohereWrapper():
    """
    Wrapper for Cohere's API. Defaults to their 'xlarge' model.
    """

    def __init__(self, apikey, model="xlarge"):
        self.apikey = apikey
        self.model = model

    def __repr__(self):
        return f"CohereWrapper(model={self.model})"

    def complete_chat(self, messages, append_role=None):
        """
        Mimicks a chat scenario via a list of {"role": <str>, "content":<str>} objects. 
        """

        prompt_text = _clean_messages_to_prompt(messages)
        if append_role is not None and len(append_role) > 0:
            prompt_text += f"\n{append_role}:"

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

    def text_completion(self, prompt, stop_sequences=[]):
        """
        Completes text.
        """
        co = cohere.Client(self.apikey)
        response = co.generate(
            prompt=prompt,
            max_tokens=300, 
            stop_sequences=stop_sequences
        )
        resp = response.generations[0].text
        return resp

class ChatBot():
    """
    Allows you to have a chat conversation with an LLM wrapper.

    In short, it manages the list of {"role": <str>, "content":<str>} objects for you, so you don't have to figure this out. It also interacts directly with the model.
    """

    def __init__(self, llm, initial_system_prompt="You are a friendly chatbot assistant."):
        """
        Initializes a ChatBot. Provide an initial_system_prompt value to request the type of chatbot you will be dealing with.
        
        Warning: not all LLMs are trained to use instructions provided in a system prompt.
        """
        self.llm = llm 
        self.messages = []
        self._append_message('system', initial_system_prompt)

    def _append_message(self, role, message):
        """
        Saves a message to the chatbot's message queue.
        """
        self.messages.append({"role":role, "content":message})

    def chat(self, message):
        """
        Chats with the chatbot.
        """
        self._append_message('user', message)
        response = self.llm.complete_chat(self.messages, "assistant")
        self._append_message('assistant', response)
        return response
    
