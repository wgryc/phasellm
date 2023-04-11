import openai
import cohere
import requests
import json
import re

def _clean_messages_to_prompt(messages):
    out_text = "\n".join([f"{str(m['role'])}: {str(m['content'])}" for m in messages])
    return out_text

def _get_stop_sequences_from_messages(messages):
    roles = set()
    for m in messages:
        roles.add(m["role"])
    stop_sequences = [f"\n{r}:" for r in roles]
    return stop_sequences

class LanguageModelWrapper():
    def __init__(self):
        pass

    def __repr__(self):
        pass

    def complete_chat(self, messages):
        pass

    def text_completion(self, prompt, stop_sequences=[]):
        pass

class Prompt():

    def __init__(self, prompt):
        self.prompt = prompt

    def __repr__(self):
        return self.prompt

    def get_prompt(self):
        return self.prompt
    
    def fill_prompts(self, **kwargs):
        pattern = r'\{\s*[a-zA-Z0-9_]+\s*\}'
        matches = re.findall(pattern, self.prompt)
        new_prompt = self.prompt
        for m in matches:
            keyword = m.replace("{", "").replace("}", "").strip()
            if keyword in kwargs:
                new_prompt = new_prompt.replace(m, kwargs[keyword])
        return new_prompt

class BinaryPreference():

    def __init__(self, prompt, prompt_vars, response1, response2):
        self.prompt = prompt 
        self.prompt_vars = prompt_vars 
        self.response1 = response1 
        self.response2 = response2 
        self.preference = -1

    def __repr__(self):
        return "<BinaryPreference>"
    
    def set_preference(self, pref):
        self.preference = pref 

    def get_preference(self):
        return self.preference 
    
    def get_preference_response(self, pref_number):
        if pref_number == 1:
            return self.response1
        elif pref_number == 2:
            return self.response2
        else:
            return None 

class BinaryEvaluator():

    def __init__(self):
        self.preferences = []

    def __repr__(self):
        pass

    def choose(self, response1, response2):
        pass 

    def get_preference_history(self):
        return self.preferences

class GPT35Evaluator():

    def __init__(self, apikey):
        self.preference = []
        self.model = OpenAIGPTWrapper(apikey)

    def __repr__(self):
        return f"BinaryEvaluator using {str(self.model)}"
    
    def choose(self, objective, prompt, response1, response2):
        prompt = f"""We would like your feedback on a large language model we are building. Specifically, we would like you to compare two different LLM responses and let us know which one is better.

Our objective for the LLM is:
{objective}

The prompt we are using for the LLM is:
{prompt}

Here are the two pieces of generated text.

1: `{response1}`

2: `{response2}`

Please simply respond '1' or '2' as to which of the texts above address our earlier objective more effectively. Do not add any additional explanations, thoughts, punctuation, or anything; simply write '1' or '2'."""

        messages = [
            {"role":"system", "content":"You are an AI assistant helping with promp engineering and model evaluation."},
            {"role":"user", "content":prompt},
        ]

        response = self.model.complete_chat(messages, ['\n'])

        # ChatGPT has a preferences for adding "." to the end of the reply.
        if len(response) == 2:
            response = response[0]

        return response

class OpenAIGPTWrapper():

    def __init__(self, apikey, model="gpt-3.5-turbo"):
        openai.api_key = apikey
        self.model = model

    def __repr__(self):
        return f"OpenAIGPTWrapper(model={self.model})"
    
    def complete_chat(self, messages, append_role=None):

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
        if len(stop_sequences) == 0:
            response = openai.Completion.create(
                model=self.model,
                prompt=prompt.get_prompt()
            )
        else:
            response = openai.Completion.create(
                model=self.model,
                prompt=prompt.get_prompt(),
                stop = stop_sequences
            )
        top_response_content = response['choices'][0]['text']
        return top_response_content

class ClaudeWrapper():

    def __init__(self, apikey, model="claude-v1"):
        self.apikey = apikey
        self.model = model

    def __repr__(self):
        return f"ClaudeWrapper(model={self.model})"
    
    def complete_chat(self, messages, append_role=None):
        r_headers = {"X-API-Key":self.apikey, "Accept":"application/json"}

        prompt_text = _clean_messages_to_prompt(messages)
        if append_role is not None and len(append_role) > 0:
            prompt_text += f"\n{append_role}: "

        r_data = {"prompt": prompt_text,
                  "model": self.model,
                  "max_tokens_to_sample": 300, 
                  "stop_sequences": _get_stop_sequences_from_messages(messages)
                }

        resp = requests.post("https://api.anthropic.com/v1/complete", headers=r_headers, json=r_data)
        completion = json.loads(resp.text)["completion"].strip()

        return completion
    
    def text_completion(self, prompt, stop_sequences=[]):
        r_headers = {"X-API-Key":self.apikey, "Accept":"application/json"}
        r_data = {"prompt": prompt.get_prompt(),
                  "model": self.model,
                  "max_tokens_to_sample": 300, 
                  "stop_sequences": stop_sequences
                }

        resp = requests.post("https://api.anthropic.com/v1/complete", headers=r_headers, json=r_data)
        completion = json.loads(resp.text)["completion"].strip()
        return completion

class CohereWrapper():

    def __init__(self, apikey, model="xlarge"):
        self.apikey = apikey
        self.model = model

    def __repr__(self):
        return f"CohereWrapper(model={self.model})"

    def complete_chat(self, messages, append_role=None):

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
        co = cohere.Client(self.apikey)
        response = co.generate(
            prompt=prompt.get_prompt(),
            max_tokens=300, 
            stop_sequences=stop_sequences
        )
        resp = response.generations[0].text
        return resp

class ChatBot():

    def __init__(self, llm, initial_system_prompt="You are a friendly chatbot assistant."):
        self.llm = llm 
        self.messages = []

    def append_message(self, role, message):
        self.messages.append({"role":role, "content":message})

    def chat(self, message):
        self.append_message('user', message)
        response = self.llm.complete_chat(self.messages, "assistant")
        self.append_message('assistant', response)
        return response
