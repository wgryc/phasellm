"""
Dolly 2.0 was announced and made available on April 12, 2023. See the announcement here: https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm

To illustrate how easy and quick it is to integrate new LLMs into PhaseLLM, we put together this Dolly implementation. Have fun!
"""

from transformers import pipeline
import torch

import llms

class DollyWrapper():

    def __init__(self):
        self.model_name = 'dolly-v2-12b'
        self.generate_text = pipeline(model="databricks/dolly-v2-12b", torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")

    def __repr__(self):
        return f"DollyWrapper(model={self.model})"

    def complete_chat(self, messages, append_role=None):

        prompt_preamble = "You are a friendly chat assistant. You are speaking to the 'user' below and will respond at the end, where it says 'assistant'."

        prompt_text = llms._clean_messages_to_prompt(messages)
        if append_role is not None and len(append_role) > 0:
            prompt_text += f"\n{append_role}:"

        resp = self.generate_text(prompt_text)

        return resp

    def text_completion(self, prompt):
        resp = self.generate_text(prompt)
        return resp
