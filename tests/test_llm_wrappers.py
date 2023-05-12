import os
import re
import sys

sys.path.append(".")

from dotenv import load_dotenv

from phasellm.llms import ClaudeWrapper, CohereWrapper, LanguageModelWrapper, OpenAIGPTWrapper
from phasellm.prompts import ChatPrompt

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
NEWS_API_API_KEY = os.getenv("NEWS_API_API_KEY")

import anthropic
import cohere
import openai

openai.api_key = OPENAI_API_KEY

llm_openai_gpt = OpenAIGPTWrapper(name="llm_1_openai", api_key=OPENAI_API_KEY)
llm_cohere = CohereWrapper(name="llm_2_cohere", api_key=COHERE_API_KEY)
llm_claude = ClaudeWrapper(name="llm_3_claude", api_key=ANTHROPIC_API_KEY)

def test_cohere_wrapper():
    test_prompt = "Task:\nList three Scandinavian countries.Example:\n- Sweden\n-Finland\n-Norway\nAnswer:\n"
    completion = llm_cohere.text_completion(prompt=test_prompt)

    assert re.search(r"(Sweden|Finland|Norway)", completion)

def test_openai_gpt_wrapper():
    pass

def test_claude_wrapper():
    pass