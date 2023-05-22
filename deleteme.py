from phasellm.exceptions import *
from phasellm.llms import *

import os
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
cohere_api_key = os.getenv("COHERE_API_KEY")


llm = OpenAIGPTWrapper(openai_api_key, model="gpt-4")

r = reviewOutputWithLLM("You are meh!", "Please only write in capital letters.", llm)
print(r)