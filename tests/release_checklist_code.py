"""
This code is used to test various aspects of PhaseLLM. We recommend running this on a P3 EC2 instance with Ubuntu 22.04 installed. To get this up and running, run the following code:

sudo apt-get update
sudo apt-get upgrade
sudo apt-get install xorg
sudo apt-get install nvidia-driver-460
sudo reboot

Run `nvidia-smi` to ensure you have GPU devices with CUDA installed.

"""

##########################################################################################
# GPU SETUP
#

import torch

# Confirm GPUs are installed and usable.
print(torch.cuda.is_available())
print(torch.cuda.current_device())

##########################################################################################
# ENVIRONMENT VARIABLES
#

# Load all environment variables and API keys

import os
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
cohere_api_key = os.getenv("COHERE_API_KEY")
hugging_face_api_key = os.getenv("HUGGING_FACE_API_KEY")

##########################################################################################
# GPT-3.5 EVALUATOR WITH COHERE AND CLAUDE COMPARISONS
#

# Run GPT-3.5 evaluator
from phasellm.eval import GPTEvaluator

# We'll use GPT-3.5 as the evaluator; this is the default setting in the class below
e = GPTEvaluator(openai_api_key)

# Our objective.
objective = "We're building a chatbot to discuss a user's travel preferences and provide advice."

# Chats that have been launched by users.
travel_chat_starts = [
    "I'm planning to visit Poland in spring.",
    "I'm looking for the cheapest flight to Europe next week.",
    "I am trying to decide between Prague and Paris for a 5-day trip",
    "I want to visit Europe but can't decide if spring, summer, or fall would be better.",
    "I'm unsure I should visit Spain by flying via the UK or via France."
]

from phasellm.llms import CohereWrapper, ClaudeWrapper
cohere_model = CohereWrapper(cohere_api_key)
claude_model = ClaudeWrapper(anthropic_api_key)

print("Running test. 1 = Cohere, and 2 = Claude.")
for tcs in travel_chat_starts:
    messages = [{"role":"system", "content":objective},
            {"role":"user", "content":tcs}]
    response_cohere = cohere_model.complete_chat(messages, "assistant")
    response_claude = claude_model.complete_chat(messages, "assistant")
    pref = e.choose(objective, tcs, response_cohere, response_claude)
    print(f"{pref}")
	
##########################################################################################
# DOLLY TESTS
#
	
from phasellm.llms import DollyWrapper
dw = DollyWrapper()

# Testing chat capability.
messages = [{"role":"user", "content":"What should I eat for lunch today?"}]
dw.complete_chat(messages, 'assistant')

# Run a text completion.
dw.text_completion("The capital of Poland is")
	
##########################################################################################
# GPT EVALUATOR WITH COHERE AND DOLLY COMPARISONS
#

import os
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
cohere_api_key = os.getenv("COHERE_API_KEY")

from phasellm.eval import GPTEvaluator

# We'll use GPT-3.5 as the evaluator.
e = GPTEvaluator(openai_api_key)

# Our objective.
objective = "We're building a chatbot to discuss a user's travel preferences and provide advice."

# Chats that have been launched by users.
travel_chat_starts = [
    "I'm planning to visit Poland in spring.",
    "I'm looking for the cheapest flight to Europe next week.",
    "I am trying to decide between Prague and Paris for a 5-day trip",
    "I want to visit Europe but can't decide if spring, summer, or fall would be better.",
    "I'm unsure I should visit Spain by flying via the UK or via France."
]

from phasellm.llms import CohereWrapper
from phasellm.llms import DollyWrapper # NEW: importing the DollyWrapper...
dw = DollyWrapper() # NEW: ... and instantiating it.

cohere_model = CohereWrapper(cohere_api_key)

print("Running test. 1 = Cohere, and 2 = Dolly.")
for tcs in travel_chat_starts:
    messages = [{"role":"system", "content":objective},
                {"role":"user", "content":tcs}]
    response_cohere = cohere_model.complete_chat(messages, "assistant")
    response_dw = dw.complete_chat(messages, "assistant") # NEW: minor change to variable name
    pref = e.choose(objective, tcs, response_cohere, response_dw)
    print(f"{pref}")

##########################################################################################
# HUGGINGFACE INFERENCE API TESTS
#

from phasellm.llms import HuggingFaceInferenceWrapper
hf = HuggingFaceInferenceWrapper(hugging_face_api_key, model_url='https://api-inference.huggingface.co/models/google/flan-t5-xxl')

# Testing chat capability.
messages = [{"role":"user", "content":"What should I eat for lunch today?"}]
hf.complete_chat(messages, 'assistant')

# Run a text completion.
hf.text_completion("The capital of Poland is")

##########################################################################################
# CHATBOT resend() TEST
#

from phasellm.llms import OpenAIGPTWrapper, ChatBot

oaiw = OpenAIGPTWrapper(openai_api_key, 'gpt-4')
cb = ChatBot(oaiw)
m = [{'role': 'system', 'content': "You are a robot that adds 'YO!' to the end of every sentence."}, {'role': 'user', 'content': 'Tell me about Poland.'}]
cb.messages = m
cb.resend()

##########################################################################################
# EVAL simulations TO EXCEL
#

from phasellm.llms import ChatBot, OpenAIGPTWrapper
from phasellm.eval import *

o = OpenAIGPTWrapper(openai_api_key)
c = ChatBot(o)
c.messages = [ {"role":"system", "content":"You're a mathbot."}, {"role":"user", "content":"What is 3*4*5*zebra?"} ]

x = simulate_n_chat_simulations(c, 4, 'responses.xlsx')