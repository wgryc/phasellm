# PhaseLLM

Large language model evaluation and workflow framework from [Phase AI](https://phaseai.com/).

- [Follow us on Twitter](https://twitter.com/phasellm) for updates.
- [Star us on GitHub](https://github.com/wgryc/phasellm).
- [Read the Docs](https://phasellm.com/docs/) -- currently limited to the module reference. Tutorials and code examples are below.

## Installation

You can install PhaseLLM via pip:

```
pip install phasellm
```

Installing from PyPI does not download the sample demos and products in the `demos-and-products` folder. Clone this repository and follow instructions in the `README.md` file in each product folder to run those.

## Introduction

The coming months and years will bring thousands of new products and experienced powered by large language models (LLMs) like ChatGPT or its increasing number of variants. Whether you're using OpenAI's ChatGPT, Anthropic's Claude, or something else all together, you'll want to test how well your models and prompts perform against user needs. As more models are launched, you'll also have a bigger range of options.

PhaseLLM is a framework designed to help manage and test LLM-driven experiences -- products, content, or other experiences that product and brand managers might be driving for their users.

Here's what PhaseLLM does:
1. We standardize API calls so you can plug and play models from OpenAI, Cohere, Anthropic, or other providers.
2. We've built evaluation frameworks so you can compare outputs and decide which ones are driving the best experiences for users.
3. We're adding automations so you can use advanced models (e.g., GPT-4) to evaluate simpler models (e.g., GPT-3) to determine what combination of prompts yield the best experiences, especially when taking into account costs and speed of model execution.

PhaseLLM is open source and we envision building more features to help with model understanding. We want to help developers, data scientists, and others launch new, robust products as easily as possible.

If you're working on an LLM product, please reach out. We'd love to help out.

## Example: Evaluating Travel Chatbot Prompts with GPT-3.5, Claude, and more

PhaseLLM makes it incredibly easy to plug and play LLMs and evaluate them, in some cases with *other* LLMs. Suppose you're building a travel chatbot, and you want to test Claude and Cohere against each other, using GPT-3.5. 

What's awesome with this approach is that (1) you can plug and play models and prompts as needed, and (2) the entire workflow takes a small amount of code. This simple example can easily be scaled to much more complex workflows.

So, time for the code... First, load your API keys.

```python
import os
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
cohere_api_key = os.getenv("COHERE_API_KEY")
```

We're going to set up the *Evaluator*, which takes two LLM model outputs and decides which one is better for the objective at hand.

```python
from phasellm.eval import GPT35Evaluator

# We'll use GPT-3.5 as the evaluator.
e = GPT35Evaluator(openai_api_key)
```

Now it's time to set up the experiment. In this case, we'll set up an `objective` which describes what we're trying to achieve with our chatbot. We'll also provide 5 examples of starting chats that we've seen with our users.

```python
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
```

Now we set up our Cohere and Claude models.

```python
from phasellm.llms import CohereWrapper, ClaudeWrapper
cohere_model = CohereWrapper(cohere_api_key)
claude_model = ClaudeWrapper(anthropic_api_key)
```

Finally, we launch our test. We run an experiments where both models generate a chat response and then we have GPT-3.5 evaluate the response.

```python
print("Running test. 1 = Cohere, and 2 = Claude.")
for tcs in travel_chat_starts:

    messages = [{"role":"system", "content":objective},
            {"role":"user", "content":tcs}]

    response_cohere = cohere_model.complete_chat(messages, "assistant")
    response_claude = claude_model.complete_chat(messages, "assistant")

    pref = e.choose(objective, tcs, response_cohere, response_claude)
    print(f"{pref}")
```

In this case, we simply print which of the two models was preferred.

Voila! You've got a suite to test your models and can plug-and-play three major LLMs.

## Contact Us

If you have questions, requests, ideas, etc. please reach out at w (at) phaseai (dot) com.
