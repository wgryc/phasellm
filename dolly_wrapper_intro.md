# Welcoming Dolly 2.0 with a Wrapper and Tests
*By Wojciech Gryc*

We're excited to hear about the [Dolly 2.0](https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm). This sort of open source, openly available, locally runnable LLM is what's necessary to enable enterprise applications.

So, should you use Dolly (e.g., [via HuggingFace](https://huggingface.co/databricks/dolly-v2-12b))? Does it perform better than your other prompts?

This is exactly why we're building PhaseLLM.

## Using DollyWrapper in PhaseLLM

We're keeping the wrapper code separate in [llmsdolly.py] for now, as we need to standardize our package. This code also uses `PyTorch` and `transformers` so we can appreciate why you might not use this right away (see our *next steps*, below).

Note that we were running this on a p3.8xlarge Ubuntu 22.04 EC2 instance.

First, some simple code to show it all works nicely!

```python
from llmsdolly import DollyWrapper
dwl = DollyWrapper()

# Testing chat capability.
messages = [{"role":"user", "content":"What should I eat for lunch today?"}]
dw.complete_chat(messages, 'assistant')

# Run a text completion.
dw.text_completion("The capital of Poland is")
```

Now, let's use GPT-3.5 to evaluate whether Dolly performs better than Cohere in the travel task in our [README.md]. I've called out the new lines of code with `# NEW` comments.

```python
import os
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
cohere_api_key = os.getenv("COHERE_API_KEY")

import llms # The PhaseLLM module; a temporary name for now

# We'll use GPT-3.5 as the evaluator.
e = llms.GPT35Evaluator(openai_api_key)

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

from llmsdolly import DollyWrapper # NEW: importing the DollyWrapper...
dwl = DollyWrapper() # NEW: ... and instantiating it.

cohere_model = llms.CohereWrapper(cohere_api_key)

print("Running test. 1 = Cohere, and 2 = Dolly.")
for tcs in travel_chat_starts:
    messages = [{"role":"system", "content":objective},
                {"role":"user", "content":tcs}]
    response_cohere = cohere_model.complete_chat(messages, "assistant")
    response_dw = dw.complete_chat(messages, "assistant") # NEW: minor change to variable name
    pref = e.choose(objective, tcs, response_cohere, response_dw)
    print(f"{pref}")
```

So, new lines of code and you can rerun everything you did before, but with Dolly 2.0!

## Next Steps

We've gotten a lot of great feedback from the NLP and dev community and will be working closely on our framework here.
- [Follow us on Twitter](https://twitter.com/PhaseLLM) to get updates.
- Star or watch this repository.
- We'll be cleaning up the code and adding documentation in the coming days!

As always, email me at w (at) phaseai (dot) com if you have questions or want to collaborate.
