# PhaseLLM Change Log

Have any questions? Need support? Please reach out on [Twitter (@phasellm)](https://twitter.com/phasellm) or via email: w (at) phaseai (dot) com

# 0.0.20 (2023-11-20)


### New Features

- `ReplicateLlama2Wrapper` to enable you to use Llama 2 via [Replicate](https://replicate.com/)
- Note that the `ChatBot` class supports the `ReplicateLlama2Wrapper`, so you can plug and play the Llama 2 model just like any other chat models; same goes for text completions

### Fixes

- _None._

# 0.0.19 (2023-11-09)

### New Features

- `phasellm.logging` comes with `PhaseLogger`, a class that allows you to automatically send your chats to evals.phasellm.com for visual, no-code reviews later
- PhaseLLM requests now contain the header information received from the APIs they are calling; you can review whatever information OpenAI, Anthropic, etc. are sending you

### Fixes

- `Claude 2` support is back! Older versions weren't parsing responses properly
- Support for versions 1.x of OpenAI's `openai` Python SDK

# 0.0.18 (2023-10-16)

### New Features

- `PhaseLLM Evaluations` project, a Django-powered front-end for evaluating LLMs and running batch jobs
- Added phasellm.html.chatbotToJson() function to enable easy exporting of chatbot messages to JSON

### Fixes

_None._

# 0.0.17 (2023-08-19)

### New Features

_None._

### Fixes

- Fixing backwards compatibility issue with new API configuration options.

# 0.0.15 and 0.0.16 (2023-08-15)

### New Features

- New RSS agent
  - Crawl and read RSS feeds with LLMs
  - Demo project in `/demos-and-products/arxiv-assistant`
- Support for OpenAI Azure implementations; use our new `AzureAPIConfiguration` class

### Fixes

- Adding support for `claude-2` due to Anthropic API changes
- Fix for user agent when running website crawls

# 0.0.13 and 0.0.14 (2023-07-15)

### New Features

- New agents:
  - `WebpageAgent` for scraping HTML from web pages (+ extracting text)
  - `WebSearchAgent` for using Google, Brave, and other search APIs to run queries for you
- New demo project: `web-search-chatbot`! This shows how you can use the new agents above in chatbot applications

### Fixes

- Installation for `phasellm` (i.e., default installation) includes `transformers` to avoid errors
- Installation option for `phasellm[complete]` to enable installing packages for running LLMs locally. The default setup will only provide support for LLM APIs (e.g., OpenAI, Anthropic, Cohere, Hugging Face)

# 0.0.12 (2023-06-30)

### New Features

_None_

### Fixes

- ChatPrompt fills were losing additional data (e.g., time stamps); this is now fixed.

# 0.0.11 (2023-06-19)

### New Features

- All LLMs now support `temperature` setting
- All LLMs now accept `kwargs` that they pass on to their various APIs
- `phasellm.llms.swap_roles()` helper funtion where `user` and `assistant` get swapped. Extremely useful for testing/evaluations and running simulations
- `phasellm.eval.simulate_n_chat_simulations()` allows you to resend the same chat history multiple times to generate an array of responses (for testing purposes)

### Fixes

- Fixed Server Side Event streaming bug where new lines weren't being escaped properly.
- Updating `requirements.txt` and `setup.py` to include all new relevant packages.

# 0.0.10 (2023-06-12)

### New Features

- We now have support for streaming LLMs
- `StreamingLanguageModelWrapper` allows you to build your own streaming wrappers
- `StreamingOpenAIGPTWrapper` for GPT-4 and GPT-3.5
- `StreamingClaudeWrapper` for Claude (Anthropic)
- Significant additions to tests, to ensure stability of future releases (see `tests` folder)
- ChatBot class supports both streaming and non-streaming LLMs, so you can plug and play with either

### Fixes

- Starting to add type hints to our code; this will take a bit of time but let us know if you have questions

# 0.0.9 (2023-06-04)

### New Features

_None_

### Fixes

- Hotfix for ChatBot chat() function to remove new fields (e.g., time stamp) when making API calls

# 0.0.8 (2023-06-04)

### New Features

- ChatBot `messages` stack now also tracks the timestamp for when a message was sent, and how long it took to process in the case of external API calls
- HTML module that enables HTML outputs for ChatBot

### Fixes

_None_

# 0.0.7 (2023-05-27)

### New Features

- ChatBot now has a `resend()` function to redo the last chat message in case of errors or if building message arrays outside of a bot
- Newsbot now has sample Claude code as well (the 100K token model is a fantastic model for news bots)
- Demo project: "chaining workshop" -- we'll be exploring unique ways to build prompt chains soon
- Demo project: basic chatbot. Use this as a base for other projects

### Fixes

- ClaudeWrapper bug fix: appending "Assistant:" to chats by default.
- Reverted `requirements.txt` to earlier version (v0.0.5)

# 0.0.6 (2023-05-08)

_Note:_ a number of changes in this release are not backwards compatible. They contain a '🚨' emoji by the bullet point in case you want to review.

### New Features

- Lots of new classes!
  - LLMs: added _HuggingFaceInferenceWrapper_ so you can now query models via HuggingFace
  - Data: added _ChatPrompt_ to build chat sessions with variables
  - Evaluation: added _EvaluationStream_ to make it easy to evaluate models
  - Exceptions: added _ChatStructureException_ to be called when a chat doesn't follow OpenAI's messaging requirements
- phasellm.eval has _isProperlyStructuredChat()_ to validate 

### Fixes

- 🚨 Changed _fill_prompt()_ to _fill()_ so we are consistent across _Prompt_ and _ChatPrompt_ classes
- 🚨 _GPT35Evaluator_ is now _GPTEvaluator_ since you can use GPT-4 as well; the evaluation approach randomizes the order in which options are presented to the LLM to avoid any bias it might have
- Fixes to _ResearchLLM_
  - _generateOverview()_ now limits examples for categorical variables to 10, though this can also be set at the top of the file to another #. Previously we'd include all possible values.
  - Making a list of categorical values in _generateOverview()_ often errored out. This has been fixed.

# 0.0.5 (2023-05-01)

### New Features

- New agents
  - EmailSenderAgent for sending emails (tested on GMail)
  - NewsSummaryAgent for newsapi.org; summarizes lists of news articles
- Demo projects
  - 'News Bot' demo that uses the new agents above to email daily summaries of news topics
  - 'Chain of Thought' demo that generats a markdown file with plans for how to analyze a data set

### Fixes

_None_

# 0.0.4 (2023-04-27)

### New Features

- Added Exceptions submodule to track specific errors/issues with LLM workflows
  - LLMCodeException for errors with LLM-generated code
  - LLMResponseException to ensure an LLM responds properly (i.e., from a list of potential responses)
- Added Agents submodule to enable autonomous agents and task execution
  - CodeExecutionAgent for executing LLM-generated code

### Fixes

- ResearchGPT will retry requests if code generated causes an error
- ResearchGPT code now includes examples for Claude and GPT-4
- Added python-dotenv to requirements
- Fixed folder structure in phasellm source code (removed subfolders for submodules)

# 0.0.3 (2023-04-24)

### New Features

- ResearchGPT (demo product)

### Fixes

- Added max # of tokens to ClaudeWrapper (required by Anthropic API)

# 0.0.2 (2023-04-24)

### New Features

- Dolly 2.0 wrapper
- BLOOM wrapper

### Fixes

- ChatBot bug where messages were erroring out

# 0.0.1 (🥳 First release!)

### New Features

- Model Support
  - GPT (3, 3.5, 4)
  - Claude
  - Cohere
  - HuggingFace (GPT-2)
  - Dolly 2.0 (via HuggingFace)
- Evaluation
  - GPT 3.5 evaluator
  - Human evaluation

### Fixes

Nothing, since this is a new release!
