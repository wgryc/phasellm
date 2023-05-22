# PhaseLLM Change Log

Have any questions? Need support? Please reach out on [Twitter (@phasellm)](https://twitter.com/phasellm) or via email: w (at) phaseai (dot) com

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
