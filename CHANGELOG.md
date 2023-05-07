# PhaseLLM Change Log

Have any questions? Need support? Please reach out on [Twitter (@phasellm](https://twitter.com/phasellm) or via email: w (at) phaseai (dot) com

# 0.0.6 (2023-05-xx)

### New Features

- Added HuggingFaceInferenceWrapper so you can now query models via HuggingFace

### Fixes

_None_

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
