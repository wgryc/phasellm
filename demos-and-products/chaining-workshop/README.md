# Chaining Workshop

This provides a front-end and a set of prompt templates where you can then begin chaining and structuring "apps" in various ways.

## Example Prompt Types

- System Message: show a message without any logic around what is shown.
- Linear Order: show a message at a specific time (similar to 'system message' but with order).
- Logic

## Sample Apps

- AmpUp.ai with a "yes/no" from the LLM
- AmpUp.ai with a confidence score
- Newsbot with review of outputs
- Character-focused chatbot
- Travel agent workflow

## Data Structure

{ prompt_id, prompt}
fallback prompt (i.e., error)

{ pid_1 -> pid_2, conditions}


## Characters

### Socrates

{ "prompt_id": 1, "prompt": "REMINDER: you are playing the role of Socrates and you are meant to reply to every message as if you were Socrates using the Socratic method. Please do so with the message below.\nMESSAGE:{message}", "next_prompt": 2}

{ "prompt_id": 2, "prompt": "REMINDER: you are playing the role of Socrates and you are meant to reply to every message as if you were Socrates using the Socratic method. Please do so with the message below.\nMESSAGE:{message}", "next_prompt": 2}



variables = user/app provided, LLM-provided

## How to Add Conditional Flows

- Output Parser: need to take the output of a model and parse it in some way. This should parse the outputs into specific variables.
- Pass a function to the next prompt? This will be limited, though -- you still need to write functions. Is that bad?
- Prebuilt template functions + custom functions.

Output Parser -> Environment Variable -> Function


OUTPUT PARSER



For all of your responses, please provide them in the following format:
---MESSAGE
This is where your actual message will go.
---SENTIMENT-SCORE
A score between 0 and 100 that shows how positive or negative the person's response was when describing their product.
---END
Include 'yes' or 'no' here. 'Yes' means we've asked 2 follow-up questions or the sentiment score has gotten close to 0 and you think it's safer to end the conversation. 'Yes' will continue the conversation.

## Conditional Flows v2

Right now, this is all hard-coded via " ---VAR" which is a poor way of doing things. 