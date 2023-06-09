# Notes on Evaluations

There's a four-step process to testing these applications:
1. input data
2. prompt
3. execute
4. evaluate

We'll begin by very specifically exploring this from the perspective of newsbot.py

## Input Data

In this case, we have the following input data for each query:
(a) A purpose for the news bot. This is basically a higher-level prompt (e.g., system prompt) that stays the same within an experiment but might be optimized or changed across models or experiments.
(b) A query. This is the actual news topic we are asking to summarize. We have multiple queries per experiment.
(c) A list of articles with descriptions and links. This is generated by our agent.

## Prompt

There are two types of prompts, based on what we're doign so far: (1) text completion prompts, and (2) chat prompts.

A text completion prompt is our traditional approach to generating prompts. You have a set of instructions, and varibales will be replaced as needed (e.g., replace {query} with the topic of interest).

A chat prompt is different. Since a chat prompt has multiple messages, we might actually need to convert variables across the entire structure of chat. Today, we do not support chat prompts, but will need to do so for the news bot demo.

## Execute

This is the actual model execution loop. In this case, we take the input data and insert it into our prompts. Then we take those prompts and execute against models. We get the results and save them.

## Evaluation

Once all of the above has taken place, we then go ahead and review all the results. We want to do this in a 'blind peer review' approach where we randomize the order of outputs so we do not know which prompt/model combination is which.