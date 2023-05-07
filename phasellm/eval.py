"""
Support for LLM evaluation.
"""

from .llms import OpenAIGPTWrapper

import random

class BinaryPreference():
    """
    Tracks a prompt, prompt variables, responses, and the calculated preference.
    """

    def __init__(self, prompt, prompt_vars, response1, response2):
        self.prompt = prompt 
        self.prompt_vars = prompt_vars 
        self.response1 = response1 
        self.response2 = response2 
        self.preference = -1

    def __repr__(self):
        return "<BinaryPreference>"
    
    def set_preference(self, pref):
        """
        Set the preference of the class.
        """
        self.preference = pref 

    def get_preference(self):
        """
        Get the preference of the class.
        """
        return self.preference 

class EvaluationStream():
    """
    Tracks human evaluation on the command line and records results.
    """

    def __init__(self, objective, prompt, models):
        """
        Initializes the stream.

        objective: what you are trying to do.
        prompt: the prompt you are using. Could be a summary thereof, too. We do not actively use this prompt in generating data for evaluation.
        models: an array of two models. These can be referenced later if need be, but are not necessary for running the evaluation workflow.
        """
        self.models = models
        self.objective = objective
        self.prompt = prompt
        self.objective = objective 
        self.evaluator = HumanEvaluatorCommandLine()
        self.prefs = [0]*len(models) # This will be a simple counter for now.

    def __repr__(self):
        return f"<EvaluationStream>"
    
    def evaluate(self, response1, response2):
        """
        Shows both sets of options for review and tracks the result.
        """
        pref = self.evaluator.choose(self.objective, self.prompt, response1, response2)
        self.prefs[pref - 1] += 1

class HumanEvaluatorCommandLine():
    """
    Presents an objective, prompt, and two potential responses and has a human choose between the two.
    """

    def __repr__(self):
        return "<HumanEvaluatorCommandLine>"
    
    def choose(self, objective, prompt, response1, response2):

        response_map = {"A": 1, "B": 2}
        response_a = response1
        response_b = response2
        if random.random() <= 0.5:
            response_map = {"A": 2, "B": 1}
            response_a = response2 
            response_b = response1

        output_string = f"""OBJECTIVE: {objective}

PROMPT: {prompt}

--------------------        
RESPONSE 'A':
{response_a}

--------------------
RESPONSE 'B':
{response_b}

--------------------
        """

        print(output_string)
        user_input = ""
        user_input = input()
        if user_input not in ["A", "B"]:
            print("Please put in 'A' or 'B' to tell us which is the better response.")
            user_input = input()

        return response_map[user_input]        

class GPTEvaluator():
    """
    Passes two model outputs to GPT-3.5 or GPT-4 and has it decide which is the better output.
    """

    def __init__(self, apikey, model="gpt-3.5-turbo"):
        self.model = OpenAIGPTWrapper(apikey, model=model)

    def __repr__(self):
        return f"GPT35Evaluator()"
    
    def choose(self, objective, prompt, response1, response2):
        """
        Presents the objective of a modeling task, a prompt, and then two responses. GPT-3.5/GPT-4 chooses the preference.
        """

        response_map = {"A": 1, "B": 2}
        response_a = response1
        response_b = response2
        if random.random() <= 0.5:
            response_map = {"A": 2, "B": 1}
            response_a = response2 
            response_b = response1

        prompt = f"""We would like your feedback on a large language model we are building. Specifically, we would like you to compare two different LLM responses and let us know which one is better.

Our objective for the LLM is:
{objective}

The prompt we are using for the LLM is:
{prompt}

Here are the two pieces of generated text.

A: `{response_a}`

B: `{response_b}`

Please simply respond 'A' or 'B' as to which of the texts above address our earlier objective more effectively. Do not add any additional explanations, thoughts, punctuation, or anything; simply write 'A' or 'B'."""

        messages = [
            {"role":"system", "content":"You are an AI assistant helping with prompt engineering and model evaluation."},
            {"role":"user", "content":prompt},
        ]

        response = self.model.complete_chat(messages, ['\n'])

        # ChatGPT has a knack for adding "." to the end of the reply.
        if len(response) == 2:
            response = response[0]

        choice = response_map[response]

        return choice
