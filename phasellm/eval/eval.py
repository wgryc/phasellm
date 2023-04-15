"""
Support for LLM evaluation.
"""

#TODO I don't like the way I'm structuring this stuff right now. I should actually run some experiments first.

from ..llms import OpenAIGPTWrapper

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

class HumanEvaluatorCommandLine():
    """
    Presents an objective, prompt, and two potential responses and has a human choose between the two.
    """

    def __repr__(self):
        return "<HumanEvaluatorComamndLine>"
    
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

# TODO Randomize the ordering so we avoid any bias from GPT-3.5.
class GPT35Evaluator():
    """
    Passes two model outputs to GPT-3.5 and has it decide which is the better output.
    """

    def __init__(self, apikey):
        self.model = OpenAIGPTWrapper(apikey)

    def __repr__(self):
        return f"GPT35Evaluator()"
    
    def choose(self, objective, prompt, response1, response2):
        """
        Presents the objective of a modeling task, a prompt, and then two responses. GPT-3.5 chooses the preference.
        """
        prompt = f"""We would like your feedback on a large language model we are building. Specifically, we would like you to compare two different LLM responses and let us know which one is better.

Our objective for the LLM is:
{objective}

The prompt we are using for the LLM is:
{prompt}

Here are the two pieces of generated text.

1: `{response1}`

2: `{response2}`

Please simply respond '1' or '2' as to which of the texts above address our earlier objective more effectively. Do not add any additional explanations, thoughts, punctuation, or anything; simply write '1' or '2'."""

        messages = [
            {"role":"system", "content":"You are an AI assistant helping with promp engineering and model evaluation."},
            {"role":"user", "content":prompt},
        ]

        response = self.model.complete_chat(messages, ['\n'])

        # ChatGPT has a preferences for adding "." to the end of the reply.
        if len(response) == 2:
            response = response[0]

        return response
