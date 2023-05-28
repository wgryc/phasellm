"""
ResearchLLM, a demo by PhaseLLM
See phasellm.com/researchllm for more information

The code below takes prompts uses a large language model (Claude, GPT-4, Cohere, etc.) to analyze your data and find insights.

To see the full workflow, review the comments in the run_analysis() function.
"""

from phasellm.llms import ChatBot, ClaudeWrapper, OpenAIGPTWrapper, CohereWrapper
from phasellm.exceptions import isAcceptableLLMResponse, LLMResponseException, LLMCodeException
from phasellm.agents import CodeExecutionAgent

# Load API keys
import os
from dotenv import load_dotenv
load_dotenv()

# ClaudeWrapper if using Anthropic
#MODEL_CLASS = ClaudeWrapper 
#MODEL_API_KEY = os.getenv("ANTHROPIC_API_KEY")
#MODEL_NAME = None

# OpenAIWrapper if using GPT-3.5 or GPT-4. We don't recommend older models.
MODEL_CLASS = OpenAIGPTWrapper
MODEL_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = "gpt-4"
#MODEL_NAME = "gpt-3.5-turbo"

# This is the ChatBot we'll be using and referncing throughout. It gets created using the class and API key above in start_bi_session()
CHATBOT = None 

# Used to print prompts and responses to screen
DEBUG = True

# We'll retry the prompt if there is an error in the code execution.
RETRY_PROMPT_ON_ERROR = True

# Imports made available to the LLM for data analysis
# Pandas is also used to load data to a data frame
import pandas as pd
import numpy as np 
import sklearn as sklearn
import scipy as scipy

# The data frame to be analyzed
df = None

# Helper function to set the data frame via other Python files (e.g., see frontend.py)
def set_df(new_df):
    global df 
    df = new_df

def start_bi_session(num_attempts = 3): 
    """
    Starts the chat sesion. Instantiates a ChatBot object with the LLM referenced in the MODEL object at the top of the file. Starts a chat session with the prompt below, which confirms the ChatBot knows what to do.
    
    We try to launch the LLM with a preamble to get it to follow our specific instructions. num_attempts is the number of times we try to get this to work before giving up.
    """
    
    global CHATBOT 
    global MODEL_CLASS
    global MODEL_API_KEY 
    global MODEL_NAME

    prompt="""You are a data science helper and will be working with me to build a model to explore a data set. You do not need to provide qualifiers like "As an AI model" because I know you are such a model. I want you to be as productive and concise as possible with me.

If you are generating code in a response, please limit your code generation to ONE (1) code block. If this means you need to add additional comments in the code, this is perfectly fine. Preceed every code block you generate with "|--START PYTHON CODE--|" and end each code block with "|--END PYTHON CODE--|".

Do you understand? Please simply write "yes" if you do, and "no" with followup questions if you do not."""
 
    launched_llm = False

    attempts = 0

    while not launched_llm and attempts < num_attempts:

        try:
            attempts += 1

            # If we the model takes no inputs...
            if MODEL_API_KEY is None and MODEL_NAME is None:
                model = MODEL_CLASS()
            elif MODEL_NAME is None:
                model = MODEL_CLASS(MODEL_API_KEY)
            else:
                model = MODEL_CLASS(MODEL_API_KEY, MODEL_NAME)

            print(f"Running ResearchLLM with {str(model)}")

            CHATBOT = ChatBot(model, prompt)
            response = CHATBOT.chat(prompt)

            if DEBUG: print(f"Understood? {response} (attempt #{attempts})") # Should print "yes" if the LLM understands...

            isAcceptableLLMResponse(response, "yes")
            launched_llm = True
        except LLMResponseException as e:
            if DEBUG: print(e)
            launched_llm = False

    if not launched_llm:
        print("\n\n*****WARNING*****\nLLM did not understand instructions. You might want to reset the model and avoid running ResearchLLM with this model..\n\n")
    
def ask_bi(msg):
    """
    Internal helper function. Generates code from the natural language query submitted.
    
    To see the full workflow, review the comments in the run_analysis() function.
    """
    response = CHATBOT.chat(msg + " Please do not plot anything, just provide the Python code. Include 'print' statements in your code so that whatever is printed to screen can be interpreted by the user.")
    if DEBUG: print(response)
    p_start = response.find("|--START PYTHON CODE--|")
    p_end = response.find("|--END PYTHON CODE--|")
    python_code = None
    if p_start >= 0 and p_end >= 0:
        python_code = response[(p_start + len("|--START PYTHON CODE--|")):p_end]
        python_code = python_code.replace("\\n", "\n")
        if DEBUG: print(python_code)
    return python_code

def ask_interpret(code):
    """
    This function takes Python code and executes it, saving the output to the 'code_output' variable. It then passes both the code and the output to an LLM to ask it to interpret the results.
    
    ResearchLLM uses this to execute analysis on the dataframe 'df', and then asks the LLM to interpet the results.
    """

    is_error = False

    try:
        agent = CodeExecutionAgent('code execution agent')
        code_output = agent.execute_code(code, globals(), locals())
    except LLMCodeException as e:
        is_error = True 
        code_output = e.exception_string 

    if not is_error:

        msg = f"""The following is the output of the code:
{code_output}

Could you please interpret this for me? Justify your answer by including the outputs of the analysis."""
        response = CHATBOT.chat(msg)
        if DEBUG: print(response)

    else:
        print("Error occurred when executing code.")
        response = "An error occurred while executing the code."

    return response, code_output, is_error
    
def run_analysis(message, is_retry=False):
    """
    Takes a message (i.e., natural language query) and uses the ChatBot object in this file to analyze the data given the natural language query. The function then returns the 'response_object', which contains the generated code, the raw output from the code, and an interpretation of the results.
    
    We also have a placeholder for error information. At this time, error messages from the Python code are included in the 'code_output' as technically they are the result of running the code.
    
    Here is how all the functions above work together:
    (1) User submits a run_analysis(message) where the message is a natural language query.
    (2) The message gets set to ask_bi(message), which gets converted into a proper LLM prompt. This then gets sent to the LLB ChatBot object, asking it to convert the natural language query to Python code.
    (3) If Python coe is returned, then we send the Python code to ask_interpret(python_code). This function runs the Python code, takes the resulting output, and generates a new prompt -- one that asks the LLM ChatBot object to interpet the original code and the output of the code.
    (4) The interpretation is then returned and included in the response_object.
    """
    response_object = {"code":"*No code generated.*", "code_output":"*No output.*", "interpretation":"*No interpretation.*", "error":"*No error.*"}
    python_code = ask_bi(message)
    if DEBUG: print(f"\n\nPYTHON CODE:\n{python_code}\n\n")
    if python_code is None:
        print("No code generated. Try again.")
        response_object["error":"No code generated."]
    else:
        response_object["code"] = python_code
        interpretation, code_output, is_error = ask_interpret(python_code)
        response_object["interpretation"] = interpretation
        response_object["code_output"] = code_output

        # We only retry once. We can probably try a few times, but we're being reasonable for now. :-)
        # Another approach could be to try a different model.
        if is_error and RETRY_PROMPT_ON_ERROR and not is_retry:

            if DEBUG: print("Error on first attempt. Trying again...")

            # One of the things we need to do is delete the last few messages of the ChatBot message history because the LLMs seem to get stuck on rewriting code they tried before. What we do is delete the last two messages in the ChatBot message history. These messages are the original prompt and the ChatBot's response.
            CHATBOT.messages = CHATBOT.messages[:-2]

            # We append the prior error to the prompt to try and avoid the error from taking place.
            new_message = message + "\n\nPlease be sensitive to avoiding the following error:\n{code_output}"

            return run_analysis(new_message, True)
        
    return response_object
