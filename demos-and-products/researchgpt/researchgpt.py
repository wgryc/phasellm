"""
ResearchGPT, a demo by PhaseLLM
See phasellm.com for more information

The code below takes prompts uses a large language model (Claude, GPT-4, Cohere, etc.) to analyze your data and find insights.

This file specifically uses Claude (Anthropic's LLM) and was tested on Claude v1.3.

To see the full workflow, review the comments in the run_analysis() function.
"""

from phasellm.llms import ChatBot, ClaudeWrapper

# Load API keys
import os
from dotenv import load_dotenv
load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
MODEL = ClaudeWrapper(ANTHROPIC_API_KEY)
CHATBOT = ChatBot(MODEL, "")

# Used to print prompts and responses to screen
DEBUG = True

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

def start_bi_session(): 
    """
    Starts the chat sesion. Instantiates a ChatBot object with the LLM referenced in the MODEL object at the top of the file. Starts a chat session with the prompt below, which confirms the ChatBot knows what to do.
    """
    
    prompt="""You are a data science helper and will be working with me to build a model to explore a data set. You do not need to provide qualifiers like "As an AI model" because I know you are such a model. I want you to be as productive and concise as possible with me.

If you are generating code in a response, please limit your code generation to ONE (1) code block. If this means you need to add additional comments in the code, this is perfectly fine. Preceed every code block you generate with "|--START PYTHON CODE--|" and end each code block with "|--END PYTHON CODE--|".

Do you understand? Please simply write "yes" if you do, and "no" with followup questions if you do not."""
    response = CHATBOT.chat(prompt)
    if DEBUG: print(f"Understood? {response}") # Should print "yes" if the LLM understands...
    assert response == "yes" # ... otherwise will fail and error out.
    
def ask_bi(msg):
    """
    Internal helper function. Generates code from the natural language query submitted.
    
    To see the full workflow, review the comments in the run_analysis() function.
    """
    response = CHATBOT.chat(msg + " Please do not plot anything, just provide the Python code.")
    if DEBUG: print(response)
    p_start = response.find("|--START PYTHON CODE--|")
    p_end = response.find("|--END PYTHON CODE--|")
    python_code = None
    if p_start >= 0 and p_end >= 0:
        python_code = response[(p_start + len("|--START PYTHON CODE--|")):p_end]
        python_code = python_code.replace("\\n", "\n")
        if DEBUG: print(python_code)
    return python_code

# Imports to enable code execution and capturing outputs to a variable.
# See: https://stackoverflow.com/questions/3906232/python-get-the-print-output-in-an-exec-statement    
import sys
from io import StringIO
import contextlib

@contextlib.contextmanager
def stdoutIO(stdout=None):
    """
    Used to hijack printing to screen so we can save the Python code output for the LLM.
    """
    old = sys.stdout
    if stdout is None:
        stdout = StringIO()
    sys.stdout = stdout
    yield stdout
    sys.stdout = old
    
def exec_code(code):
    """
    Executes arbitrary Python code and saves the output (or error!) to a variable.
    
    Returns the variable and a boolean (is_error) depending on whether an error took place.
    
    To see the full workflow, review the comments in the run_analysis() function.
    """
    is_error = False
    with stdoutIO() as s:
        try:
            exec(code)
        except Exception as err:
            print(f"Error occurred...\n{str(err)}")
            is_error = True
    return s.getvalue(), is_error
    
def ask_interpret(code):
    """
    This function takes Python code and executes it, saving the output to the 'code_output' variable. It then passes both the code and the output to an LLM to ask it to interpret the results.
    
    ResearchGPT uses this to execute analysis on the dataframe 'df', and then asks the LLM to interpet the results.
    """
    code_output, is_error = exec_code(code)

    if not is_error:

        msg = f"""The following is the output of the code:
{code_output}

Could you please interpret this for me? Justify your answer by including the outputs of the analysis."""
        response = CHATBOT.chat(msg)
        if DEBUG: print(response)

    else:
        print("Error occurred when executing code.")
        response = "An error occurred while executing the code."

    return response, code_output
    
def run_analysis(message):
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
        interpretation, code_output = ask_interpret(python_code)
        response_object["interpretation"] = interpretation
        response_object["code_output"] = code_output
        
    return response_object
