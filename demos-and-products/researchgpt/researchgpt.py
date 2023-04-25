from phasellm.llms import ChatBot, ClaudeWrapper

import os
from dotenv import load_dotenv

load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
MODEL = ClaudeWrapper(ANTHROPIC_API_KEY)
CHATBOT = ChatBot(MODEL, "")

DEBUG = True

import pandas as pd
import numpy as np 
import sklearn as sklearn
import scipy as scipy

df = None
#df = pd.read_csv("nypd-motor-vehicle-collisions.csv")
#df = pd.read_csv("incomes.csv")

def set_df(new_df):
    global df 
    df = new_df

def start_bi_session(): 
    prompt="""You are a data science helper and will be working with me to build a model to explore a data set. You do not need to provide qualifiers like "As an AI model" because I know you are such a model. I want you to be as productive and concise as possible with me.

If you are generating code in a response, please limit your code generation to ONE (1) code block. If this means you need to add additional comments in the code, this is perfectly fine. Preceed every code block you generate with "|--START PYTHON CODE--|" and end each code block with "|--END PYTHON CODE--|".

Do you understand? Please simply write "yes" if you do, and "no" with followup questions if you do not."""
    response = CHATBOT.chat(prompt)
    if DEBUG: print(f"Understood? {response}")
    assert response == "yes"
    
def ask_bi(msg):
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
    
# See: https://stackoverflow.com/questions/3906232/python-get-the-print-output-in-an-exec-statement    
import sys
from io import StringIO
import contextlib

@contextlib.contextmanager
def stdoutIO(stdout=None):
    old = sys.stdout
    if stdout is None:
        stdout = StringIO()
    sys.stdout = stdout
    yield stdout
    sys.stdout = old
    
def exec_code(code):
    is_error = False
    with stdoutIO() as s:
        try:
            exec(code)
        except Exception as err:
            print(f"Error occurred...\n{str(err)}")
            is_error = True
    return s.getvalue(), is_error
    
def ask_interpret(code):
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
