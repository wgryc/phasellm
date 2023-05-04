"""
A Flask frontend for the COT demo

To run, start a Python REPL and in the same directory as this file and run the following:
> from frontend import *
> run() # Or, run('0.0.0.0', 80)

"""

from flask import Flask, request, render_template
import pandas as pd
import numpy as np

from researchllm import *

APP = Flask(__name__)

##########################################################################
#
# DATA SET SETUP (START)
# Please review the code below to set up your own data set for analysis.
#

# Data set to load and analyze.
DATA_SETUP_INTRO = "I am researching the relationship between income and sociodemographic census info."
DATA_FILE_LOC = "incomes.csv"

# Another sample we explored.
#DATA_SETUP_INTRO = "I am researching car crashes in NYC."
#DATA_FILE_LOC = "nypd-motor-vehicle-collisions.csv"

# Want to analyze your own data set? Simply replace the two variables above:
# DATA_SETUP_INTRO = "What are you researching? Please provide a short description.
# DATA_FILE_LOC = "The location of the CSV file."
# Note that you DO NOT have to provide metadata about the CSV file. This gets generated automatically.

# Loads the CSV file.
# If you want to load another file (e.g., Excel file), replace the code below with the relevant function (e.g., read_excel()).
df = pd.read_csv(DATA_FILE_LOC)

#
# DATA SET SETUP (END)
#
##########################################################################

def generateOverview(df):
    """
    Generates a prompt providing an overview of a data set. This should only be used to generate the initial data prompt for now.
    """
    description = ""
    for column in df:
        col_name = df[column].name
        col_type = df[column].dtype
        col_description = f"Column Name: {col_name}\nColumn Type: {col_type}"
        if col_type == "object":
            column_values = df[col_name].values
            uniques = np.unique(column_values)
            col_description += f"\nSample Values: {str(uniques)}"
        description += col_description + "\n\n"
    return description.strip()

# The prompt used to set up the entire chat session. This prompt is used regularly for analysis.
base_prompt = f"{DATA_SETUP_INTRO} I have imported Pandas as `pd`, Numpy as `np`, `scipy`, and `sklearn`, and have a dataframe called `df` loaded into Python. `df` contains the following variables and variable types:\n\n" + generateOverview(df) 

# Calls the researchllm.py function to set the current dataframe as the main one for analysis.
set_df(df)
start_bi_session()

##########################################################################
#
# FLASK FUNCTIONS
# Everything below manages the frontend.
#
##########################################################################

@APP.route('/get_prompt')
def get_prompt():
    """
    Returns a JSON object with the prompt being passed on to the language model.
    """
    return {"status":"ok", "prompt":base_prompt}

@APP.route('/')
def index():
    """
    Displays the index page accessible at '/'
    """
    return render_template('interface01.html')

@APP.route("/runcode", methods = ['POST'])
def runcode():
    """
    Runs code in the POST request.
    """
    code_to_run = request.json['code']
    response, code_output, is_error = ask_interpret_clean(code_to_run)
    return {"response":response, "code_output":code_output, "is_error":is_error}

@APP.route("/text_completion", methods = ['POST'])
def analysis():
    """
    Calls the researchllm.py code to request analysis and interpretation thereof.
    
    See run_analysis(message) in researchllm.py for more information.
    """
    text_to_complete = request.json["input"]
    new_request = base_prompt + text_to_complete
    response_object = run_analysis(new_request)
    return {"status":"ok", "content":response_object["interpretation"], "code":response_object["code"], "code_output":response_object["code_output"], "error":response_object["error"]}

def run(host="127.0.0.1", port=5000):
    APP.run(host=host, port=port)
