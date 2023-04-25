""" RUNNING...
from frontend import *
run('0.0.0.0', 80)
"""

from researchgpt import *

from flask import Flask, request, render_template
import pandas as pd
import numpy as np

APP = Flask(__name__)

DATA_SETUP = [
    {"intro": "I am researching car crashes in NYC.", "file-loc":"../nypd-motor-vehicle-collisions.csv"},
    {"intro": "I am researching the relationship between income and sociodemographic census info.", "file-loc":"../incomes.csv"},
]
DATA_FIELD = 1

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

df = pd.read_csv(DATA_SETUP[DATA_FIELD]['file-loc'])
base_prompt = f"{DATA_SETUP[DATA_FIELD]['intro']} have imported Pandas as `pd`, Numpy as `np`, `scipy`, and `sklearn`, and have a dataframe called `df` loaded into Python. `df` contains the following variables and variable types:\n\n" + generateOverview(df) 
set_df(df)

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
    return render_template('index.html')

@APP.route("/text_completion", methods = ['POST'])
def analysis():
    text_to_complete = request.json["input"]
    new_request = base_prompt + text_to_complete
    response_object = run_analysis(new_request)
    return {"status":"ok", "content":response_object["interpretation"], "code":response_object["code"], "code_output":response_object["code_output"], "error":response_object["error"]}

def run(host="127.0.0.1", port=5000):
    """
    Launches a local web server for interfacing with PhaseLLM. This is meant to be for testing purposes only.
    """
    start_bi_session()
    #run_analysis(prompt_2)
    APP.run(host=host, port=port)
