# Import all the data, apps, etc. we have built...
from apps import *

import os
from dotenv import load_dotenv

from phasellm.llms import OpenAIGPTWrapper, ChatBot, Prompt

load_dotenv()
MODEL_LLM = OpenAIGPTWrapper
#MODEL_STRING = "gpt-4"
MODEL_STRING = "gpt-3.5-turbo" # Use for speed.
MODEL_API_KEY = os.getenv("OPENAI_API_KEY")
llm = MODEL_LLM(MODEL_API_KEY, MODEL_STRING)

CHATBOT = None 

APP_PROMPT_STATE = 0
APP_CODE = None

from flask import Flask, request, render_template, jsonify

APP = Flask(__name__)

# We have a function because we'll eventually add other things, like system prompts, variables, etc.
# Returns True if successful, False otherwise
def resetChatBot():
    global CHATBOT 
    CHATBOT = ChatBot(llm)
    return True 

resetChatBot()

@APP.route('/submit_chat_message', methods = ['POST'])
def sendchat():
    global CHATBOT
    message = request.json["input"]
    response = process_message(message)
    return {"status":"ok", "content":response}

@APP.route('/resetchatbot')
def resetchatbot():
    if resetChatBot():
        return jsonify({"status":"ok", "message":"ChatBot has been restarted."})
    else:
        return jsonify({"status":"error", "message":"ChatBot could not be restarted."})

def process_message(message):
    global APP_PROMPT_STATE
    global APP_CODE
    global CHATBOT
    prompt = Prompt(APP_CODE["prompts"][APP_PROMPT_STATE]["prompt"])
    filled_prompt = prompt.fill(message = message)

    print(f"\n\n{filled_prompt}\n\n")

    response = CHATBOT.chat(filled_prompt)

    print(f"\n\n{response}\n\n")

    APP_PROMPT_STATE = APP_CODE["prompts"][APP_PROMPT_STATE]["next_prompt"]
    return response

@APP.route('/')
def index():

    global APP_PROMPT_STATE
    global APP_CODE

    # Loop and print all args...
    #for key, value in request.args.items():
    #    print(f"{key} :: {value}")
    #print(request.args)

    if "reset" in request.args:
        if request.args['reset'] == 'true':
            resetChatBot()

    app_name = ""
    system_message = ""
    if "app" in request.args:
        app_code = request.args['app']
        if app_code in APP_DATA_SETS:
            system_message = APP_DATA_SETS[app_code]["prompts"][0]["message"]
            app_name = app_code
            APP_PROMPT_STATE = 0
            APP_CODE = APP_DATA_SETS[app_code]
            APP_PROMPT_STATE = APP_DATA_SETS[app_code]["prompts"][0]["next_prompt"]

    return render_template('index.html', app_name=app_name, sys_msg=system_message)

def run(host="127.0.0.1", port=5000):
    """
    Launches a local web server for interfacing with PhaseLLM. This is meant to be for testing purposes only.
    """
    APP.run(host=host, port=port)

MAIN_HOST = "127.0.0.1"
MAIN_PORT = 8000
if __name__ == '__main__':
    run(MAIN_HOST, MAIN_PORT)