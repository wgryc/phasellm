# Import all the data, apps, etc. we have built...
from apps import *

import os
from dotenv import load_dotenv

from phasellm.llms import OpenAIGPTWrapper, ChatBot, Prompt

load_dotenv()
MODEL_LLM = OpenAIGPTWrapper
MODEL_STRING = "gpt-4"
#MODEL_STRING = "gpt-3.5-turbo" # Use for speed.
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

def parseResponse(r):
    lines = r.strip().split("\n")

    # Should eventually throw an error.
    if r[0:3] != "---":
        return None
    #assert r[0:3] == "---"

    var_name = None 
    v = ""

    rdict = {}

    for line in lines:
        if line[0:3] == "---":
            if var_name is not None:
                rdict[var_name] = v.strip()
            var_name = line[3:].strip().upper()
            v = ""
        else:
            v += line

    rdict[var_name] = v.strip()

    return rdict

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

def isInt(v):
    try:
        int(v)
    except:
        return False 
    return True 

def process_message(message):
    global APP_PROMPT_STATE
    global APP_CODE
    global CHATBOT
    prompt = Prompt(APP_CODE["prompts"][APP_PROMPT_STATE]["prompt"])
    filled_prompt = prompt.fill(message = message)

    print(f"\n\n{filled_prompt}\n\n")

    response = CHATBOT.chat(filled_prompt)

    print(f"\n\n{response}\n\n")

    response_dict = parseResponse(response)

    next_prompt = -1
    if isInt(APP_CODE["prompts"][APP_PROMPT_STATE]["next_prompt"]):
        next_prompt = APP_CODE["prompts"][APP_PROMPT_STATE]["next_prompt"]

    if response_dict is not None:
        print(response_dict)
        if "NEXT" in response_dict:
            if response_dict["NEXT"].upper() == "NO":
                response = "Chat is over!"
            else:
                if "RESPONSE" in response_dict:
                    response = response_dict["RESPONSE"]
        if "DANGER" in response_dict:
            if isInt(response_dict["DANGER"]):
                danger_score = int(response_dict["DANGER"])
                if danger_score > 80:
                    response = "Dangerous topic! Chat is over!"
                else:
                    if "RESPONSE" in response_dict:
                        response = response_dict["RESPONSE"]

    APP_PROMPT_STATE = next_prompt 

    return response

@APP.route("/")
def index():
    applist = ""
    for key in APP_DATA_SETS:
        applist += f"""
        <p><a href='/app?reset=true&app={APP_DATA_SETS[key]["code"]}'>{APP_DATA_SETS[key]["name"]}</a>
        """
    return render_template('applist.html', applist=applist)

@APP.route('/app')
def llmapp():

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

    return render_template('app.html', app_name=app_name, sys_msg=system_message)

def run(host="127.0.0.1", port=5000):
    """
    Launches a local web server for interfacing with PhaseLLM. This is meant to be for testing purposes only.
    """
    APP.run(host=host, port=port)

MAIN_HOST = "127.0.0.1"
MAIN_PORT = 8000
if __name__ == '__main__':
    run(MAIN_HOST, MAIN_PORT)