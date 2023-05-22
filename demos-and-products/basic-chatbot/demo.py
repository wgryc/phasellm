
import os
from dotenv import load_dotenv

from phasellm.llms import OpenAIGPTWrapper, ChatBot

load_dotenv()
MODEL_LLM = OpenAIGPTWrapper
MODEL_STRING = "gpt-4"
MODEL_API_KEY = os.getenv("OPENAI_API_KEY")
llm = MODEL_LLM(MODEL_API_KEY, MODEL_STRING)

CHATBOT = None 

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
    response = CHATBOT.chat(message)
    return {"status":"ok", "content":response,}

@APP.route('/resetchatbot')
def resetchatbot():
    if resetChatBot():
        return jsonify({"status":"ok", "message":"ChatBot has been restarted."})
    else:
        return jsonify({"status":"error", "message":"ChatBot could not be restarted."})

@APP.route('/')
def index():

    # Loop and print all args...
    #for key, value in request.args.items():
    #    print(f"{key} :: {value}")
    #print(request.args)

    if "reset" in request.args:
        if request.qrgs['reset'] == 'true':
            resetChatBot()

    return render_template('index.html')

def run(host="127.0.0.1", port=5000):
    """
    Launches a local web server for interfacing with PhaseLLM. This is meant to be for testing purposes only.
    """
    APP.run(host=host, port=port)

MAIN_HOST = "127.0.0.1"
MAIN_PORT = 8000
if __name__ == '__main__':
    run(MAIN_HOST, MAIN_PORT)