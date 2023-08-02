import os
from dotenv import load_dotenv

from phasellm.llms import ClaudeWrapper, ChatBot
from phasellm.agents import WebSearchAgent

from flask import Flask, request, render_template, jsonify

load_dotenv()
llm = ClaudeWrapper(os.getenv("ANTHROPIC_API_KEY"), model='claude-2')
web_search_agent = WebSearchAgent(
    api_key=os.getenv("GOOGLE_SEARCH_API_KEY")
)

CHATBOT: ChatBot

APP = Flask(__name__)


def reset_chatbot():
    """
    Reset the chatbot state.
    Returns:

    """
    global CHATBOT
    CHATBOT = ChatBot(llm)
    return True


# Call reset_chatbot() to initialize the chatbot.
reset_chatbot()


@APP.route('/submit-chat-message', methods=['POST'])
def route_send_chat():
    try:
        global CHATBOT
        message = request.json["input"]

        query = CHATBOT.chat(
            f'Come up with a google search query that will provide more information to help answer the question: '
            f'"{message}". Respond with only the query.'
        )
        print(f'Google search query: {query}')

        # Submit the query to the Google Search Agent.
        results = web_search_agent.search_google(
            query,
            custom_search_engine_id=os.getenv("GOOGLE_SEARCH_ENGINE_ID"),
            num=2
        )

        sources = []
        # Add the contents of the top result into the chatbot message queue.
        if len(results) >= 1:
            for result in results:
                CHATBOT.append_message(
                    role='search result',
                    message=result.content
                )
                sources.append(result.url)

        # Resubmit the message with the new search result as context.
        response = CHATBOT.chat(message + '. Answer using the information from the search results above.')

        return {"status": "ok", "content": response, "sources": sources}
    except Exception as e:
        return {"status": "error", "message": e}


@APP.route('/reset-chatbot')
def route_reset_chatbot():
    if reset_chatbot():
        return jsonify({"status": "ok", "message": "ChatBot has been restarted."})
    else:
        return jsonify({"status": "error", "message": "ChatBot could not be restarted."})


@APP.route('/')
def route_index():

    if "reset" in request.args:
        if request.args['reset'] == 'true':
            reset_chatbot()

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
