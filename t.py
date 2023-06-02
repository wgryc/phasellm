import os
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
cohere_api_key = os.getenv("COHERE_API_KEY")

from pprint import pprint

import phasellm.html as h
from phasellm.llms import ClaudeWrapper, ChatBot

claude_model = ClaudeWrapper(anthropic_api_key)
cb = ChatBot(claude_model)

print(cb.chat("Hey Claude, how's the weather up there?"))
print(cb.chat("What is 3x3+1?"))

pprint(cb.messages)

h.toHtmlFile(h.chatbotToHtml(cb), 't.html')