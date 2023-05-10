### IMPORTS 

from phasellm.llms import OpenAIGPTWrapper, ClaudeWrapper, ChatPrompt
from phasellm.agents import NewsSummaryAgent
import json 

### ENVIRONMENT VARIABLES

import os
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
news_api_api_key = os.getenv("NEWS_API_API_KEY")

### SETUP THE EXPERIMENTAL DATA

queries = ['spacex', 'federal reserve', 'shopify', 'openai', 'biden', 'trump', 'met gala', 'king charles', 'poland', 'euro']
JSON_FILE = "news_articles.json"

llm_1 = OpenAIGPTWrapper(openai_api_key, model="gpt-4")
llm_2 = OpenAIGPTWrapper(openai_api_key, model="gpt-4") # ClaudeWrapper(anthropic_api_key)

chat_prompt_raw_1 = [
{"role":"system",
 "content": "You are a helpful news summarizer. We will provide you with a list of news articles and will ask that you summarize them and retain links to source by adding footnotes. For example, if you have a news article describing XYZ and URL to the article, you would discuss XYZ[1] and add '[1] URL' to the bottom of the message. Note that the footnotes should be counted as of the summary; you do not need to keep the numbers from the earlier order, just from your summary. In other words, footnotes should start at 1, 2, 3, etc..."},
{"role":"user",
 "content": "The articles below are about '{query}'. Please summarize them into a short paragraph with link retained as per the earlier instructions.\n\n{news_articles}"},
]

chat_prompt_raw_2 = [
{"role":"system",
 "content": "You are a helpful news summarizer. We will provide you with a list of news articles and will ask that you summarize them and retain links to source by adding footnotes. For example, if you have a news article describing XYZ and URL to the article, you would discuss XYZ[1] and add '[1] URL' to the bottom of the message. The footnote numbers should start at [1] and increase consecutively. In other words, footnotes should start at 1, 2, 3, etc. For the actual paragraph, you can reorder reference articles and choose the ones to include as to make the paragraph as informative, pithy, and concise as possible. You can also have multiple footnotes per sentence if this helps tell the story. While you should avoid adding your own commentary in most cases, feel free to do so if it will help the reader understand the context of the paragraph you are writing."},
{"role":"user",
 "content": "The articles below are about '{query}'. Please take on the role of an entertaining, successful, AI-driven investigative journalists and summarize them into a short paragraph. Make sure to follow the 'system' instructions.\n\n{news_articles}"},
]

chat_prompt_1 = ChatPrompt(chat_prompt_raw_1)
chat_prompt_2 = ChatPrompt(chat_prompt_raw_2)

### DATA HELPERS

def create_data_set(queries, json_file):
    article_dict = {}
    news_agent = NewsSummaryAgent(news_api_api_key, name="tester agent")
    for query in queries:
        news_articles = news_agent.getQuery(query, days_back=1, include_descriptions=True, max_articles=30)
        article_dict[query] = {"articles":news_articles}

    update_data_set(article_dict, json_file)

def update_data_set(dict_obj, json_file):
    with open(json_file, 'w') as writer:
        writer.write(json.dumps(dict_obj))

def load_data_set(json_file):
    articles = None
    with open(json_file, 'r') as reader:
        articles = json.loads(reader.read())
    return articles

### RUNNING DATA SET CREATION

create_data_set(queries, JSON_FILE)

articles = load_data_set(JSON_FILE)
for query, article_dict in articles.items():

    print(f"Generating news summary for '{query}'")

    print("... llm_1")
    llm_1_completion = llm_1.complete_chat(chat_prompt_1.fill(query=query, news_articles=article_dict['articles']))

    print("... llm_2")
    llm_2_completion = llm_2.complete_chat(chat_prompt_2.fill(query=query, news_articles=article_dict['articles']))
    
    # Saving results...
    article_dict["llm_1"] = llm_1_completion
    article_dict["llm_2"] = llm_2_completion
    articles[query] = article_dict

update_data_set(articles, JSON_FILE)