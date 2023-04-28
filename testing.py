from phasellm.agents import * 
from phasellm.llms import OpenAIGPTWrapper

import os
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

news_api_api_key = os.getenv("NEWS_API_API_KEY")

query = "spacex"

n = NewsSummaryAgent(news_api_api_key, name="tester agent")
news_articles = n.getQuery(query, days_back=7, include_descriptions=False, max_articles=100)

o = OpenAIGPTWrapper(openai_api_key, model="gpt-4")

system = "You are a helpful news summarizer. We will provide you with a list of news articles and will ask that you summarize them and retain links to source by adding footnotes. For example, if you have a news article describing XYZ and URL to the article, you would discuss XYZ[1] and add '[1] URL' to the bottom of the message. Note that the footnotes should be counted as of the summary; you do not need to keep the numbers from the earlier order, just from your summary. In other words, footnotes should start at 1, 2, 3, etc..."
user_prompt = f"The articles below are about '{query}'. Please summarize them into a short paragraph with link retained as per the earlier instructions.\n\n{news_articles}"

messages = [{"role":"system", "content":system}, {"role":"user", "content":user_prompt}]

print(o.complete_chat(messages))