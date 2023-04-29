from phasellm.agents import EmailSenderAgent, NewsSummaryAgent
from phasellm.llms import OpenAIGPTWrapper

import os
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
news_api_api_key = os.getenv("NEWS_API_API_KEY")

gmail_email = os.getenv("GMAIL_EMAIL")
gmail_password = os.getenv("GMAIL_PASSWORD")

query = "first republic bank"

n = NewsSummaryAgent(news_api_api_key, name="tester agent")
news_articles = n.getQuery(query, days_back=1, include_descriptions=True, max_articles=30)

o = OpenAIGPTWrapper(openai_api_key, model="gpt-4")

system = "You are a helpful news summarizer. We will provide you with a list of news articles and will ask that you summarize them and retain links to source by adding footnotes. For example, if you have a news article describing XYZ and URL to the article, you would discuss XYZ[1] and add '[1] URL' to the bottom of the message. Note that the footnotes should be counted as of the summary; you do not need to keep the numbers from the earlier order, just from your summary. In other words, footnotes should start at 1, 2, 3, etc..."
user_prompt = f"The articles below are about '{query}'. Please summarize them into a short paragraph with link retained as per the earlier instructions.\n\n{news_articles}"

messages = [{"role":"system", "content":system}, {"role":"user", "content":user_prompt}]

news_message = o.complete_chat(messages)
news_subject = f"News about: {query}"

e = EmailSenderAgent('Wojciech Gryc', 'smtp.gmail.com', gmail_email, gmail_password, 587)
e.sendPlainEmail('wgryc@fastmail.com', news_subject, news_message)