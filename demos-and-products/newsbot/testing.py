"""
In this file, we compare Claude and GPT-4 across 10 different news summaries to see which one is better.
"""


"""
There's a four-step process to testing these applications:
1. input data
2. prompt
3. execute
4. evaluate

We'll begin by very specifically exploring this from the perspective of newsbot.py

INPUT DATA

In this case, we have the following input data for each query:
(a) A purpose for the news bot
(b) A query
(c) A list of articles with descriptions and links

PROMPT

Next, we have a prompt that takes the three data points above and sends them off to the LLM.

EXECUTE

This is simple; we execute the call and get a result.

EVALUATE

Here is where we evaluate everything. This is where we decide which of the two (or more) models we prefer or pick.

Note that we might be testing multiple things here: the prompt, or the model, or potentially both.

NOTES

I think the above would be easiest to do via front-end. Doing this in a command line is very, very tricky in my opinion. The other option would be to do it all behind the scenes with another LLM.

"""

# Note: Claude won, 6 to 4...

PICKLE_FILE = "data.pickle"

import pickle 

from phasellm.agents import NewsSummaryAgent
from phasellm.eval import GPT35Evaluator, HumanEvaluatorCommandLine
from phasellm.llms import OpenAIGPTWrapper, ClaudeWrapper, Prompt 

import os
from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
news_api_api_key = os.getenv("NEWS_API_API_KEY")

class EvaluationStream():

    def __init__(self, objective, prompt, models):
        self.models = models
        self.objective = objective
        self.prompt = prompt # TODO Should include the templated variables instead...? Or the Prompt object...?
        self.objective = objective 
        self.evaluator = HumanEvaluatorCommandLine()
        self.prefs = [0]*len(models) # This will be a simple counter for now.

    def __repr__(self):
        return f"<EvaluationStream>"
    
    def evaluate(self, response1, response2): # TODO Issue here is that options 1 and 2 aren't tied to models. Clean up.
        pref = self.evaluator.choose(self.objective, self.prompt, response1, response2)
        self.prefs[pref - 1] += 1
    
def getArticlesAndSummarize(llm, query, news_articles):

    # Set up messages for summarization.
    system = "You are a helpful news summarizer. We will provide you with a list of news articles and will ask that you summarize them and retain links to source by adding footnotes. For example, if you have a news article describing XYZ and URL to the article, you would discuss XYZ[1] and add '[1] URL' to the bottom of the message. Note that the footnotes should be counted as of the summary; you do not need to keep the numbers from the earlier order, just from your summary. In other words, footnotes should start at 1, 2, 3, etc..."
    user_prompt = f"The articles below are about '{query}'. Please summarize them into a short paragraph with link retained as per the earlier instructions.\n\n{news_articles}"
    messages = [{"role":"system", "content":system}, {"role":"user", "content":user_prompt}]

    news_message = llm.complete_chat(messages)

    return news_message

def create_data_set(queries, pickle_file):
    article_dict = {}
    news_agent = NewsSummaryAgent(news_api_api_key, name="tester agent")
    for query in queries:
        news_articles = news_agent.getQuery(query, days_back=1, include_descriptions=True, max_articles=30)
        article_dict[query] = {"articles":news_articles}

    with open(pickle_file, 'wb') as handle:
        pickle.dump(article_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

def update_data_set(dict_obj, pickle_file):
    with open(pickle_file, 'wb') as handle:
        pickle.dump(dict_obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_data_set(pickle_file):
    articles = None
    with open(pickle_file, 'rb') as handle:
        articles = pickle.load(handle)
    return articles

# Only need to run this once.
queries = ['spacex', 'federal reserve', 'shopify', 'openai', 'biden', 'trump', 'met gala', 'king charles', 'poland', 'euro']
#create_data_set(queries, PICKLE_FILE)

llm_1 = OpenAIGPTWrapper(openai_api_key, model="gpt-4")
llm_2 = ClaudeWrapper(anthropic_api_key)

"""articles = load_data_set(PICKLE_FILE)
for key, article_dict in articles.items():

    print(f"Generating news summary for '{key}'")

    print("... llm_1")
    llm_1_completion = getArticlesAndSummarize(llm_1, key, article_dict['articles'])

    print("... llm_2")
    llm_2_completion = getArticlesAndSummarize(llm_1, key, article_dict['articles'])

    article_dict["llm_1"] = llm_1_completion
    article_dict["llm_2"] = llm_2_completion

    articles[key] = article_dict
"""
    
#update_data_set(articles, PICKLE_FILE)

articles = load_data_set(PICKLE_FILE)

# TODO Not saving the actual original data, vars, etc. to propery input into the evuator.
es = EvaluationStream("Which news summary is higher quality and more engaging?", "You are a helpful news summarizer. We will provide you with a list of news articles and will ask that you summarize them and retain links to source by adding footnotes. For example, if you have a news article describing XYZ and URL to the article, you would discuss XYZ[1] and add '[1] URL' to the bottom of the message. Note that the footnotes should be counted as of the summary; you do not need to keep the numbers from the earlier order, just from your summary. In other words, footnotes should start at 1, 2, 3, etc...", [llm_1, llm_2])

for key, article_dict in articles.items():

    r1 = article_dict["llm_1"]
    r2 = article_dict["llm_2"]
    es.evaluate(r1, r2)

print(es.prefs)

# TODO: need to update human evaluator task to track things over time
# TODO: same with GPT-3 evaluator
# TODO: this is all a great blog post??? See if anyone cares?