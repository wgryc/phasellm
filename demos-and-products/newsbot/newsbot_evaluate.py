from phasellm.eval import EvaluationStream

import json 

JSON_FILE = "news_articles.json"

def load_data_set(json_file):
    articles = None
    with open(json_file, 'r') as reader:
        articles = json.loads(reader.read())
    return articles

articles = load_data_set(JSON_FILE)

# Note that we don't pass the two LLMs to the Evaluation Stream -- no need to do so in this example.
es = EvaluationStream("Which news summary is higher quality and more engaging?", "You are a helpful news summarizer. We will provide you with a list of news articles and will ask that you summarize them and retain links to source by adding footnotes. For example, if you have a news article describing XYZ and URL to the article, you would discuss XYZ[1] and add '[1] URL' to the bottom of the message. Note that the footnotes should be counted as of the summary; you do not need to keep the numbers from the earlier order, just from your summary. In other words, footnotes should start at 1, 2, 3, etc...", [None, None])

for key, article_dict in articles.items():
    r1 = article_dict["llm_1"]
    r2 = article_dict["llm_2"]
    es.evaluate(r1, r2)

print(es.prefs)