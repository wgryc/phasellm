#
#
# -- A SIMPLE EXAMPLE OF SUBMITTING A TEXT BATCH TO MULTIPLE LLMS --
#
# -- USAGE EXAMPLE -- 
#
# $ python scratchpad.py -i "tests/test_data/news_articles_test_set_1.input.json" -o "tests/test_data/news_articles_test_set_1.output.json"
#
#

import os
import sys

from pathlib import Path

# p = Path(__file__).parents[1]
# print(p)
# sys.path.insert(0, p)
# sys.path.append(p)
# print(sys.path)

import click
import json
import pandas as pd

from dataclasses import dataclass
from dotenv import load_dotenv
from typing import Dict, List

from phasellm.llms import ClaudeWrapper, CohereWrapper, LanguageModelWrapper, OpenAIGPTWrapper
from phasellm.prompts import ChatPrompt

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
NEWS_API_API_KEY = os.getenv("NEWS_API_API_KEY")

import anthropic
import cohere
import openai

openai.api_key = OPENAI_API_KEY

llm_openai_gpt = OpenAIGPTWrapper(name="llm_1_openai", api_key=OPENAI_API_KEY)
llm_cohere = CohereWrapper(name="llm_2_cohere", api_key=COHERE_API_KEY)
llm_claude = ClaudeWrapper(name="llm_3_claude", api_key=ANTHROPIC_API_KEY)

llms: List[LanguageModelWrapper] = [
    llm_openai_gpt,
    llm_cohere,
    llm_claude
]

chat_prompt_raw_1: List[Dict] = [
    {
        "role": "system",
        "content": "You are a helpful news summarizer. We will provide you with a list of news articles and will ask that you summarize them and retain links to source by adding footnotes. For example, if you have a news article describing XYZ and URL to the article, you would discuss XYZ[1] and add '[1] URL' to the bottom of the message. Note that the footnotes should be counted as of the summary; you do not need to keep the numbers from the earlier order, just from your summary. In other words, footnotes should start at 1, 2, 3, etc..."
    },
    {
        "role": "user",
        "content": "The articles below are about '{query}'. Please summarize them into a short paragraph with link retained as per the earlier instructions.\n\n{news_articles}"
    },
]

chat_prompt_raw_2: List[Dict] = [
    {
        "role":"system",
        "content": "You are a helpful news summarizer. We will provide you with a list of news articles and will ask that you summarize them and retain links to source by adding footnotes. For example, if you have a news article describing XYZ and URL to the article, you would discuss XYZ[1] and add '[1] URL' to the bottom of the message. The footnote numbers should start at [1] and increase consecutively. In other words, footnotes should start at 1, 2, 3, etc. For the actual paragraph, you can reorder reference articles and choose the ones to include as to make the paragraph as informative, pithy, and concise as possible. You can also have multiple footnotes per sentence if this helps tell the story. While you should avoid adding your own commentary in most cases, feel free to do so if it will help the reader understand the context of the paragraph you are writing."},
    {
        "role": "user",
        "content": "The articles below are about '{query}'. Please take on the role of an entertaining, successful, AI-driven investigative journalists and summarize them into a short paragraph. Make sure to follow the 'system' instructions.\n\n{news_articles}"
    },
]

def load_evalution_data(file_path: str) -> List[Dict]:
    print(f"Loading evaluation data from {file_path}")

    with open(file_path) as f:
        evaluation_data = json.load(f)

    return evaluation_data

def save_evaluation_results(evaluation_results: List[Dict], file_path: str) -> None:
    print(f"Saving evaluation results to {file_path}")

    with open(file_path, "w") as f:
        f.write(json.dumps(evaluation_results, indent=4))

@click.command()
@click.option("-i", "--evaluation_data_file", required=True, type=str, help="Input file path for raw, unprocessed evaluation data.")
@click.option("-o", "--evaluation_results_file", required=True, type=str, help="Output file path for LLM evaluation results.")
def evaluate_llms(evaluation_data_file: str, evaluation_results_file: str):
    chat_prompt_1 = ChatPrompt(messages=chat_prompt_raw_1)
    
    evaluation_data = load_evalution_data(file_path=evaluation_data_file)
    print(f"Evaluation data with {len(evaluation_data['cases'])} evaluation cases loaded.")

    for evaluation_case in evaluation_data["cases"]:
        topic = evaluation_case["topic"]
        news_articles = evaluation_case["inputs"]["raw_text"][0:4097]
        evaluation_results = {}

        print(f"Generating a news summary for the '{topic}' topic...")

        messages = chat_prompt_1.fill_vars(query=topic, news_articles=news_articles)
        
        for llm in llms:
            llm_response = None

            try:
                print(f"Calling LLM '{llm.name}'")

                llm_response = llm.complete_chat(messages=messages)
            except Exception as e:
                print(e)
            finally:
                print(f"Response from LLM '{llm.name}' received.")
        
            evaluation_results[f"{llm.name}"] = llm_response
        
        evaluation_case["inputs"]["prompt"] = messages
        evaluation_case["results"] = evaluation_results
    
        print("*"*100)
  
    save_evaluation_results(evaluation_results=evaluation_data, file_path=evaluation_results_file)

if __name__ == '__main__':
    evaluate_llms()
