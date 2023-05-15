import os
import random
import sys

sys.path.append(".")

import click
import json
import pandas as pd

from dataclasses import dataclass
from dotenv import load_dotenv
from typing import Any, Dict, List

from phasellm.eval import MetaLLMEvaluator
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

text_summarisation_llms: List[LanguageModelWrapper] = [
    OpenAIGPTWrapper(name="llm_1_openai", api_key=OPENAI_API_KEY),
    CohereWrapper(name="llm_2_cohere", api_key=COHERE_API_KEY),
    ClaudeWrapper(name="llm_3_claude", api_key=ANTHROPIC_API_KEY)
]

text_summary_ranker_llm = OpenAIGPTWrapper(name="meta_llm_openai", api_key=OPENAI_API_KEY)

summarize_texts_prompt_raw_1: List[Dict] = [
    {
        "role": "system",
        "content": "You are a helpful news summarizer. We will provide you with a list of news articles and will ask that you summarize them and retain links to source by adding footnotes. For example, if you have a news article describing XYZ and URL to the article, you would discuss XYZ[1] and add '[1] URL' to the bottom of the message. Note that the footnotes should be counted as of the summary; you do not need to keep the numbers from the earlier order, just from your summary. In other words, footnotes should start at 1, 2, 3, etc..."
    },
    {
        "role": "user",
        "content": "The articles below are about '{query}'. Please summarize them into a short paragraph with link retained as per the earlier instructions.\n\n{news_articles}"
    },
]

summarize_texts_prompt_raw_2: List[Dict] = [
    {
        "role": "system",
        "content": "You are a helpful news summarizer. We will provide you with a list of news articles and will ask that you summarize them and retain links to source by adding footnotes. For example, if you have a news article describing XYZ and URL to the article, you would discuss XYZ[1] and add '[1] URL' to the bottom of the message. The footnote numbers should start at [1] and increase consecutively. In other words, footnotes should start at 1, 2, 3, etc. For the actual paragraph, you can reorder reference articles and choose the ones to include as to make the paragraph as informative, pithy, and concise as possible. You can also have multiple footnotes per sentence if this helps tell the story. While you should avoid adding your own commentary in most cases, feel free to do so if it will help the reader understand the context of the paragraph you are writing."},
    {
        "role": "user",
        "content": "The articles below are about '{query}'. Please take on the role of an entertaining, successful, AI-driven investigative journalists and summarize them into a short paragraph. Make sure to follow the 'system' instructions.\n\n{news_articles}"
    },
]

rank_text_summaries_prompt_raw_1: List[Dict] = [
    {
        "role": "system",
        "content": """You are a helpful news ranker. We will provide you with a list of news summaries and will ask you to rank them on the basis of the specified objective.""",
    },
    {
        "role": "user",
        "content": """
Objective:
Which of the following news summaries is the most useful, is the most readable, has the higher quality, and is more engaging?
The news summaries to be ranked are provided as a list
Each summary starts with the name of its creator in the format [summarizer] >>>

News summaries:
'{text_summaries}'

Example news summaries to be ranked:
- [John] >>> Whatever.
- [Mary] >>> President Biden is scheduled to speak in Berlin.
- [Steve] >>> Floods in China

Example news summary ranking:
1. [Mary]
2. [Steve]
3. [John]

Ranking:

"""
    }
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

def rank_text_summaries(
    meta_llm_evaluator: MetaLLMEvaluator,
    llm_responses: Dict,
    rank_text_summaries_prompt: List[Dict]
) -> Dict:
    llm_ranking = {}

    try:
        print(f"Ranking text summaries with a meta LLM '{meta_llm_evaluator.name}'")

        llm_ranking = {
            "prompt": rank_text_summaries_prompt,
            "ranking": meta_llm_evaluator.rank(
                prompt=rank_text_summaries_prompt,
                llm_responses=list(llm_responses.items())
            )
        }
    except Exception as e:
        print(e)
    finally:
        print(f"Response from a meta LLM '{meta_llm_evaluator.name}' received.")
    
    return llm_ranking

def as_text_summary_list(text_summaries: Dict[str, Any]) -> str:
    summaries = [
        f"- [{llm}] >>> {text_summary}"
        for llm, text_summary in text_summaries.items()
    ]

    return "n".join(random.sample(summaries, len(summaries)))

@click.command()
@click.option("-i", "--evaluation_data_file", required=True, type=str, help="Input file path for raw, unprocessed evaluation data.")
@click.option("-o", "--evaluation_results_file", required=True, type=str, help="Output file path for LLM evaluation results.")
def evaluate_llms(evaluation_data_file: str, evaluation_results_file: str):
    evaluation_data = load_evalution_data(file_path=evaluation_data_file)
    print(f"Evaluation data with {len(evaluation_data['cases'])} evaluation cases loaded.")

    meta_llm_evaluator = MetaLLMEvaluator(name="meta_llm_evaluator", meta_llm=text_summary_ranker_llm)

    for evaluation_case in evaluation_data["cases"]:
        topic = evaluation_case["topic"]
        news_articles = evaluation_case["inputs"]["raw_text"][0:4097]
        evaluation_results = {}

        print(f"Generating a news summary for the '{topic}' topic...")

        summarize_texts_prompt = ChatPrompt(messages=summarize_texts_prompt_raw_1).fill_vars(
            query=topic, news_articles=news_articles
        )
        
        llm_responses: Dict[str, Any] = {}

        for llm in text_summarisation_llms:
            llm_response = None

            try:
                print(f"Generating text summary with LLM '{llm.name}'")

                llm_response = llm.complete_chat(messages=summarize_texts_prompt)
            except Exception as e:
                print(e)
            finally:
                print(f"Response from LLM '{llm.name}' received.")
        
            llm_responses[f"{llm.name}"] = llm_response
            evaluation_results[f"{llm.name}"] = llm_response
        
        rank_text_summaries_prompt = ChatPrompt(
            messages=rank_text_summaries_prompt_raw_1
        ).fill_vars(text_summaries=as_text_summary_list(text_summaries=llm_responses))
        
        text_summary_ranking = rank_text_summaries(
            meta_llm_evaluator=meta_llm_evaluator,
            llm_responses=llm_responses,
            rank_text_summaries_prompt=rank_text_summaries_prompt
        )

        evaluation_results["ranking"] = text_summary_ranking
        evaluation_case["inputs"]["prompt"] = summarize_texts_prompt
        evaluation_case["results"] = evaluation_results
    
        print("*"*100)
  
    save_evaluation_results(evaluation_results=evaluation_data, file_path=evaluation_results_file)

    print(f"Done. Goodbye.")

if __name__ == '__main__':
    evaluate_llms()
