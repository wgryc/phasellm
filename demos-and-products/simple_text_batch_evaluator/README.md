# Simple Text Batch Evaluatior

## Summary

An example of how to 1) sumamrize a batch of texts (news articles) with multiple LLMs, and to 2) use an LLM to rank the text summaries.

## Usage Examples

```bash
$ python demos-and-products/simple_text_batch_evaluator/llm_evaluation_text_batch.py \
-i "demos-and-products/simple_text_batch_evaluator/news_articles_sample.input.json" \
-o "demos-and-products/simple_text_batch_evaluator/news_articles_sample.output.json"

```

The above 

1. reads a batch of topic-tagged news articles from `demos-and-products/simple_text_batch_evaluator/news_articles_sample.input.json`
2. submits each batch of news articles to multiple LLMs using a prompt for a text summary
3. submits the text summaries received to another LLM which asks it to rank the text summaries.
4. The text summary rankings are saved to `"demos-and-products/simple_text_batch_evaluator/news_articles_sample.output.json"`.

