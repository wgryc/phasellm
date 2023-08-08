import os

from dotenv import load_dotenv

from feedparser import FeedParserDict

from phasellm.llms import OpenAIGPTWrapper

from phasellm.agents import EmailSenderAgent, RSSAgent

load_dotenv()

# Load OpenAI and newsapi.org API keys.
openai_api_key = os.getenv("OPENAI_API_KEY")

# Load Gmail credentials.
gmail_email = os.getenv("GMAIL_EMAIL")
gmail_password = os.getenv("GMAIL_PASSWORD")

# Set up the LLM
llm = OpenAIGPTWrapper(openai_api_key, model="gpt-4")


def summarize(title: str, abstract: str, interests: str):
    """
    This function summarizes why the paper might be relevant to the user's interests.
    Args:
        title: The title of the paper.
        abstract: The abstract of the paper.
        interests: The user's interests.

    Returns: The summary of why the paper might be relevant to the user's interests.

    """
    # Summarize why the paper might be relevant to the user's interests.
    summary_prompt = \
        f"""
        You are an LLM tasked with summarizing why an academic paper is relevant to a user's interests.
        The user is interested in {interests}. The paper is titled {title} and has the following
        abstract: {abstract}. Please summarize why this paper is relevant to the user's interests.
        """
    return llm.text_completion(prompt=summary_prompt)


def send_email(title: str, summary: str) -> None:
    """
    This function sends an email to the user with the title of the paper and the summary.
    Args:
        title: The title of the paper.
        summary: The summary of the paper.

    Returns:

    """
    # Send email
    email_agent = EmailSenderAgent(
        sender_name='arXiv Assistant',
        smtp='smtp.gmail.com',
        sender_address=gmail_email,
        password=gmail_password,
        port=587
    )
    email_agent.send_plain_email(recipient_email=gmail_email, subject=title, content=summary)


def analyze_and_email(paper: FeedParserDict, interests: str, retries: int = 0) -> None:
    """
    This function analyzes the latest papers from arXiv and emails the user if any of them are relevant to their
    interests.
    Args:
        paper: The paper to analyze.
        interests: The user's interests.
        retries: The number of retry attempts made so far.
    Returns:

    """

    title = paper['title']
    abstract = paper['summary']
    interest_analysis_prompt = \
        f"""
        You are an LLM tasked with determining whether or not an academic paper is relevant to a user's 
        interests. The user is interested in {interests}. The paper is titled {title} and has the following
        abstract: {abstract}. Is this paper relevant to the user's interests? If so, respond with 'yes'. If not,
        respond with 'no'. Answer with only 'yes' or 'no', no punctuation.
        """
    interested = llm.text_completion(prompt=interest_analysis_prompt)
    if interested == 'yes':
        summary = summarize(title=title, abstract=abstract, interests=interests)
        send_email(title=title, summary=summary)
    elif interested == 'no':
        pass
    else:
        # Retry up to 3 times.
        if retries <= 2:
            analyze_and_email(paper=paper, interests=interests, retries=retries + 1)
        else:
            raise ValueError("LLM did not respond with 'yes' or 'no' after 3 attempts.")


def main():
    """
    Entry point for the arXiv assistant.
    Returns:

    """
    # Ask user what they want to read about.
    interests = input("What kinds of papers do you want to be notified about?")

    rss_agent = RSSAgent(url='https://arxiv.org/rss/cs')
    with rss_agent.poll(60) as poller:
        for papers in poller():
            for paper in papers:
                analyze_and_email(
                    paper=paper,
                    interests=interests
                )


if __name__ == '__main__':
    main()
