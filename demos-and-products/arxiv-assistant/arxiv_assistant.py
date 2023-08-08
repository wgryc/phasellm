import os
import re

from dotenv import load_dotenv

from feedparser import FeedParserDict

from phasellm.llms import ClaudeWrapper

from phasellm.agents import EmailSenderAgent, RSSAgent

load_dotenv()

# Load OpenAI and newsapi.org API keys.
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

# Load Gmail credentials.
gmail_email = os.getenv("GMAIL_EMAIL")
gmail_password = os.getenv("GMAIL_PASSWORD")  # https://myaccount.google.com/u/1/apppasswords

# Set up the LLM
llm = ClaudeWrapper(anthropic_api_key)


def interest_analysis(title: str, abstract: str, interests: str):
    interest_analysis_prompt = \
        f"""
        I want to determine if an academic paper is relevant to my interests. I am interested in: {interests}. The paper 
        is titled: {title}. It has the following abstract: {abstract}. Is this paper relevant to my interests? Respond 
        with either 'yes' or 'no'. Do not explain your reasoning.
        
        Example responses are given between the ### ### symbols. Respond exactly as shown in the examples.
        
        ###yes###
        or
        ###no###
        """
    return llm.text_completion(prompt=interest_analysis_prompt)


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
        Summarize why the the following paper is relevant to my interests. My interests are: {interests}. The paper is 
        titled: {title}. It has the following abstract: {abstract}.
        """
    return llm.text_completion(prompt=summary_prompt)


def send_email(title: str, abstract: str, link: str, summary: str) -> None:
    """
    This function sends an email to the user with the title of the paper and the summary.
    Args:
        title: The title of the paper.
        abstract: The abstract of the paper.
        link: The link to the paper.
        summary: The summary of the paper.

    Returns:

    """
    # Send email
    print('Sending email...')

    content = f"Title: {title}\n\nSummary:\n{summary}\n\nAbstract:\n{abstract}\n\nLink: {link}"

    email_agent = EmailSenderAgent(
        sender_name='arXiv Assistant',
        smtp='smtp.gmail.com',
        sender_address=gmail_email,
        password=gmail_password,
        port=587
    )
    email_agent.send_plain_email(recipient_email=gmail_email, subject=title, content=content)


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
    # Allow for a maximum of 1 retry.
    max_retries = 1

    title = paper['title']
    abstract = paper['summary']
    link = paper['link']
    interested = interest_analysis(title=title, abstract=abstract, interests=interests)

    # Find the answer within the response.
    answer = re.search(r'###(yes|no)###', interested)
    if not answer:
        if retries < max_retries:
            analyze_and_email(paper=paper, interests=interests, retries=retries + 1)
    else:
        interested = answer.group(0)

    # Send email if the user is interested.
    if interested == '###yes###':
        summary = summarize(title=title, abstract=abstract, interests=interests)
        send_email(title=title, abstract=abstract, link=link, summary=summary)
    elif interested == '###no###':
        pass
    else:
        print(f'LLM did not respond in the expected format after {max_retries}. Skipping paper:\n{title}')


def main():
    """
    Entry point for the arXiv assistant.
    Returns:

    """
    # Ask user what they want to read about.
    interests = input("What kinds of papers do you want to be notified about?")

    papers_processed = 0

    rss_agent = RSSAgent(url='https://arxiv.org/rss/cs')
    with rss_agent.poll(60) as poller:
        for papers in poller():
            print(f'Found {len(papers)} new paper(s).')
            for paper in papers:
                analyze_and_email(
                    paper=paper,
                    interests=interests
                )
                papers_processed += 1
                print(f'Processed {papers_processed} paper(s).')


if __name__ == '__main__':
    main()
