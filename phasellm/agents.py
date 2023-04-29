"""
Agents to help with workflows.
"""

import sys
from io import StringIO
import contextlib

import requests
from datetime import datetime, timedelta

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from .exceptions import LLMCodeException

class Agent():
    """
    Abstract class for agents.
    """

    def __init__(self, name=''):
        self.name = name
        pass 

    def __repr__(self):
        return f"Agent(name='{self.name}')"
    
@contextlib.contextmanager
def stdoutIO(stdout=None):
    """
    Used to hijack printing to screen so we can save the Python code output for the LLM (or any other arbitrary code).
    """
    old = sys.stdout
    if stdout is None:
        stdout = StringIO()
    sys.stdout = stdout
    yield stdout
    sys.stdout = old

class CodeExecutionAgent(Agent):
    """
    Agent used for executing arbitrary code.
    """

    def __init__(self, name=''):
        self.name = name 

    def execute_code(self, code, globals=None, locals=None):
        """
        Executes arbitrary Python code and saves the output (or error!) to a variable.
        
        Returns the variable and a boolean (is_error) depending on whether an error took place.
        """
        is_error = False
        with stdoutIO() as s:
            try:
                exec(code, globals, locals)
            except Exception as err:
                raise LLMCodeException(code, str(err))

        return s.getvalue()

class EmailSenderAgent(Agent):
    """
    Send emails via an SMTP server.
    """

    def __init__(self, sender_name, smtp, sender_address, password, port):
        self.sender_name = sender_name
        self.smtp = smtp
        self.sender_address = sender_address 
        self.password = password
        self.port = port 

    def __repr__(self):
        return f"EmailSenderAgent(name={self.name})"
    
    def sendPlainEmail(self, recipient_email, subject, content):
        s = smtplib.SMTP(host=self.smtp, port=self.port)
        s.ehlo()
        s.starttls()
        s.login(self.sender_address, self.password)


        message = MIMEMultipart()
        
        message['From'] = f"{self.sender_name} <{self.sender_address}>"
        message['To'] = recipient_email
        message['Subject'] = subject
        message.attach(MIMEText(content, 'plain'))

        s.send_message(message)


class NewsSummaryAgent(Agent):
    """
    newsapi.org agent. Takes a query, calls the API, and summarizes news articles.
    """

    def __init__(self, apikey=None, name=''):
        self.apikey = apikey 
        self.name = name

    def __repr__(self):
        return f"NewsSummaryAgent(name={self.name})"
    
    def getQuery(self, query, days_back=1, include_descriptions=True, max_articles=25):
        """
        Gets all articles for a query for the # of days back.

        days_back: how far back we go with the query
        
        include_descriptions: will include article descriptions as well as titles; otherwise only titles 

        Returns a String with all the information so that an LLM can summarize it.
        """

        start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')

        api_url = f"https://newsapi.org/v2/everything?q={query}&from={start_date}&sortBy=publishedAt&apiKey={self.apikey}"

        headers = {'Accept': 'application/json'}
        r = requests.get(api_url, headers=headers)
        json_data = r.json()

        articles = json_data['articles']

        return_me = f"'---------------\nNEWS ARTICLES ABOUT {query} SINCE {start_date}\n---------------\n'"

        article_counter = 0

        if len(articles) == 0:
            return_me += "\nNo articles found.\n"
        else:
            for article in articles:
                article_counter += 1
                article_desc = f"\nTITLE: {article['title']}"
                if include_descriptions:
                    article_desc += f"\nDESCRIPTION: {article['description']}\n"
                article_desc += f"URL: {article['url']}"
                return_me += article_desc
                if article_counter > max_articles: break

        return_me += "---------------"

        return return_me