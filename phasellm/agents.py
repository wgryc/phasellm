"""
Agents to help with workflows.
"""

import os
import sys
import docker
import smtplib
import requests

from io import StringIO

from pathlib import Path

from typing import Generator

from abc import ABC, abstractmethod

from contextlib import contextmanager

from datetime import datetime, timedelta

from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from .exceptions import LLMCodeException


class Agent(ABC):
    """
    Abstract class for agents.
    """

    @abstractmethod
    def __init__(self, name=''):
        self.name = name

    def __repr__(self):
        return f"Agent(name='{self.name}')"


@contextmanager
def stdout_io(stdout=None):
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
        super().__init__(name=name)

    def __repr__(self):
        return f"CodeExecutionAgent(name={self.name})"

    @staticmethod
    def execute_code(code: str, globals=None, locals=None):
        """
        Executes arbitrary Python code and saves the output (or error!) to a variable.
        
        Returns the variable and a boolean (is_error) depending on whether an error took place.
        """
        # TODO consider changing globals and locals parameter names to prevent shadowing the built-in functions.
        with stdout_io() as s:
            try:
                exec(code, globals, locals)
            except Exception as err:
                raise LLMCodeException(code, str(err))

        return s.getvalue()


class SandboxedCodeExecutionAgent(Agent):
    """
    Agent used for executing arbitrary code in a sandboxed environment. We choose to use docker for this, so if you're
    running this code, you'll need to have docker installed and running.
    """
    CODE_FILENAME = 'sandbox_code.py'

    def __init__(self, name: str = '', docker_image: str = 'python:3', scratch_dir: str = None):
        super().__init__(name=name)

        self.docker_image = docker_image

        if scratch_dir is None:
            scratch_dir = f'.tmp/sandboxed_code_execution'
        self.scratch_dir = scratch_dir

    def __repr__(self):
        return f"SandboxedCodeExecutionAgent(name={self.name})"

    def __enter__(self):
        """
        Initializes the docker client.
        """
        self.client = docker.from_env()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Closes the docker client.
        """
        self.client.close()

    def _create_scratch_dir(self):
        """
        Creates a directory if it does not exist.
        """
        if not os.path.exists(self.scratch_dir):
            os.makedirs(self.scratch_dir)

    def _code_to_temp_file(self, code: str):
        """
        Writes the code to a temporary file so that it can be volume mounted to a docker container.
        """
        self._create_scratch_dir()
        with open(os.path.join(self.scratch_dir, self.CODE_FILENAME), 'w') as f:
            f.write(code)

    def execute_code(self, code: str) -> Generator:
        """
        Runs a docker container with the specified image and command.
        """
        self._code_to_temp_file(code)

        # TODO consider implementing a procedure for installing python packages.
        return self.client.containers.run(
            image=self.docker_image,
            command=f'python code/{self.CODE_FILENAME}',
            volumes={Path(self.scratch_dir).absolute(): {'bind': '/code', 'mode': 'rw'}},
            stream=True,
            auto_remove=True
        )


class EmailSenderAgent(Agent):
    """
    Send emails via an SMTP server.
    """

    def __init__(self, sender_name: str, smtp: str, sender_address: str, password: str, port: int, name: str = ''):
        """
        Initialize an EmailSenderAgent object.

        Keyword arguments:
        sender_name -- name of the sender (i.e., "Wojciech")
        smtp -- the smtp server (e.g., smtp.gmail.com)
        sender_address -- the sender's email address
        password -- the password for the email account
        port -- the port used by the SMTP server
        """
        super().__init__(name=name)
        self.sender_name = sender_name
        self.smtp = smtp
        self.sender_address = sender_address
        self.password = password
        self.port = port

    def __repr__(self):
        return f"EmailSenderAgent(name={self.name})"

    def sendPlainEmail(self, recipient_email: str, subject: str, content: str):
        # TODO deprecating this to be more Pythonic with naming conventions.
        print('sendPlainEmail() is deprecated. Use send_plain_email instead.')
        self.send_plain_email(recipient_email, subject, content)

    def send_plain_email(self, recipient_email: str, subject: str, content: str):
        """
        Sends an email encoded as plain text.

        Keywords arguments:
        recipient_email -- the person receiving the email
        subject -- email subject
        content -- the plain text context for the email
        """
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
    newsapi.org agent. Takes a query, calls the API, and gets news articles.
    """

    def __init__(self, apikey: str = None, name: str = ''):
        """
        Initializes the agent. Requires a newsapi.org API key.
        """
        super().__init__(name=name)
        self.apikey = apikey

    def __repr__(self):
        return f"NewsSummaryAgent(name={self.name})"

    def getQuery(self, query, days_back=1, include_descriptions=True, max_articles=25):
        # TODO deprecating this to be more Pythonic with naming conventions.
        print('getQuery() is deprecated. Use get_query instead.')
        self.get_query(query, days_back, include_descriptions, max_articles)

    def get_query(self, query: str, days_back: int = 1, include_descriptions: bool = True, max_articles: int = 25):
        """
        Gets all articles for a query for the # of days back. Returns a String with all the information so that an LLM
        can summarize it. Note that obtaining too many articles will likely cause an issue with prompt length.

        Keyword arguments:
        query -- what keyword to look for in news articles
        days_back -- how far back we go with the query
        include_descriptions -- will include article descriptions as well as titles; otherwise only titles 
        max_articles -- how many articles to include in the summary
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
