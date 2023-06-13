"""
Agents to help with workflows.
"""

import re
import os
import sys
import docker
import smtplib
import requests

from io import StringIO

from pathlib import Path

from abc import ABC, abstractmethod

from contextlib import contextmanager

from datetime import datetime, timedelta

from docker.models.containers import Container

from typing import Generator, Union, Dict, List, Optional

from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from .exceptions import LLMCodeException


class Agent(ABC):
    """
    Abstract class for agents.
    """

    @abstractmethod
    def __init__(self, name: str = ''):
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

    def __init__(self, name: str = ''):
        super().__init__(name=name)

    def __repr__(self):
        return f"CodeExecutionAgent(name={self.name})"

    @staticmethod
    def execute_code(code: str, globals=None, locals=None) -> str:
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

    def __init__(self, name: str = '', docker_image: str = 'python:3', scratch_dir: Union[Path, str] = None):
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
        Waits for the container to exit, removes it, and closes the docker client.
        """
        self.client.close()

    def _create_scratch_dir(self) -> None:
        """
        Creates a directory if it does not exist.
        """
        if not os.path.exists(self.scratch_dir):
            os.makedirs(self.scratch_dir)

    def _write_code_file(self, code: str) -> None:
        """
        Writes the code to a temporary file so that it can be volume mounted to a docker container.
        """
        with open(os.path.join(self.scratch_dir, self.CODE_FILENAME), 'w') as f:
            f.write(code)

    def _write_requirements_file(self, packages: List[str]) -> None:
        """
        Writes a requirements.txt file to the scratch directory.
        """
        with open(os.path.join(self.scratch_dir, 'requirements.txt'), 'w') as f:
            for package in packages:
                f.write(f'{package}\n')

    @staticmethod
    def _modules_to_packages(code: str) -> List[str]:
        """
        Scans the code for modules and maps them to a package. If no package is specified in the mapping whitelist,
        then the package is ignored.
        """
        module_package_mappings_whitelist = {
            "numpy": "numpy",
            "pandas": "pandas",
            "scipy": "scipy",
            "sklearn": "scikit-learn",
            "matplotlib": "matplotlib",
            "seaborn": "seaborn",
            "statsmodels": "statsmodels"
        }
        pattern = re.compile(r"(?:(?<=^import\s)|(?<=^from\s))\w+", flags=re.MULTILINE)
        modules = pattern.findall(code)

        final_packages = []
        for module in modules:
            try:
                if module_package_mappings_whitelist[module] is not None:
                    final_packages.append(module_package_mappings_whitelist[module])
            except KeyError:
                pass
        return final_packages

    def execute_code(self, code: str) -> Generator:
        """
        Runs a docker container with the specified image and command.
        """
        self._create_scratch_dir()
        packages = self._modules_to_packages(code)
        self._write_requirements_file(packages)
        self._write_code_file(code)

        # Prepare the command.
        requirements_command = None
        if len(packages) > 0:
            requirements_command = 'pip install -r code/requirements.txt'
        python_command = f'python code/{self.CODE_FILENAME}'

        container: Container = self.client.containers.run(
            image=self.docker_image,
            volumes={Path(self.scratch_dir).absolute(): {'bind': '/code', 'mode': 'rw'}},
            detach=True,
            tty=True
        )

        if requirements_command is not None:
            res = container.exec_run(requirements_command)
            if res.exit_code != 0:
                raise LLMCodeException(code, res.output.decode('utf-8'))

        res = container.exec_run(python_command, stream=True)

        try:
            while True:
                line = next(res.output)
                yield line
        except StopIteration:
            pass

        container.stop()
        container.remove()


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

    def sendPlainEmail(self, recipient_email: str, subject: str, content: str) -> None:
        # TODO deprecating this to be more Pythonic with naming conventions.
        print('sendPlainEmail() is deprecated. Use send_plain_email instead.')
        self.send_plain_email(recipient_email, subject, content)

    def send_plain_email(self, recipient_email: str, subject: str, content: str) -> None:
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

    def getQuery(
            self,
            query: str,
            days_back: int = 1,
            include_descriptions: bool = True,
            max_articles: int = 25
    ) -> str:
        # TODO deprecating this to be more Pythonic with naming conventions.
        print('getQuery() is deprecated. Use get_query instead.')
        return self.get_query(query, days_back, include_descriptions, max_articles)

    def get_query(
            self,
            query: str,
            days_back: int = 1,
            include_descriptions: bool = True,
            max_articles: int = 25
    ) -> str:
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

        api_url = \
            f"https://newsapi.org/v2/everything?" \
            f"q={query}" \
            f"&from={start_date}" \
            f"&sortBy=publishedAt" \
            f"&apiKey={self.apikey}"

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
                if article_counter > max_articles:
                    break

        return_me += "---------------"

        return return_me
