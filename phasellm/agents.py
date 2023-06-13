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

from docker import DockerClient
from docker.models.containers import Container, ExecResult

from typing import Generator, Union, Dict, List, Optional, NamedTuple

from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from .exceptions import LLMCodeException


class Agent(ABC):

    @abstractmethod
    def __init__(self, name: str = ''):
        """
        Abstract class for an agent.
        Args:
            name: The name of the agent.
        """
        self.name = name

    def __repr__(self):
        return f"Agent(name='{self.name}')"


@contextmanager
def stdout_io(stdout=None):
    """
    Used to hijack printing to screen so we can save the Python code output for the LLM (or any other arbitrary code).
    Args:
        stdout: The stdout to use.

    Returns:

    """
    old = sys.stdout
    if stdout is None:
        stdout = StringIO()
    sys.stdout = stdout
    yield stdout
    sys.stdout = old


class CodeExecutionAgent(Agent):

    def __init__(self, name: str = ''):
        """
        Creates a new CodeExecutionAgent.

        This agent is NOT sandboxed and should only be used for trusted code.

        Args:
            name: The name of the agent.
        """
        super().__init__(name=name)

    def __repr__(self):
        return f"CodeExecutionAgent(name={self.name})"

    @staticmethod
    def execute_code(code: str, globals=None, locals=None) -> str:
        """
        Executes arbitrary Python code and saves the output (or error!) to a variable.

        Returns the variable and a boolean (is_error) depending on whether an error took place.

        Args:
            code: Python code to execute.
            globals: python globals
            locals: python locals

        Returns:
            The logs from the code execution.

        """
        # TODO consider changing globals and locals parameter names to prevent shadowing the built-in functions.
        with stdout_io() as s:
            try:
                exec(code, globals, locals)
            except Exception as err:
                raise LLMCodeException(code, str(err))

        return s.getvalue()


class ExecCommands(NamedTuple):
    requirements: str
    python: str


class SandboxedCodeExecutionAgent(Agent):
    CODE_FILENAME = 'sandbox_code.py'

    def __init__(
            self,
            name: str = '',
            docker_image: str = 'python:3',
            scratch_dir: Union[Path, str] = None,
            module_package_mappings_whitelist: Dict[str, str] = None
    ):
        """
        Creates a new SandboxedCodeExecutionAgent.

        This agent is for executing arbitrary code in a sandboxed environment. We choose to use docker for this, so if
        you're running this code, you'll need to have docker installed and running.

        Args:
            name: Name of the agent.
            docker_image: Docker image to use for the sandboxed environment.
            scratch_dir: Scratch directory to use for copying files (bind mounting) to the sandboxed environment.
            module_package_mappings_whitelist: Dictionary of module to package mappings. This is used to determine
            which packages are allowed to be installed in the sandboxed environment.
        """
        super().__init__(name=name)

        if module_package_mappings_whitelist is None:
            module_package_mappings_whitelist = {
                "numpy": "numpy",
                "pandas": "pandas",
                "scipy": "scipy",
                "sklearn": "scikit-learn",
                "matplotlib": "matplotlib",
                "seaborn": "seaborn",
                "statsmodels": "statsmodels",
                "tensorflow": "tensorflow",
                "torch": "torch",
            }
        self.module_package_mappings_whitelist = module_package_mappings_whitelist

        self.docker_image = docker_image

        if scratch_dir is None:
            scratch_dir = f'.tmp/sandboxed_code_execution'
        self.scratch_dir = scratch_dir

        # Pre-compile regexes for performance (helps if executing code in a loop).
        self._module_regex = re.compile(r"(?:(?<=^import\s)|(?<=^from\s))\w+", flags=re.MULTILINE)

        # Create the docker client.
        self._client: DockerClient = docker.from_env()
        self._ping_client()

        # Placeholder for the container.
        self._container: Optional[Container] = None

    def __repr__(self):
        return f"SandboxedCodeExecutionAgent(" \
               f"name={self.name}, " \
               f"docker_image={self.docker_image}, " \
               f"scratch_dir={self.scratch_dir})"

    def __enter__(self):
        """
        Runs When entering the context manager.
        Returns:

        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Runs when exiting the context manager.
        Args:
            exc_type:
            exc_val:
            exc_tb:

        Returns:

        """
        self.close()

    def _ping_client(self):
        """
        Pings the docker client to make sure it's running.
        Returns:

        """
        if not self._client.ping():
            raise ConnectionError('Docker is not running. Please start docker.')

    def _create_scratch_dir(self) -> None:
        """
        Creates the scratch directory if it doesn't exist.
        Returns:

        """
        if not os.path.exists(self.scratch_dir):
            os.makedirs(self.scratch_dir)

    def _write_code_file(self, code: str) -> None:
        """
        Writes the code to a file in the scratch directory.
        Args:
            code: The code string to write to the file.

        Returns:

        """
        with open(os.path.join(self.scratch_dir, self.CODE_FILENAME), 'w') as f:
            f.write(code)

    def _write_requirements_file(self, packages: List[str]) -> None:
        """
        Writes a requirements.txt file to the scratch directory.
        Args:
            packages: List of packages to write to the requirements.txt file.

        Returns:

        """
        with open(os.path.join(self.scratch_dir, 'requirements.txt'), 'w') as f:
            for package in packages:
                f.write(f'{package}\n')

    def _modules_to_packages(self, code: str) -> List[str]:
        """
        Scans the code for modules and maps them to a package. If no package is specified in the mapping whitelist,
        then the package is ignored.
        Args:
            code:

        Returns:

        """

        modules = self._module_regex.findall(code)

        final_packages = []
        for module in modules:
            try:
                if self.module_package_mappings_whitelist[module] is not None:
                    final_packages.append(self.module_package_mappings_whitelist[module])
            except KeyError:
                pass
        return final_packages

    def _prep_commands(self, packages: List[str]) -> ExecCommands:
        """
        Prepares the commands to be run in the docker container.
        Args:
            packages: List of packages to install in the docker container.

        Returns:
            A tuple containing the requirements command and the python command in the form
            (requirements_command, python_command).

        """
        requirements_command = None
        if len(packages) > 0:
            requirements_command = 'pip install -r code/requirements.txt'
        # Note that -u is used to force unbuffered output.
        python_command = f'python -u code/{self.CODE_FILENAME}'

        return ExecCommands(requirements=requirements_command, python=python_command)

    @staticmethod
    def _handle_exec_errors(output: str, exit_code: int, code: str):
        """
        Handles errors that occur during code execution.
        Returns:

        """
        if exit_code is not None and exit_code != 0:
            raise LLMCodeException(code, output)

    def _execute(self, code: str, auto_stop_container: bool) -> Generator:
        """
        Starts the container, installs packages defined in the code (if they are provided in the
        module_package_mappings_whitelist), and executes the provided code inside the container.
        Args:
            code: The code string to execute.
            auto_stop_container: Whether or not to automatically stop the container after execution.
        Returns:
            A Generator that yields the stdout and stderr of the code execution.
        """

        self._create_scratch_dir()

        packages: List[str] = self._modules_to_packages(code)

        self._write_requirements_file(packages)
        self._write_code_file(code)

        commands: ExecCommands = self._prep_commands(packages)

        try:
            # If the container is already running, use it. Otherwise, start a new container.
            if self._container is None:
                self.start_container()

            # Run the requirements command if it exists.
            if commands.requirements is not None:
                res: ExecResult = self._container.exec_run(commands.requirements)
                self._handle_exec_errors(output=res.output, exit_code=res.exit_code, code=code)

            # Run the python command.
            exec_handle = self._client.api.exec_create(container=self._container.name, cmd=commands.python)
            res: ExecResult = self._client.api.exec_start(exec_handle['Id'], stream=True)

            # Yield the output of the python command.
            output = []
            for data in res:
                chunk = data.decode('utf-8')
                output.append(chunk)
                yield chunk
            output = ''.join(output)

            # Handle errors for streaming output.
            exit_code = self._client.api.exec_inspect(exec_handle['Id'])['ExitCode']
            self._handle_exec_errors(output=output, exit_code=exit_code, code=code)
        finally:
            if auto_stop_container:
                self.stop_container()

    def close(self):
        """
        Stops all containers and closes client sessions. This should be called when you're done using the agent.

        This method automatically runs when exiting the context manager. If you do not use a context manager, you
        should call this method manually.

        Returns:

        """
        self.stop_container()
        # Closes client sessions
        self._client.close()

    def start_container(self) -> None:
        """
        Starts the docker container.
        Returns:

        """
        if self._container is not None:
            raise RuntimeError('Container is already running.')

        container: Container = self._client.containers.create(
            image=self.docker_image,
            volumes={Path(self.scratch_dir).absolute(): {'bind': '/code', 'mode': 'rw'}},
            auto_remove=False,
            tty=True,
        )
        container.start()
        self._container = container

    def stop_container(self) -> None:
        """
        Stops the docker container and removes it, if it exists.
        Returns:

        """
        if self._container is not None:
            self._container.stop()
            self._container.remove()
            self._container = None

    def execute_code(self, code: str, stream: bool = True, auto_stop_container: bool = False) -> Union[str, Generator]:
        """
        Executes the provided code inside a sandboxed container.
        Args:
            code: The code string to execute.
            stream: Whether or not to stream the output of the code execution.
            auto_stop_container: Whether or not to automatically stop the container after the code execution.
        Returns:
            A string output of the whole code execution stdout and stderr if stream is False, otherwise a Generator
            that yields the stdout and stderr of the code execution.
        """
        generator = self._execute(code=code, auto_stop_container=auto_stop_container)
        if stream:
            return generator
        return ''.join(list(generator))


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
