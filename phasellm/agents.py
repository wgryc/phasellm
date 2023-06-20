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

from warnings import warn

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
            module_package_mappings: Dict[str, str] = None
    ):
        """
        Creates a new SandboxedCodeExecutionAgent.

        This agent is for executing arbitrary code in a sandboxed environment. We choose to use docker for this, so if
        you're running this code, you'll need to have docker installed and running.

        Examples:
                >>> from typing import Generator
                >>> from phasellm.agents import SandboxedCodeExecutionAgent
            Managing the docker client yourself:
                >>> agent = SandboxedCodeExecutionAgent()
                >>> logs = agent.execute_code('print("Hello World!")')
                >>> for log in logs:
                ...     print(log)
                Hello World!
                >>> agent.close()
            Using the context manager:
                >>> with SandboxedCodeExecutionAgent() as agent:
                ...     logs: Generator = agent.execute_code('print("Hello World!")')
                ...     for log in logs:
                ...         print(log)
                Hello World!
            Code with custom packages is possible! Note that the package must exist in the module_package_mappings
            dictionary:
                >>> module_package_mappings = {
                ...     "numpy": "numpy"
                ...}
                >>> with SandboxedCodeExecutionAgent(module_package_mappings=module_package_mappings) as agent:
                ...     logs = agent.execute_code('import numpy as np; print(np.__version__)')
                ...     for log in logs:
                ...         print(log)
                1.24.3
            Disable log streaming (waits for code to finish executing before returning logs):
                >>> with SandboxedCodeExecutionAgent() as agent:
                ...     logs = agent.execute_code('print("Hello World!")', stream=False)
                ...     print(logs)
                Hello World!
            Custom docker image:
                >>> with SandboxedCodeExecutionAgent(docker_image='python:3.7') as agent:
                Hello World!
            Custom scratch directory:
                >>> with SandboxedCodeExecutionAgent(scratch_dir='my_dir') as agent:
            Stop container after each call to agent.execute_code()
                >>> with SandboxedCodeExecutionAgent() as agent:
                ...     logs = agent.execute_code('print("Hello 1")', auto_stop_container=True)
                ...     assert agent._container is None
                ...     logs = agent.execute_code('print("Hello 2")', auto_stop_container=True)
                ...     assert agent._container is None
        Args:
            name: Name of the agent.
            docker_image: Docker image to use for the sandboxed environment.
            scratch_dir: Scratch directory to use for copying files (bind mounting) to the sandboxed environment.
            module_package_mappings: Dictionary of module to package mappings. This is used to determine
            which packages are allowed to be installed in the sandboxed environment.
        """
        super().__init__(name=name)

        if module_package_mappings is None:
            module_package_mappings = {
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
        self.module_package_mappings = module_package_mappings

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
            SandboxedCodeExecutionAgent()
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Runs when exiting the context manager.
        Args:
            exc_type: The exception type.
            exc_val: The exception value.
            exc_tb: The exception traceback.

        Returns:

        """
        self.close()

    def _ping_client(self) -> None:
        """
        Pings the docker client to make sure it's running.

        Returns:

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.
        """
        self._client.ping()

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
            code: The code to scan for modules.

        Returns:
            A list of packages to install in the sandboxed environment.
        """

        modules = self._module_regex.findall(code)

        final_packages = []
        for module in modules:
            try:
                if self.module_package_mappings[module] is not None:
                    final_packages.append(self.module_package_mappings[module])
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
    def _handle_exec_errors(output: str, exit_code: int, code: str) -> None:
        """
        Handles errors that occur during code execution.
        Args:
            output: The output of the code execution.
            exit_code: The exit code of the code execution.
            code: The code that was executed.

        Returns:

        """
        if exit_code is not None and exit_code != 0:
            raise LLMCodeException(code, output)

    def _execute(self, code: str, auto_stop_container: bool) -> Generator:
        """
        Starts the container, installs packages defined in the code (if they are provided in the
        module_package_mappings), and executes the provided code inside the container.
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

    def close(self) -> None:
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

    def __init__(self, sender_name: str, smtp: str, sender_address: str, password: str, port: int, name: str = ''):
        """
        Create an EmailSenderAgent.

        Sends emails via an SMPT server.

        Args:
            sender_name: Name of the sender (i.e., "Wojciech")
            smtp: The smtp server (e.g., smtp.gmail.com)
            sender_address: The sender's email address
            password: The password for the email account
            port: The port used by the SMTP server
            name: The name of the agent (optional)
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
        """
        DEPRECATED: see send_plain_email
        Args:
            recipient_email: The person receiving the email
            subject: Email subject
            content: The plain text context for the email

        Returns:

        """
        # TODO deprecating this to be more Pythonic with naming conventions.
        warn('sendPlainEmail() is deprecated. Use send_plain_email() instead.')
        self.send_plain_email(recipient_email=recipient_email, subject=subject, content=content)

    def send_plain_email(self, recipient_email: str, subject: str, content: str) -> None:
        """
        Sends an email encoded as plain text.
        Args:
            recipient_email: The person receiving the email
            subject: Email subject
            content: The plain text context for the email

        Returns:

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

    def __init__(self, apikey: str = None, name: str = ''):
        """
        Create a NewsSummaryAgent.

        Takes a query, calls the API, and gets news articles.
        Args:
            apikey: The API key for newsapi.org
            name: The name of the agent (optional)
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
        """
        DEPRECATED: see get_query
        Args:
            query: What keyword to look for in news articles
            days_back: How far back we go with the query
            include_descriptions: Will include article descriptions as well as titles; otherwise only titles
            max_articles: How many articles to include in the summary

        Returns:
            A news summary string
        """
        # TODO deprecating this to be more Pythonic with naming conventions.
        warn('getQuery() is deprecated. Use get_query() instead.')
        return self.get_query(
            query=query,
            days_back=days_back,
            include_descriptions=include_descriptions,
            max_articles=max_articles
        )

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

        Args:
            query: What keyword to look for in news articles
            days_back: How far back we go with the query
            include_descriptions: Will include article descriptions as well as titles; otherwise only titles
            max_articles: How many articles to include in the summary

        Returns:
            A news summary string
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
