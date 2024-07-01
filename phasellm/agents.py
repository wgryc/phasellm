"""
Agents to help with workflows.
"""

import re
import os
import sys
import time
import docker
import smtplib
import requests
import subprocess
import feedparser

from queue import Queue

from io import StringIO

from pathlib import Path

from warnings import warn

from threading import Thread

from functools import partial

from bs4 import BeautifulSoup

from dataclasses import dataclass

from abc import ABC, abstractmethod

from fake_useragent import UserAgent

from contextlib import contextmanager

from datetime import datetime, timedelta

from playwright.sync_api import sync_playwright

from docker import DockerClient
from docker.models.containers import Container, ExecResult

from typing import Generator, Union, Dict, List, Optional, NamedTuple, Callable

from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from .exceptions import LLMCodeException


class Agent(ABC):

    @abstractmethod
    def __init__(self, name: str = ""):
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

    """
    old = sys.stdout
    if stdout is None:
        stdout = StringIO()
    sys.stdout = stdout
    yield stdout
    sys.stdout = old


class CodeExecutionAgent(Agent):

    def __init__(self, name: str = ""):
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
    CODE_FILENAME = "sandbox_code.py"

    def __init__(
        self,
        name: str = "",
        docker_image: str = "python:3",
        scratch_dir: Union[Path, str] = None,
        module_package_mappings: Dict[str, str] = None,
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

            Code with custom packages is possible! Note that the package must exist in the module_package_mappings dictionary:
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
            scratch_dir = f".tmp/sandboxed_code_execution"
        self.scratch_dir = scratch_dir

        # Pre-compile regexes for performance (helps if executing code in a loop).
        self._module_regex = re.compile(
            r"(?:(?<=^import\s)|(?<=^from\s))\w+", flags=re.MULTILINE
        )

        # Create the docker client.
        self._client: DockerClient = docker.from_env()
        self._ping_client()

        # Get the docker image.
        self._client.images.pull(self.docker_image)

        # Placeholder for the container.
        self._container: Optional[Container] = None

    def __repr__(self):
        return (
            f"SandboxedCodeExecutionAgent("
            f"name={self.name}, "
            f"docker_image={self.docker_image}, "
            f"scratch_dir={self.scratch_dir})"
        )

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

        """
        self.close()

    def _ping_client(self) -> None:
        """
        Pings the docker client to make sure it's running.

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.

        """
        self._client.ping()

    def _create_scratch_dir(self) -> None:
        """
        Creates the scratch directory if it doesn't exist.
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
        with open(os.path.join(self.scratch_dir, self.CODE_FILENAME), "w") as f:
            f.write(code)

    def _write_requirements_file(self, packages: List[str]) -> None:
        """
        Writes a requirements.txt file to the scratch directory.

        Args:
            packages: List of packages to write to the requirements.txt file.

        Returns:

        """
        with open(os.path.join(self.scratch_dir, "requirements.txt"), "w") as f:
            for package in packages:
                f.write(f"{package}\n")

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
            requirements_command = "pip install -r code/requirements.txt"
        # Note that -u is used to force unbuffered output.
        python_command = f"python -u code/{self.CODE_FILENAME}"

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
            auto_stop_container: Whether to automatically stop the container after execution.

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
                self._handle_exec_errors(
                    output=res.output, exit_code=res.exit_code, code=code
                )

            # Run the python command.
            exec_handle = self._client.api.exec_create(
                container=self._container.name, cmd=commands.python
            )
            res: ExecResult = self._client.api.exec_start(
                exec_handle["Id"], stream=True
            )

            # Yield the output of the python command.
            output = []
            for data in res:
                chunk = data.decode("utf-8")
                output.append(chunk)
                yield chunk
            output = "".join(output)

            # Handle errors for streaming output.
            exit_code = self._client.api.exec_inspect(exec_handle["Id"])["ExitCode"]
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
            raise RuntimeError("Container is already running.")

        container: Container = self._client.containers.create(
            image=self.docker_image,
            volumes={
                Path(self.scratch_dir).absolute(): {"bind": "/code", "mode": "rw"}
            },
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

    def execute_code(
        self, code: str, stream: bool = True, auto_stop_container: bool = False
    ) -> Union[str, Generator]:
        """
        Executes the provided code inside a sandboxed container.

        Args:
            code: The code string to execute.
            stream: Whether to stream the output of the code execution.
            auto_stop_container: Whether to automatically stop the container after the code execution.

        Returns:
            A string output of the whole code execution stdout and stderr if stream is False, otherwise a Generator
            that yields the stdout and stderr of the code execution.

        """
        generator = self._execute(code=code, auto_stop_container=auto_stop_container)
        if stream:
            return generator
        return "".join(list(generator))


class EmailSenderAgent(Agent):

    def __init__(
        self,
        sender_name: str,
        smtp: str,
        sender_address: str,
        password: str,
        port: int,
        name: str = "",
    ):
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

        """
        # TODO deprecating this to be more Pythonic with naming conventions.
        warn("sendPlainEmail() is deprecated. Use send_plain_email() instead.")
        self.send_plain_email(
            recipient_email=recipient_email, subject=subject, content=content
        )

    def send_plain_email(
        self, recipient_email: str, subject: str, content: str
    ) -> None:
        """
        Sends an email encoded as plain text.

        Args:
            recipient_email: The person receiving the email
            subject: Email subject
            content: The plain text context for the email

        """
        s = smtplib.SMTP(host=self.smtp, port=self.port)
        s.ehlo()
        s.starttls()
        s.login(self.sender_address, self.password)

        message = MIMEMultipart()
        message["From"] = f"{self.sender_name} <{self.sender_address}>"
        message["To"] = recipient_email
        message["Subject"] = subject
        message.attach(MIMEText(content, "plain"))

        s.send_message(message)


class NewsSummaryAgent(Agent):

    def __init__(self, apikey: str = None, name: str = ""):
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
        max_articles: int = 25,
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
        warn("getQuery() is deprecated. Use get_query() instead.")
        return self.get_query(
            query=query,
            days_back=days_back,
            include_descriptions=include_descriptions,
            max_articles=max_articles,
        )

    def get_query(
        self,
        query: str,
        days_back: int = 1,
        include_descriptions: bool = True,
        max_articles: int = 25,
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

        start_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

        api_url = (
            f"https://newsapi.org/v2/everything?"
            f"q={query}"
            f"&from={start_date}"
            f"&sortBy=publishedAt"
            f"&apiKey={self.apikey}"
        )

        headers = {"Accept": "application/json"}
        r = requests.get(api_url, headers=headers)
        json_data = r.json()

        articles = json_data["articles"]

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


class WebpageAgent(Agent):

    def __init__(self, name: str = ""):
        """
        Create a WebpageAgent.

        This agent helps you scrape webpages.

        Examples:
            >>> from phasellm.agents import WebpageAgent

            Use default parameters:
                >>> agent = WebpageAgent()
                >>> text = agent.scrape('https://10millionsteps.com/ai-inflection')

            Keep html tags:
                >>> agent = WebpageAgent()
                >>> text = agent.scrape('https://10millionsteps.com/ai-inflection', text_only=False, body_only=False)

            Keep html tags, but only return body content:
                >>> agent = WebpageAgent()
                >>> text = agent.scrape('https://10millionsteps.com/ai-inflection', text_only=False, body_only=True)

            Use a headless browser to enable scraping of dynamic content:
                >>> agent = WebpageAgent()
                >>> text = agent.scrape('https://10millionsteps.com/ai-inflection', text_only=False, body_only=True,
                ...                     use_browser=True)

            Pass custom headers:
                >>> agent = WebpageAgent()
                >>> headers = {'Example': 'header'}
                >>> text = agent.scrape('https://10millionsteps.com/ai-inflection', headers=headers)

            Wait for a selector to load (useful for dynamic content, only works when use_browser=True):
                >>> agent = WebpageAgent()
                >>> text = agent.scrape('https://10millionsteps.com/ai-inflection', use_browser=True,
                ...                     wait_for_selector='#dynamic')

        Args:
            name: The name of the agent (optional)

        """
        super().__init__(name=name)

        self.session = requests.Session()

    def __repr__(self):
        return f"WebpageAgent(name={self.name})"

    @staticmethod
    def _validate_url(url: str) -> None:
        """
        This method validates that a url can be used by the agent.
        """
        if not url.startswith("http"):
            raise ValueError(f"Url must use HTTP(S). Invalid URL: {url}")

        # TODO consider adding more validations.

    @staticmethod
    def _handle_errors(res: requests.Response) -> None:
        """
        This method handles errors that occur during a request.

        Args:
            res: The response from the request.

        """
        if res.status_code != 200:
            raise Exception(
                f"WebpageAgent received a non-200 status code: {res.status_code}\n"
                f"{res.reason}"
            )

    @staticmethod
    def _parse_html(html: str, text_only: bool = True, body_only: bool = False) -> str:
        """
        This method parses the given html string.

        Args:
            html: The html to parse.
            text_only: If True, only the text of the webpage is returned. If False, the entire HTML is returned.
            body_only: If True, only the body of the webpage is returned. If False, the entire HTML is returned.

        Returns:
            The string containing the webpage text or html.

        """
        if text_only or body_only:
            soup = BeautifulSoup(html, features="lxml")
            if text_only and body_only:
                text = soup.body.get_text()
            elif text_only:
                text = soup.get_text()
            else:
                text = str(soup.body)
        else:
            text = html
        return text.strip()

    @staticmethod
    def _prep_headers(headers: Dict = None) -> Dict:
        """
        This method prepares the headers for a request. It fills in missing headers with default values. It also
        adds a fake user agent to reduce the likelihood of being blocked.

        Args:
            headers: The headers to use for the request.

        Returns:
            The headers to use for the request.
        """
        if headers is None:
            headers = {}

        if "Accept" not in headers:
            headers["Accept"] = (
                "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
            )
        if "User-Agent" not in headers:
            headers["User-Agent"] = UserAgent().chrome
        if "Referrer" not in headers:
            headers["Referrer"] = "https://www.google.com/"
        if "Accept-Encoding" not in headers:
            headers["Accept-Encoding"] = "gzip, deflate, br"
        if "Accept-Language" not in headers:
            headers["Accept-Language"] = "*"
        if "Connection" not in headers:
            headers["Connection"] = "keep-alive"
        if "Upgrade-Insecure-Requests" not in headers:
            headers["Upgrade-Insecure-Requests"] = "1"
        if "Cache-Control" not in headers:
            headers["Cache-Control"] = "max-age=0"

        return headers

    def _scrape_html(self, url: str, headers: Dict = None) -> str:
        """
        This method scrapes a webpage and returns a string containing the html of the webpage.

        Args:
            url: The URL of the webpage to scrape.
            headers: A dictionary of headers to use for the request.

        Returns:
            A string containing the html of the webpage.

        """

        res = self.session.get(url=url, headers=headers, timeout=30)

        self._handle_errors(res=res)

        try:
            return res.content.decode(res.encoding)
        except Exception as e:
            raise Exception(
                f"WebpageAgent could not decode the response from the URL: {url}\n{e}"
            )

    @staticmethod
    def _scrape_html_and_js(
        url: str, headers: Dict, wait_for_selector: str = None
    ) -> str:
        """
        This method scrapes a webpage and returns a string containing the html of the webpage. It uses a headless
        browser to render the webpage and execute javascript.

        Args:
            url: The URL of the webpage to scrape.
            headers: A dictionary of headers to use for the request.
            wait_for_selector: The selector to wait for before returning the HTML. Useful for when you know something
            should be on the page, but it is not there yet since it needs to be rendered by javascript.

        Returns:
            A string containing the html of the webpage.

        """
        # Ensure chromium is installed for the headless browser.
        subprocess.call("playwright install chromium")

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(extra_http_headers=headers)
            page.goto(url)
            if wait_for_selector is None:
                # Wait until there are no network connections for at least `500` ms.
                page.wait_for_load_state("networkidle")
            else:
                # Wait until the `selector` defined by 'wait_for_selector' is added to the DOM.
                page.wait_for_selector(wait_for_selector)
            data = page.content()
            browser.close()
        return data

    def scrape(
        self,
        url: str,
        headers: Dict = None,
        use_browser: bool = False,
        wait_for_selector: str = None,
        text_only: bool = True,
        body_only: bool = True,
    ) -> str:
        """
        This method scrapes a webpage and returns a string containing the html or text of the webpage.

        Args:
            url: The URL of the webpage to scrape.
            headers: A dictionary of headers to use for the request.
            use_browser: If True, the webpage is rendered using a headless browser, allowing javascript to run and
                hydrate the page. If False, the webpage is scraped as-is.
            wait_for_selector: The selector to wait for before returning the HTML. Useful for when you know something
                should be on the page, but it is not there yet since it needs to be rendered by javascript. Only used when
                use_browser is True.
            text_only: If True, only the text of the webpage is returned. If False, the entire HTML is returned.
            body_only: If True, only the body of the webpage is returned. If False, the entire HTML is returned.

        Returns:
            A string containing the text of the webpage.

        """

        self._validate_url(url=url)

        headers = self._prep_headers(headers=headers)

        if use_browser:
            data = self._scrape_html_and_js(
                url=url, headers=headers, wait_for_selector=wait_for_selector
            )
        else:
            data = self._scrape_html(url=url, headers=headers)

        data = self._parse_html(html=data, text_only=text_only, body_only=body_only)

        return data


@dataclass
class WebSearchResult:
    """
    This dataclass represents a single search result.
    """

    title: str
    url: str
    description: str
    content: str


class WebSearchAgent(Agent):

    def __init__(
        self,
        name: str = "",
        api_key: str = None,
        rate_limit: float = 1,
        text_only: bool = True,
        body_only: bool = True,
        use_browser: bool = False,
        wait_for_selector: str = None,
    ):
        """
        Create a WebSearchAgent.

        This agent helps you search the web using a web search API. Currently, the agent supports Google and Brave.

        Examples:
            >>> from phasellm.agents import WebSearchAgent

            Search with Google:
                >>> agent = WebSearchAgent(
                ...     name='Google Search Agent',
                ...     api_key='YOUR_API_KEY'
                ... )
                >>> results = agent.search_google(
                ...     query='test'
                ...     custom_search_engine_id='YOUR_CUSTOM_SEARCH_ENGINE_ID'
                ... )

            Search with Brave:
                >>> agent = WebSearchAgent(
                ...     name='Brave Search Agent',
                ...     api_key='YOUR_API_KEY'
                ... )
                >>> results = agent.search_brave(query='test')

            Iterate over the results:
                >>> for result in results:
                ...     print(result.title)
                ...     print(result.url)
                ...     print(result.description)
                ...     print(result.content)

        Args:
            name: The name of the agent (optional).
            api_key: The API key to use for the search engine.
            rate_limit: The number of seconds to wait between requests for webpage content.
            text_only: If True, only the text of the webpage is returned. If False, the entire HTML is returned.
            body_only: If True, only the body of the webpage is returned. If False, the entire HTML is returned.
            use_browser: If True, the webpage is rendered using a headless browser, allowing javascript to run and
                hydrate the page. If False, the webpage is scraped as-is.
            wait_for_selector: The selector to wait for before returning the HTML. Useful for when you know something
                should be on the page, but it is not there yet since it needs to be rendered by javascript. Only used if
                use_browser is True.

        """
        super().__init__(name=name)

        self.api_key = api_key
        self.rate_limit = rate_limit

        self.webpage_agent = WebpageAgent()
        self.session = requests.Session()

        # Parameters for the WebpageAgent
        self.text_only = text_only
        self.body_only = body_only
        self.use_browser = use_browser
        self.wait_for_selector = wait_for_selector

    def __repr__(self):
        return f"WebSearchAgent(name={self.name})"

    @staticmethod
    def _prepare_url(base_url: str, params: Dict) -> str:
        """
        This method prepares a URL for a request.

        Args:
            base_url: The base url.
            params: A dictionary of parameters to use for the request.

        Returns:
            The prepared URL.

        """
        req = requests.PreparedRequest()
        req.prepare_url(url=base_url, params=params)
        return req.url

    @staticmethod
    def _handle_errors(res: requests.Response) -> None:
        """
        This method handles errors that occur during a request.

        Args:
            res: The response from the request.

        """
        if res.status_code != 200:
            raise Exception(
                f"WebSearchAgent received a non-200 status code: {res.status_code}\n"
                f"{res.reason}"
            )

    def _send_request(
        self, base_url: str, headers: Dict = None, params: Dict = None
    ) -> Dict:
        """
        This method sends a request to a URL.

        Args:
            base_url: The base URL to send the request to.
            headers: A dictionary of headers to use for the request.
            params: A dictionary of parameters to use for the request.

        Returns:
            The response from the request.

        """
        url = self._prepare_url(base_url=base_url, params=params)

        res = self.session.get(url=url, headers=headers)

        self._handle_errors(res=res)

        return res.json()

    def search_brave(self, query: str, **kwargs) -> List[WebSearchResult]:
        """
        This method performs a web search using Brave.

        Get an API key here (credit card required):
        https://api.search.brave.com/register

        Args:
            query: The query to search for.
            **kwargs: Additional parameters to pass to the API.

        Returns:
            A list of WebSearchResult objects.

        """
        if kwargs is None:
            kwargs = {}

        headers = {"X-Subscription-Token": self.api_key, "Accept": "application/json"}
        params = {"q": query, **kwargs}

        res = self._send_request(
            base_url="https://api.search.brave.com/res/v1/web/search",
            headers=headers,
            params=params,
        )

        # https://api.search.brave.com/app/documentation/query
        categories = [
            "discussions",
            "faq",
            "infobox",
            "news",
            "query",
            "videos",
            "web",
            "mixed",
        ]
        results = []
        for category in categories:
            if category not in res:
                continue
            if "results" not in res[category]:
                continue
            for result in res[category]["results"]:
                # Rate limit
                time.sleep(self.rate_limit)

                # Get the content of the webpage
                try:
                    content = self.webpage_agent.scrape(
                        url=result["url"],
                        text_only=self.text_only,
                        body_only=self.body_only,
                        use_browser=self.use_browser,
                        wait_for_selector=self.wait_for_selector,
                    )
                except Exception:
                    # Skip when the webpage cannot be scraped.
                    continue

                results.append(
                    WebSearchResult(
                        title=result["title"],
                        url=result["url"],
                        description=result["description"],
                        content=content,
                    )
                )
        return results

    def search_google(
        self, query: str, custom_search_engine_id: str = None, **kwargs
    ) -> List[WebSearchResult]:
        """
        This method performs a web search using Google.

        Get an API key here:
        https://developers.google.com/custom-search/v1/overview

        You must create a custom search engine and pass its ID. To create or view custom search engines, visit:
        https://programmablesearchengine.google.com/u/1/controlpanel/all

        Args:
            query: The search query.
            custom_search_engine_id: The ID of the custom search engine to use.
            **kwargs: Any additional keyword arguments to pass to the API.

        Returns:
            A list of WebSearchResult objects.

        """
        if kwargs is None:
            kwargs = {}

        headers = {"Accept": "application/json"}

        params = {
            "q": query,
            "key": self.api_key,
            "cx": custom_search_engine_id,
            **kwargs,
        }

        res = self._send_request(
            base_url="https://www.googleapis.com/customsearch/v1",
            headers=headers,
            params=params,
        )

        results = []
        if "items" in res:
            for item in res["items"]:
                # Rate limit
                time.sleep(self.rate_limit)

                # Get the content of the webpage.
                content = "No content."
                try:
                    content = self.webpage_agent.scrape(
                        url=item["link"],
                        text_only=self.text_only,
                        body_only=self.body_only,
                        use_browser=self.use_browser,
                        wait_for_selector=self.wait_for_selector,
                    )
                except Exception:
                    # Skip when the webpage cannot be scraped.
                    continue

                title = "Untitled"
                if "title" in item:
                    title = item["title"]

                description = "No description available."
                if "snippet" in item:
                    description = item["snippet"]

                results.append(
                    WebSearchResult(
                        title=title,
                        url=item["link"],
                        description=description,
                        content=content,
                    )
                )

        return results


class RSSAgent(Agent):

    def __init__(self, name: str = "", url: str = None, **kwargs):
        """
        Create a RSSAgent

        This agent helps you read data from RSS feeds.

        Args:
            name: The name of the agent.
            url: The URL of the RSS feed.
            **kwargs: Any additional keyword arguments to pass to feedparser.parse(). You may need to pass a user agent
                header or other headers for some RSS feeds. See https://feedparser.readthedocs.io/en/latest/http.html.

        Examples:

            Read an RSS feed once, passing a user agent header:
                >>> from phasellm.agents import RSSAgent
                >>> agent = RSSAgent(url='https://arxiv.org/rss/cs', agent="it's me!")
                >>> data = agent.read()

            Poll the arXiv CS RSS feed every 60 seconds:
                >>> from phasellm.agents import RSSAgent
                >>> agent = RSSAgent(url='https://arxiv.org/rss/cs')
                >>> with agent.poll(interval=60) as poller:
                >>>     for data in poller():
                >>>         print(data)

            Poll the arXiv CS RSS feed every 60 seconds and stop after 5 minutes:
                >>> from phasellm.agents import RSSAgent
                >>> agent = RSSAgent(url='https://arxiv.org/rss/cs')
                >>> def poll_helper(p: Callable[[], Generator[List[Dict], None, None]]):
                >>>     for data in poller():
                >>>         print(data)
                >>> with agent.poll(interval=60) as poller:
                >>>     t = Thread(target=poll_helper, kwargs={'p': poller})
                >>>     t.start()
                >>>     time.sleep(300)
                >>> t.join()

            Poll and print the data and polling time after each update is received.
                >>> from phasellm.agents import RSSAgent
                >>> agent = RSSAgent(url='https://arxiv.org/rss/cs')
                >>> with agent.poll(interval=60) as poller:
                >>>     for data in poller():
                >>>         print(f'data: {data}')
                >>>         print(f'polling time: {agent.poll_time}')

        """
        if not url:
            raise Exception("Must provide a URL for the RSSAgent.")

        super().__init__(name=name)

        self.url = url
        self.kwargs = kwargs

        # Private attribute for tracking polling state of the agent.
        self._polling = False
        self._poll_start_time = None
        self._poll_end_time = None

    def __repr__(self):
        return f"RSSAgent(name={self.name})"

    @staticmethod
    def _yield_data(queue: Queue) -> Generator[List[Dict], None, None]:
        """
        This method is responsible for yielding data from the queue. It stops generating when it receives None.

        Args:
            queue: The queue to yield data from.

        Returns:
            A generator that yields data from the queue.

        """
        while True:
            data = queue.get(block=True)
            if data is None:
                break
            yield data

    def _poll_thread(self, queue: Queue, interval: int = 60) -> None:
        """
        This method is responsible for polling the RSS feed and putting new data in the queue.

        Args:
            queue: The queue to put data in.
            interval: The number of seconds to wait between polls.

        """
        last_item = None
        while self._polling:
            data = self.read()

            # Scrub through the data until we find the last item.
            for i in range(len(data)):
                if data[i] == last_item:
                    data = data[:i]
                    break

            # Put the data in the queue
            queue.put(data)

            # Update the last item
            if len(data) > 0:
                last_item = data[0]

            # Wait for interval seconds
            time.sleep(interval)
        # Signal the end of polling
        queue.put(None)

    def read(self) -> List[Dict]:
        """
        This method reads data from an RSS feed.

        Returns:
            A list of dictionaries containing the data from the RSS feed.

        """
        return feedparser.parse(self.url, **self.kwargs)["entries"]

    @contextmanager
    def poll(
        self, interval: int = 60
    ) -> Generator[Callable[[], Generator[List[str], None, None]], None, None]:
        """
        This method polls an RSS feed for new data.

        Args:
            interval: The number of seconds to wait between polls.

        Returns:
            A generator that yields a list of dictionaries containing the data from the RSS feed.

        """
        thread = None
        try:
            queue = Queue()
            self._polling = True
            thread = Thread(
                target=self._poll_thread, kwargs={"queue": queue, "interval": interval}
            )
            thread.start()
            self._poll_start_time = datetime.now()
            yield partial(self._yield_data, queue=queue)
        finally:
            if thread:
                self._polling = False
                self._poll_end_time = datetime.now()
                thread.join()

    @property
    def poll_time(self) -> timedelta:
        """
        This method calculates the amount of time the agent has been polling.

        Returns:
            A timedelta object.

        """

        if self._polling and self._poll_start_time is not None:
            return datetime.now() - self._poll_start_time
        if not self._polling and self._poll_start_time and self._poll_end_time:
            return self._poll_end_time - self._poll_start_time
        return timedelta(0)
