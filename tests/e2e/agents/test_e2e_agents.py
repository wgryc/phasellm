import os
import time
import shutil

import docker.errors

from pathlib import Path

from threading import Thread

from unittest import TestCase

from dotenv import load_dotenv

from tests.utils import Timeout

from phasellm.exceptions import LLMCodeException

from typing import Callable, List, Dict, Generator

from phasellm.agents import SandboxedCodeExecutionAgent, WebpageAgent, WebSearchResult, WebSearchAgent, RSSAgent

load_dotenv()
google_search_api_key = os.getenv("GOOGLE_SEARCH_API_KEY")
google_search_engine_id = os.getenv("GOOGLE_SEARCH_ENGINE_ID")
brave_search_api_key = os.getenv("BRAVE_SEARCH_API_KEY")


class TestE2ESandboxedCodeExecutionAgent(TestCase):

    def setUp(self) -> None:
        """
        Runs before every test.
        """
        # Set the temp_path for the agent to use.
        self.scratch_dir = Path('./.tmp')

    def tearDown(self) -> None:
        """
        Runs after every test.
        """
        # Delete the .tmp scratch directory if it exist.
        if os.path.isdir(self.scratch_dir):
            shutil.rmtree(self.scratch_dir)

    def test_execute_code_stream_result(self):
        code = (
            "import time\n"
            "print('Hello, world!')\n"
            "time.sleep(1)\n"
            "print('Hello again')"
        )

        expected = ['Hello, world!\n', 'Hello again\n']
        with SandboxedCodeExecutionAgent(scratch_dir=self.scratch_dir) as fixture:
            logs = fixture.execute_code(code, stream=True)
            for i, log in enumerate(logs):
                self.assertTrue(log == expected[i], f"{log}\n!=\n{expected[i]}")

    def test_execute_code_concat_result(self):
        code = (
            "print('0')\n"
            "print('1')"
        )

        with SandboxedCodeExecutionAgent() as fixture:
            actual = fixture.execute_code(code, stream=False)

        expected = '0\n1\n'
        self.assertTrue(actual == expected, f"{actual}\n!=\n{expected}")

    def test_execute_code_external_package(self):
        code = (
            'import numpy as np\n'
            'print(np.array([1, 2, 3]))'
        )

        with SandboxedCodeExecutionAgent() as fixture:
            actual = fixture.execute_code(code, stream=False)

        expected = '[1 2 3]\n'

        self.assertTrue(actual == expected, f"{actual}\n!=\n{expected}")

    def test_execute_code_external_package_fail(self):
        code = (
            'import fake_package\n'
            'print(test)'
        )

        expected_exception_contains = "ModuleNotFoundError: No module named 'fake_package'"
        exception = False
        try:
            with SandboxedCodeExecutionAgent() as fixture:
                _ = fixture.execute_code(code, stream=False)
        except LLMCodeException as e:
            exception = True
            self.assertTrue(
                e.exception_string.__contains__(expected_exception_contains),
                f"{e.exception_string}\n!=\n{expected_exception_contains}"
            )

        self.assertTrue(exception, "Expected LLMCodeException, got nothing.")

    def test_execute_code_no_context_manager(self):
        code = (
            "print('Hello, world!')"
        )
        expected = 'Hello, world!\n'

        fixture = SandboxedCodeExecutionAgent()
        try:
            logs = fixture.execute_code(code, stream=False)

            # Check the output
            self.assertTrue(logs == expected, f"\n{logs}\n!=\n{expected}")

            # Check that the container is still running
            self.assertTrue(fixture._container.status == 'created', f"{fixture._container.status} != created")

            # Get the container name for the next assertions
            container_name = fixture._container.name

            # Close the client & container
            fixture.close()

            # Check that the container was removed from the object.
            self.assertTrue(fixture._container is None, f"Container should be None, got {fixture._container}")

            # Check that the container was shut down.
            container_not_found = True
            try:
                fixture._client.containers.get(container_name)
            except docker.errors.NotFound:
                container_not_found = False
            self.assertFalse(container_not_found, f"Container {container_name} should not exist.")
        except Exception as e:
            fixture.close()
            raise e

    def test_execute_code_multiple_executions_one_container(self):
        """
        Test that multiple code executions can be run on the same container.
        Returns:

        """
        code_1 = (
            "print('1')"
        )
        code_2 = (
            "print('2')"
        )

        with SandboxedCodeExecutionAgent(scratch_dir=self.scratch_dir) as fixture:
            expected = "1\n"
            logs = fixture.execute_code(code_1, stream=False)
            self.assertTrue(logs == expected, f"\n{logs}\n!=\n{expected}")
            container_name_1 = fixture._container.name

            expected = "2\n"
            logs = fixture.execute_code(code_2, stream=False)
            self.assertTrue(logs == expected, f"\n{logs}\n!=\n{expected}")
            container_name_2 = fixture._container.name

            self.assertTrue(container_name_1 == container_name_2, f"{container_name_1} != {container_name_2}")

    def test_execute_code_multiple_executions_multiple_containers(self):
        """
        Test that multiple code executions can be run on different containers.
        Returns:

        """
        code_1 = (
            "print('1')"
        )
        code_2 = (
            "print('2')"
        )

        with SandboxedCodeExecutionAgent(scratch_dir=self.scratch_dir) as fixture:
            expected = "1\n"
            logs = fixture.execute_code(code_1, stream=False, auto_stop_container=True)
            self.assertTrue(logs == expected, f"\n{logs}\n!=\n{expected}")
            self.assertTrue(fixture._container is None, f"Container should be None, got {fixture._container}")

            expected = "2\n"
            logs = fixture.execute_code(code_2, stream=False, auto_stop_container=True)
            self.assertTrue(logs == expected, f"\n{logs}\n!=\n{expected}")
            self.assertTrue(fixture._container is None, f"Container should be None, got {fixture._container}")


class TestE2EWebpageAgent(TestCase):

    def setUp(self):
        self.fixture = WebpageAgent()

    def test_scrape_single_html_text(self):
        text = self.fixture.scrape(
            url='https://www.cbc.ca/news/canada/google-facebook-canadian-news-1.6894029',
            text_only=True,
            body_only=False,
            use_browser=False
        )
        self.assertTrue(
            "Government says law will apply to companies with 'significant bargaining power imbalance'" in
            text, f"Text does not contain expected string.\n{text}"
        )

    def test_scrape_single_html(self):
        text = self.fixture.scrape(
            url='https://www.cbc.ca/news/canada/google-facebook-canadian-news-1.6894029',
            text_only=False,
            body_only=False,
            use_browser=False
        )
        self.assertTrue(
            '<title data-rh="true">When will Canadian news disappear from Google, Facebook? What the Bill C-18 rift '
            'means for you | CBC News</title>'
            in text, f"Text does not contain expected string.\n{text}"
        )

    def test_scrape_single_html_javascript(self):
        text = self.fixture.scrape(
            url='https://github.com/facebook/react',
            text_only=False,
            body_only=False,
            use_browser=True
        )
        self.assertTrue(
            'Go to file\n</a>'
            in text, f"Text does not contain expected string.\n{text}"
        )

    def test_scrape_single_html_text_javascript(self):
        text = self.fixture.scrape(
            url='https://github.com/facebook/react',
            text_only=True,
            body_only=False,
            use_browser=True
        )
        self.assertTrue(
            'Go to file'
            in text, f"Text does not contain expected string.\n{text}"
        )

    def test_scrape_single_xml_text(self):
        text = self.fixture.scrape(
            url='https://www.w3schools.com/xml/note.xml',
            text_only=True,
            body_only=False,
            use_browser=False
        )
        self.assertTrue(
            'Tove'
            in text, f"Text does not contain expected string.\n{text}"
        )

    def test_scrape_single_xml(self):
        text = self.fixture.scrape(
            url='https://www.w3schools.com/xml/note.xml',
            text_only=False,
            body_only=False,
            use_browser=False
        )
        self.assertTrue(
            '<to>Tove</to>'
            in text, f"Text does not contain expected string.\n{text}"
        )

    def test_scrape_invalid(self):
        exception = False
        try:
            self.fixture.scrape(
                url='https://arxiv.org/pdf/2306.17759.pdf',
                text_only=True,
                body_only=False,
                use_browser=False
            )
        except Exception:
            exception = True

        self.assertTrue(exception, "Expected ValueError, got nothing.")

    def test_scrape_multiple(self):
        urls = [
            'https://www.cbc.ca/news/canada/google-facebook-canadian-news-1.6894029',
            'https://arxiv.org/abs/2306.17759'
        ]
        for url in urls:
            text = self.fixture.scrape(
                url=url,
                text_only=True,
                body_only=False,
                use_browser=False
            )
            self.assertTrue(
                len(text) > 0,
                f"Text is empty.\n{text}"
            )

    def test_scrape_single_html_text_only_body_only(self):
        text = self.fixture.scrape(
            url='https://10millionsteps.com/ai-inflection',
            text_only=True,
            body_only=True,
            use_browser=False
        )
        self.assertTrue(
            'There are two broad types of risks we need to consider in this AI-enabled future.\n'
            'Extrinsic risks' in text,
            f"Text does not contain expected string.\n{text}"
        )

    def test_scrape_single_html_only_body(self):
        text = self.fixture.scrape(
            url='https://10millionsteps.com/ai-inflection',
            text_only=False,
            body_only=True,
            use_browser=False
        )
        self.assertTrue(
            '<p>There are two broad types of risks we need to consider in this AI-enabled future.</p>\n'
            '<p><em>Extrinsic risks' in text,
            f"Text does not contain expected string.\n{text}"
        )


class TestE2EWebSearchAgent(TestCase):

    def test_search_google(self):
        self.assertTrue(brave_search_api_key is not None, "Brave search API key is not set.")
        self.assertTrue(google_search_engine_id is not None, "Google search engine ID is not set.")

        self.fixture = WebSearchAgent(
            api_key=google_search_api_key
        )
        res = self.fixture.search_google(
            query='test',
            custom_search_engine_id=google_search_engine_id
        )

        self.assertTrue(
            len(res) > 0,
            f"Result is empty.\n{res}"
        )
        self.assertTrue(
            isinstance(res[0], WebSearchResult),
            f"Result is not of type WebSearchResult.\n{res}"
        )

    def test_search_brave(self):
        self.assertTrue(brave_search_api_key is not None, "Brave search API key is not set.")

        self.fixture = WebSearchAgent(
            api_key=brave_search_api_key
        )
        res = self.fixture.search_brave(
            query='test'
        )

        self.assertTrue(
            len(res) > 0,
            f"Result is empty.\n{res}"
        )
        self.assertTrue(
            isinstance(res[0], WebSearchResult),
            f"Result is not of type WebSearchResult.\n{res}"
        )


class TestE2ERSSAgent(TestCase):

    def setUp(self) -> None:
        self.fixture = RSSAgent(url='https://news.ycombinator.com/rss')
        time.sleep(3)

    def test_read_success(self):
        data = self.fixture.read()
        self.assertTrue(
            len(data) > 0,
            f"Result is empty.\n{data}"
        )

    def test_read_failure(self):
        self.fixture.url = 'https://arxiv.org/rss/doesnotexist'
        data = self.fixture.read()
        self.assertTrue(len(data) == 0, "Expected empty result, got something.")

    def test_poll_1_second(self):

        results = []

        def _poll_helper(p: Callable[[], Generator[List[Dict], None, None]]):
            for data in p():
                results.extend(data)

        # Execute the poller for 1 second.
        # Arxiv has a 3 request/second rate limit.
        with self.fixture.poll(interval=3) as poller:
            thread = Thread(target=_poll_helper, kwargs=({'p': poller}))
            thread.start()
            time.sleep(1)
        thread.join()

        self.assertTrue(
            len(results) > 0,
            f"Result is empty.\n{results}"
        )

    def test_poll_10_results(self):

        results = []

        timeout = Timeout(seconds=5)
        timeout.start()

        # Get 10 results.
        # Arxiv has a 3 request/second rate limit.
        with self.fixture.poll(interval=3) as poller:
            for data in poller():
                results.extend(data)
                if len(results) >= 10:
                    timeout.stop()
                    break
                timeout.check()

        self.assertTrue(
            len(results) >= 10,
            f"{len(results)} != 10"
        )

    def test_poll_time(self):
        """
        This method tests that the poll_time property is set correctly.
        Returns:

        """

        def _poll_helper(p: Callable[[], Generator[List[Dict], None, None]]):
            for _ in p():
                pass

        timeout = Timeout(seconds=5)
        timeout.start()

        # Poll for 1 second.
        # Arxiv has a 3 request/second rate limit.
        with self.fixture.poll(interval=3) as poller:
            thread = Thread(target=_poll_helper, kwargs=({'p': poller}))
            thread.start()
            time.sleep(1)
            timeout.check()
        thread.join()

        self.assertTrue(
            self.fixture.poll_time.seconds >= 1,
            f"{self.fixture.poll_time.seconds} is not >= 1"
        )
        self.assertTrue(
            self.fixture.poll_time.seconds < 2,
            f"{self.fixture.poll_time.seconds} is not < 2"
        )
