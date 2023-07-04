import os
import time
import shutil
import random

import docker.errors

from pathlib import Path

from unittest import TestCase

from phasellm.exceptions import LLMCodeException

from phasellm.agents import SandboxedCodeExecutionAgent, WebpageAgent


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
        time.sleep(random.random())

    def test_scrape_single_html_text(self):
        text = self.fixture.scrape(
            url='https://www.cbc.ca/news/canada/google-facebook-canadian-news-1.6894029',
            text_only=True
        )
        self.assertTrue(
            'Government says law will apply to companies with' in
            text, f"Text does not contain expected string.\n{text}"
        )

    def test_scrape_single_html(self):
        text = self.fixture.scrape(
            url='https://www.cbc.ca/news/canada/google-facebook-canadian-news-1.6894029',
            text_only=False
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
            use_javascript=True
        )
        self.assertTrue(
            'Go to file\n</a>'
            in text, f"Text does not contain expected string.\n{text}"
        )

    def test_scrape_single_html_text_javascript(self):
        text = self.fixture.scrape(
            url='https://github.com/facebook/react',
            text_only=True,
            use_javascript=True
        )
        self.assertTrue(
            'Go to file'
            in text, f"Text does not contain expected string.\n{text}"
        )

    def test_scrape_single_xml_text(self):
        text = self.fixture.scrape(
            url='https://www.w3schools.com/xml/note.xml',
            text_only=True
        )
        self.assertTrue(
            'Tove'
            in text, f"Text does not contain expected string.\n{text}"
        )

    def test_scrape_single_xml(self):
        text = self.fixture.scrape(
            url='https://www.w3schools.com/xml/note.xml',
            text_only=False
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
                text_only=True
            )
        except ValueError:
            exception = True

        self.assertTrue(exception, "Expected ValueError, got nothing.")

    def test_scrape_multiple(self):
        urls = [
            'https://www.cbc.ca/news/canada/google-facebook-canadian-news-1.6894029',
            'https://arxiv.org/abs/2306.17759'
        ]
        for url in urls:
            text = self.fixture.scrape(url=url, text_only=True)
            self.assertTrue(
                len(text) > 0,
                f"Text is empty.\n{text}"
            )
