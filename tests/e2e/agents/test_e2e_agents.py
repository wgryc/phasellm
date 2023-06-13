import os
import shutil

from pathlib import Path

from unittest import TestCase

from phasellm.agents import CodeExecutionAgent, SandboxedCodeExecutionAgent, EmailSenderAgent, NewsSummaryAgent


class TestE2ECodeExecutionAgent(TestCase):

    def test_execute_code(self):
        pass


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
            logs = fixture.execute_code(code)
            for i, log in enumerate(logs):
                actual = log.decode('utf-8')
                self.assertTrue(actual == expected[i], f"{actual}\n!=\n{expected[i]}")

    def test_execute_code_concat_result(self):
        code = (
            "print('0')\n"
            "print('1')"
        )

        with SandboxedCodeExecutionAgent() as fixture:
            logs = fixture.execute_code(code)
            actual = b''.join(logs).decode('utf-8')

        expected = '0\n1\n'
        self.assertTrue(actual == expected, f"{actual}\n!=\n{expected}")

    def test_execute_code_external_package(self):
        code = (
            'import numpy as np\n'
            'print(np.array([1, 2, 3]))'
        )

        with SandboxedCodeExecutionAgent() as fixture:
            logs = fixture.execute_code(code)
            actual = b''.join(logs).decode('utf-8')

        expected = '[1 2 3]\n'

        self.assertTrue(actual == expected, f"{actual}\n!=\n{expected}")


class TestE2EEmailSenderAgent(TestCase):

    def test_send_plain_email(self):
        pass

    def test_send_plain_email_deprecated(self):
        pass


class TestE2ENewsSummaryAgent(TestCase):

    def test_get_query(self):
        pass

    def test_get_query_deprecated(self):
        pass
