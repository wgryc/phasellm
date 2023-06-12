from unittest import TestCase

from phasellm.agents import CodeExecutionAgent, SandboxedCodeExecutionAgent, EmailSenderAgent, NewsSummaryAgent


class TestE2ECodeExecutionAgent(TestCase):

    def test_execute_code(self):
        pass


class TestE2ESandboxedCodeExecutionAgent(TestCase):

    def test_execute_code_stream_result(self):
        code = "print('Hello, world!')\nprint('Hello again')"
        with SandboxedCodeExecutionAgent() as fixture:
            logs = fixture.execute_code(code)
            for log in logs:
                print(log.decode('utf-8'))

    def test_execute_code_concat_result(self):
        code = "print('0')\nprint('1')"
        with SandboxedCodeExecutionAgent() as fixture:
            logs = fixture.execute_code(code)
            logs = b''.join(logs)
            print(logs.decode('utf-8'))

    def test_execute_code_external_package(self):
        pass


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
