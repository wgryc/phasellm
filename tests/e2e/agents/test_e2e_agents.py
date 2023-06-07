from unittest import TestCase

from phasellm.agents import SandboxedCodeExecutionAgent


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
