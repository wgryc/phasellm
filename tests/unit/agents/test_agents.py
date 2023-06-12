from unittest import TestCase

from unittest.mock import MagicMock, patch

from phasellm.agents import SandboxedCodeExecutionAgent


class TestSandboxedCodeExecutionAgent(TestCase):

    @patch('phasellm.agents.SandboxedCodeExecutionAgent._code_to_temp_file')
    def test_execute_code(self, _):
        """
        Test that the agent executes code with the correct parameters.
        """
        # Set up fixture and mock the client so that we can make assertions about the parameters passed to it.
        fixture = SandboxedCodeExecutionAgent()
        fixture.client = MagicMock()

        # Execute the code.
        fixture.execute_code('print("Hello, world!")')

        '''
        Ensure that the agent is called with the stream and auto_remove parameters.
        
        auto_remove is critically important because it ensures that the container is removed after execution. Without 
        it, containers can build up and consume too much disk space on the host machine.
        '''
        self.assertTrue(fixture.client.containers.run.called_with(
            stream=True,
            auto_remove=True
        ), "The agent was not called with the correct parameters.")
