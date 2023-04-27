"""
Agents to help with workflows.
"""

import sys
from io import StringIO
import contextlib

from .exceptions import LLMCodeException

class Agent():
    """
    Abstract class for agents.
    """

    def __init__(self, name=''):
        self.name = name
        pass 

    def __repr__(self):
        return f"Agent(name='{self.name}')"
    
@contextlib.contextmanager
def stdoutIO(stdout=None):
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

    def __init__(self, name=''):
        self.name = name 

    def execute_code(self, code, globals=None, locals=None):
        """
        Executes arbitrary Python code and saves the output (or error!) to a variable.
        
        Returns the variable and a boolean (is_error) depending on whether an error took place.
        """
        is_error = False
        with stdoutIO() as s:
            try:
                exec(code, globals, locals)
            except Exception as err:
                raise LLMCodeException(code, str(err))

        return s.getvalue()
