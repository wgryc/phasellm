import re

from typing import Dict, List, Pattern, Set

def _stringify(messages: List[Dict]) -> str:
    """
    Converts a list of message objects (such as { "role": str, "content": str }) into a flat string required for some upstream LLM APIs.
    This is influenced by the OpenAI chat completion API.
    """

    return "\n".join(
        [
            f"{str(message['role'])}: {str(message['content'])}"
            for message in messages
        ]
    )

def _find_stop_sequences(messages: List[Dict]) -> List[str]:
    """
    Generetes a list of stop sequence strings from the "role" fields in the message objects.
    """
    roles: Set[str] = set()
    
    for message in messages:
        role = message.get("role", None)

        if role:
            roles.add(role)
    
    stop_sequences: List[str] = [f"\n{role}:" for role in roles]
    
    return stop_sequences

class Prompt():
    """
    Prompts are sent to an LLM which generate some kind of response.

    Prompts can include replaceable variables (surrounded by curly brackets) which can be filled via the fill_vars() method.
    Variables make it easier to loop through prompts that follow a specific regular pattern or structure.
    For example:
    > Hello {name}!    
    """

    def __init__(self, prompt: str) -> None:
        self.prompt = prompt
        self.prompt_var_regex: Pattern = r'\{\s*[a-zA-Z0-9_]+\s*\}'

    def __repr__(self) -> str:
        return self.prompt

    # TODO: Remove if not used or if deprecated
    def get_prompt(self) -> str:
        """
        Return the raw prompt command (i.e., does not fill in variables.)
        """
        return self.prompt
    
    def fill_vars(self, **kwargs) -> str:
        """
        Fill in variables (surrounded by curly brackets) in the prompt.
        """
        
        prompt_vars = re.findall(self.prompt_var_regex, self.prompt)
        filled_prompt = self.prompt

        for prompt_var in prompt_vars:
            keyword = prompt_var.replace("{", "").replace("}", "").strip()
            
            if keyword in kwargs:
                filled_prompt = filled_prompt.replace(prompt_var, kwargs[keyword])
    
        return filled_prompt

class ChatPrompt():
    """
    Prompts are sent to an LLM which generate some kind of response.
    Prompts which are used for longer chats with an upstream LLM also include `messages` to provide instructions to an LLM.

    Prompts can include replaceable variables (surrounded by curly brackets) which can be filled via the fill_vars() method.
    Variables make it easier to loop through prompts that follow a specific regular pattern or structure.
    For example:
    > Hello {name}!   
    """

    def __init__(self, messages: List[Dict]=[]) -> None:
        self.messages = messages
        self.prompt_var_regex: Pattern = r'\{\s*[a-zA-Z0-9_]+\s*\}'

    def __repr__(self) -> str:
        return "ChatPrompt()"
    
    # TODO: Remove if not used or if deprecated
    def chat_repr(self) -> str:
        return _stringify(messages=self.messages)
    
    def fill_vars(self, **kwargs) -> List[Dict]:
        filled_messages: List[Dict] = []

        for i in range(0, len(self.messages)):
            role = self.messages[i]["role"]
            content = self.messages[i]["content"]

            role_vars = re.findall(self.prompt_var_regex, role)
            filled_role = role

            for role_var in role_vars:
                keyword = role_var.replace("{", "").replace("}", "").strip()

                if keyword in kwargs:
                    filled_role = filled_role.replace(role_var, kwargs[keyword])

            content_vars = re.findall(self.prompt_var_regex, content)
            filled_content = content 

            for content_var in content_vars:
                keyword = content_var.replace("{", "").replace("}", "").strip()

                if keyword in kwargs:
                    filled_content = filled_content.replace(content_var, kwargs[keyword])

            filled_messages.append({"role": filled_role, "content": filled_content})

        return filled_messages
