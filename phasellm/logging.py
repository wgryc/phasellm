"""
Logging support. This allows you to use the phasellm library to send chats to evals.phasellm.com and review them via our hosted front-end.
"""

import requests
import json 

from typing import List, Optional

from .llms import Message, ChatBot

_PHASELLM_EVALS_BASE_URL = "https://evals.phasellm.com/api/v0.1"

class PhaseLogger:

    def __init__(
            self,
            apikey: str,
    ):
        """
        Helper class for logging chats to evals.phasellm.com.

        Args:
            apikey: The API key associated with your evals.phasellm.com account.
        """
        super().__init__()
        self.apikey = apikey

    def log(self, messages:List[Message], chat_id:Optional[int] = None, title:Optional[str] = None, source_id:Optional[str] = None) -> int:
        """
        Saves or updates the relevant chat at evals.phasellm.com 

        Args:
            messages: The messages array from the chat.
            chat_id: Optional chat ID. If you provide a chat ID from an earlier log event, the messages will overwrite the original chat. This should be used for updating conversations rather than replacing them.
            title: Optional title for the chat.
            source_id: Optional String representing an ID for the chat. This is to enable easier referencing of chats for end users and is not used by PhaseLLM Evals.
            
        Returns:
            The chat_id associated with the chat.
        """

        save_url = _PHASELLM_EVALS_BASE_URL + "/save_chat"
        headers = {
            'Authorization': f'Bearer {self.apikey}',
            'Content-Type': 'application/json'
        }
        payload = {"messages": messages}
        if chat_id is not None:
            payload["chat_id"] = chat_id

        if title is not None:
            payload["title"] = title 

        if source_id is not None:
            payload["source_id"] = source_id

        response = requests.post(save_url, json=payload, headers=headers)
        data = json.loads(response.text)
        if data['status'] == "error":
            raise Exception(f"PhaseLLM Evals: an error occured. {data['message']}")
        
        return data['chat_id']
    
    def logChatBot(self, chatbot:ChatBot, chat_id:Optional[int] = None, title:Optional[str] = None, source_id:Optional[str] = None) -> int:
        message_array = []
        for m in chatbot.messages:
            new_m = {"role": m["role"], "content":m["content"]}
            message_array.append(new_m)
        return self.log(message_array, chat_id, title, source_id)