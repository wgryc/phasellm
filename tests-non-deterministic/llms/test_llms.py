import os

from unittest import TestCase

from dotenv import load_dotenv

from phasellm.llms import OpenAIGPTWrapper, ChatBot

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")


class TestChatBot(TestCase):

    def test_openai_gpt_chat_temperature(self):
        prompt = 'What is the capital of Jupiter?'
        verbose = True

        # Test low temperature
        llm = OpenAIGPTWrapper(openai_api_key, "gpt-3.5-turbo", temperature=0)
        fixture = ChatBot(llm)
        low_temp_res = fixture.chat(prompt)

        # Test high temperature
        llm = OpenAIGPTWrapper(openai_api_key, "gpt-3.5-turbo", temperature=2)
        fixture = ChatBot(llm)
        high_temp_res = fixture.chat(prompt)

        if verbose:
            print(f'Low temp response:\n{low_temp_res}')
            print(f'Low temperature len: {len(low_temp_res)}')

            print(f'High temp response:\n{high_temp_res}')
            print(f'High temperature len: {len(high_temp_res)}')

        # Responses should differ.
        self.assertNotEqual(low_temp_res, high_temp_res)

        # High temperature should generally produce longer responses.
        self.assertTrue(len(low_temp_res) < len(high_temp_res))
