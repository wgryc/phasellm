import os

from typing import Generator

from unittest import TestCase

from dotenv import load_dotenv

from phasellm.llms import StreamingOpenAIGPTWrapper, StreamingClaudeWrapper, ChatBot

from tests.e2e.llms.utils import \
    StreamingChatCompletionProbe, probe_streaming_chat_completion, common_streaming_chat_assertions, \
    StreamingTextCompletionProbe, probe_streaming_text_completion, common_streaming_text_assertions

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")


class E2ETestStreamingOpenAIGPTWrapper(TestCase):

    def test_complete_chat(self):
        fixture = StreamingOpenAIGPTWrapper(openai_api_key, model="gpt-3.5-turbo")

        messages = [{"role": "user", "content": "What should I eat for lunch today?"}]
        generator = fixture.complete_chat(messages, append_role='assistant')

        results: StreamingChatCompletionProbe = probe_streaming_chat_completion(generator)

        common_streaming_chat_assertions(self, results, chunk_time_seconds_threshold=0.1, verbose=True)

    def test_text_completion_success(self):
        fixture = StreamingOpenAIGPTWrapper(openai_api_key, model="text-davinci-003")

        prompt = "Three countries in North America are: "
        generator = fixture.text_completion(prompt)

        result: StreamingTextCompletionProbe = probe_streaming_text_completion(generator)

        common_streaming_text_assertions(self, result, verbose=True)

    def test_text_completion_failure(self):
        fixture = StreamingOpenAIGPTWrapper(openai_api_key, model="gpt-3.5-turbo")

        prompt = "The capital of Canada is"
        generator = fixture.text_completion(prompt)

        self.assertTrue(isinstance(generator, Generator), "Expecting a generator.")

        exception = None
        try:
            # Convert the generator to a list to evaluate it.
            list(generator)
        except Exception as e:
            exception = e

        print(f'Exception:\n{exception}')

        self.assertTrue(exception is not None, "Expecting an exception.")


class E2ETestStreamingClaudeWrapper(TestCase):

    def test_complete_chat(self):
        fixture = StreamingClaudeWrapper(anthropic_api_key, model="claude-v1")

        messages = [{"role": "user", "content": "What should I eat for lunch today?"}]
        generator = fixture.complete_chat(messages, append_role='assistant')

        self.assertTrue(isinstance(generator, Generator), "Expecting a generator.")

        results: StreamingChatCompletionProbe = probe_streaming_chat_completion(generator)

        common_streaming_chat_assertions(self, results, chunk_time_seconds_threshold=0.1, verbose=True)

    def test_text_completion_success(self):
        fixture = StreamingClaudeWrapper(anthropic_api_key, model="claude-v1")

        prompt = "Three countries in North America are: "
        generator = fixture.text_completion(prompt)

        self.assertTrue(isinstance(generator, Generator), "Expecting a generator.")

        result = probe_streaming_text_completion(generator)

        common_streaming_text_assertions(self, result, verbose=True)


class E2ETestStreamingChatBot(TestCase):

    def test_chat(self):
        llm = StreamingOpenAIGPTWrapper(openai_api_key, model="gpt-3.5-turbo")
        fixture = ChatBot(llm)
        generator = fixture.chat('Who are you')

        self.assertTrue(isinstance(generator, Generator), "Expecting a generator.")

        # Check the results of the generator.
        results: StreamingChatCompletionProbe = probe_streaming_chat_completion(generator)
        common_streaming_chat_assertions(self, results, chunk_time_seconds_threshold=0.2, verbose=False)

        # Check the state of the ChatBot.
        self.assertTrue(
            len(fixture.messages) > 1,
            "Expecting more than one message in the stack."
        )
        self.assertTrue(
            fixture.messages[-1]['role'] == 'assistant',
            "Expecting the last message to be from the assistant."
        )

        # Check that the generator executed by ChatBot.chat() stores the realized result in the stack.
        self.assertTrue(
            len(fixture.messages[-1]['content']) != 0,
            "Expecting the last message to have content."
        )
        self.assertTrue(
            fixture.messages[-1]['content'] == results.res,
        )

        # Make another call to ChatBot.chat() to ensure it is capable of receiving a new message.
        generator = fixture.chat('Where do you come from?')

        # Check the results of the generator.
        results: StreamingChatCompletionProbe = probe_streaming_chat_completion(generator)
        common_streaming_chat_assertions(self, results, chunk_time_seconds_threshold=0.1, verbose=True)

        # Ensure there are 5 messages in the stack. (1 system, 2 user, 2 assistant)
        self.assertTrue(
            len(fixture.messages) == 5,
            "Expecting 5 messages in the stack (1 system, 2 user, 2 assistant)."
        )
        # Check that the messages are in the correct order.
        self.assertTrue(
            fixture.messages[0]['role'] == 'system',
            "Expecting the first message to be from the system."
        )
        self.assertTrue(
            fixture.messages[1]['role'] == 'user',
            "Expecting the first message to be from the user."
        )
        self.assertTrue(
            fixture.messages[2]['role'] == 'assistant',
            "Expecting the second message to be from the assistant."
        )
        self.assertTrue(
            fixture.messages[3]['role'] == 'user',
            "Expecting the third message to be from the user."
        )
        self.assertTrue(
            fixture.messages[4]['role'] == 'assistant',
            "Expecting the last message to be from the assistant."
        )

        print(f'ChatBot messages:\n{fixture.messages}')