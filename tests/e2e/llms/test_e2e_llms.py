import os
import types

from unittest import TestCase

from dotenv import load_dotenv

from phasellm.llms import StreamingOpenAIGPTWrapper, StreamingClaudeWrapper

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

        self.assertTrue(isinstance(generator, types.GeneratorType), "Expecting a generator.")

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

        self.assertTrue(isinstance(generator, types.GeneratorType), "Expecting a generator.")

        results: StreamingChatCompletionProbe = probe_streaming_chat_completion(generator)

        common_streaming_chat_assertions(self, results, chunk_time_seconds_threshold=0.1, verbose=True)

    def test_text_completion_success(self):
        fixture = StreamingClaudeWrapper(anthropic_api_key, model="claude-v1")

        prompt = "Three countries in North America are: "
        generator = fixture.text_completion(prompt)

        self.assertTrue(isinstance(generator, types.GeneratorType), "Expecting a generator.")

        result = probe_streaming_text_completion(generator)

        common_streaming_text_assertions(self, result, verbose=True)
