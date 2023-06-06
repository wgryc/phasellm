import os
import time
import types

from unittest import TestCase

from dotenv import load_dotenv

from phasellm.llms import StreamingOpenAIGPTWrapper

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")


class E2ETestStreamingOpenAIGPTWrapper(TestCase):

    def test_complete_chat(self):
        fixture = StreamingOpenAIGPTWrapper(openai_api_key, model="gpt-3.5-turbo")
        messages = [{"role": "user", "content": "What should I eat for lunch today?"}]
        generator = fixture.complete_chat(messages, 'assistant')

        self.assertTrue(isinstance(generator, types.GeneratorType), "Expecting a generator.")

        res = ''
        chunk_times = []
        time_start = time.time()
        for chunk in generator:
            # Track the time it takes to generate each chunk.
            time_end = time.time()
            chunk_times.append(time_end - time_start)
            time_start = time_end

            # Print the response length so far.
            res = res + chunk
            print(f'Generated: {len(res)} characters.')

        # Compute the mean chunk time
        mean_chunk_time = sum(chunk_times) / len(chunk_times)
        print(f'Mean chunk time: {mean_chunk_time} seconds.')

        self.assertTrue(len(res) > 0, "Expecting a non-empty response.")
        self.assertTrue(len(chunk_times) > 1, "Expecting more than one chunk.")
        self.assertTrue(mean_chunk_time < 0.1, "Expecting a mean chunk time of less than 0.1 seconds.")

        print(f'Result:\n{res}')

    def test_text_completion_success(self):
        fixture = StreamingOpenAIGPTWrapper(openai_api_key, model="text-davinci-003")

        prompt = "Three countries in North America are: "
        generator = fixture.text_completion(prompt)

        self.assertTrue(isinstance(generator, types.GeneratorType), "Expecting a generator.")

        res = ''
        chunk_count = 0
        for chunk in generator:
            chunk_count += 1
            res = res + chunk

        print(f'Result:\n{res}')
        print(f'Chunk count: {chunk_count}')

        self.assertTrue(chunk_count > 1, "Expecting more than one chunk.")
        self.assertTrue(len(res) > 0, "Expecting a non-empty response.")

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
