import os
import time
import types

from unittest import TestCase

from dotenv import load_dotenv

from phasellm.llms import StreamingOpenAIGPTWrapper

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")


class E2ETestStreamingOpenAIGPTWrapper(TestCase):

    def setUp(self) -> None:
        self.fixture = StreamingOpenAIGPTWrapper(openai_api_key)

    def tearDown(self) -> None:
        self.fixture = None

    def test_complete_chat(self):
        messages = [{"role": "user", "content": "What should I eat for lunch today?"}]
        generator = self.fixture.complete_chat(messages, 'assistant')

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

    # def test_text_completion(self):
    #     prompt = "The capital of Poland is"
    #     generator = self.fixture.text_completion(prompt)
