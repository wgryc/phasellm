from unittest import TestCase

from phasellm.llms import Prompt, OpenAIGPTWrapper, StreamingOpenAIGPTWrapper


class TestPrompt(TestCase):

    def test_prompt_fill(self):
        p = "1: {fill_1}, 2: {fill_2}, 3: {fill_3}"
        prompt = Prompt(p)

        actual = prompt.fill(fill_1="one", fill_2="two", fill_3="three")

        expected = "1: one, 2: two, 3: three"

        self.assertEqual(actual, expected, f"{actual} != {expected}")


class TestOpenAIGPTWrapper(TestCase):
    CONFIG_ERROR = 'Must pass apikey or api_config. If using kwargs, check capitalization.'

    def test_config_error_incorrect_kwarg(self):
        error = False
        try:
            self.fixture = OpenAIGPTWrapper(apiKey='test')
        except Exception as e:
            self.assertEqual(e.__str__(), self.CONFIG_ERROR)
            error = True
        self.assertTrue(error, 'Expected error to occur.')

    def test_config_error_missing_config(self):
        error = False
        try:
            self.fixture = OpenAIGPTWrapper()
        except Exception as e:
            self.assertEqual(e.__str__(), self.CONFIG_ERROR)
            error = True
        self.assertTrue(error, 'Expected error to occur.')
