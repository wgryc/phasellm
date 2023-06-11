from unittest import TestCase

from phasellm.llms import Prompt


class TestPrompt(TestCase):

    def test_prompt_fill(self):
        p = "1: {fill_1}, 2: {fill_2}, 3: {fill_3}"
        prompt = Prompt(p)

        actual = prompt.fill(fill_1="one", fill_2="two", fill_3="three")

        expected = "1: one, 2: two, 3: three"

        self.assertEqual(actual, expected, f"{actual} != {expected}")
