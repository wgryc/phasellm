import os

from unittest import TestCase

from dotenv import load_dotenv

from phasellm.llms import \
    HuggingFaceInferenceWrapper, \
    BloomWrapper, \
    OpenAIGPTWrapper, StreamingOpenAIGPTWrapper, \
    ClaudeWrapper, StreamingClaudeWrapper, \
    GPT2Wrapper, \
    DollyWrapper, \
    CohereWrapper, \
    ChatBot

# LLM wrapper tests
from tests.e2e.llms.utils import test_complete_chat, test_text_completion_success, test_text_completion_failure
# Streaming LLM wrapper tests
from tests.e2e.llms.utils import test_streaming_complete_chat, test_streaming_complete_chat_sse, \
    test_streaming_text_completion_success, test_streaming_text_completion_failure, test_streaming_text_completion_sse
# Chatbot tests
from tests.e2e.llms.utils import test_chatbot_chat, test_chatbot_resend
# Streaming chatbot tests
from tests.e2e.llms.utils import test_streaming_chatbot_chat, test_streaming_chatbot_resend

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
cohere_api_key = os.getenv("COHERE_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
hugging_face_api_key = os.getenv("HUGGING_FACE_API_KEY")


class E2ETestHuggingFaceInferenceWrapper(TestCase):

    def test_complete_chat(self):
        fixture = HuggingFaceInferenceWrapper(
            hugging_face_api_key,
            model_url="https://api-inference.huggingface.co/models/bigscience/bloom"
        )
        test_complete_chat(self, fixture, verbose=False)

    def test_text_completion_success(self):
        fixture = HuggingFaceInferenceWrapper(
            hugging_face_api_key,
            model_url="https://api-inference.huggingface.co/models/bigscience/bloom"
        )
        test_text_completion_success(self, fixture, verbose=False)


class E2ETestBloomWrapper(TestCase):

    def test_complete_chat(self):
        fixture = BloomWrapper(hugging_face_api_key)
        test_complete_chat(self, fixture, verbose=False)

    def test_text_completion_success(self):
        fixture = BloomWrapper(hugging_face_api_key)
        test_text_completion_success(self, fixture, verbose=False)


class E2ETestOpenAIGPTWrapper(TestCase):

    def test_complete_chat(self):
        fixture = OpenAIGPTWrapper(openai_api_key, model="gpt-3.5-turbo")
        test_complete_chat(self, fixture, verbose=False)

    def test_text_completion_success(self):
        fixture = OpenAIGPTWrapper(openai_api_key, model="text-davinci-003")
        test_text_completion_success(self, fixture, verbose=False)

    def test_text_completion_failure(self):
        """
        Tests that the OpenAIGPTWrapper raises an exception when a chat model is used for text completion.
        """
        fixture = OpenAIGPTWrapper(openai_api_key, model="gpt-3.5-turbo")
        test_text_completion_failure(self, fixture, verbose=False)


class E2ETestClaudeWrapper(TestCase):

    def test_complete_chat(self):
        fixture = ClaudeWrapper(anthropic_api_key, model="claude-v1")
        test_complete_chat(self, fixture, verbose=False)

    def test_text_completion_success(self):
        fixture = ClaudeWrapper(anthropic_api_key, model="claude-v1")
        test_text_completion_success(self, fixture, verbose=False)


class E2ETestGPT2Wrapper(TestCase):

    def test_complete_chat(self):
        fixture = GPT2Wrapper()
        test_complete_chat(self, fixture, verbose=False)

    def test_text_completion_success(self):
        fixture = GPT2Wrapper()
        test_text_completion_success(self, fixture, verbose=False)


class E2ETestDollyWrapper(TestCase):

    def test_complete_chat(self):
        fixture = DollyWrapper()
        test_complete_chat(self, fixture, verbose=False)

    def test_text_completion_success(self):
        fixture = DollyWrapper()
        test_text_completion_success(self, fixture, verbose=False)


class E2ETestCohereWrapper(TestCase):

    def test_complete_chat(self):
        fixture = CohereWrapper(cohere_api_key, model="xlarge")
        test_complete_chat(self, fixture, verbose=False)

    def test_text_completion_success(self):
        fixture = CohereWrapper(cohere_api_key, model="xlarge")
        test_text_completion_success(self, fixture, verbose=False)


class E2ETestStreamingOpenAIGPTWrapper(TestCase):

    def test_complete_chat(self):
        """
        Tests that the StreamingOpenAIGPTWrapper can be used to perform chat completion.
        """
        fixture = StreamingOpenAIGPTWrapper(openai_api_key, model="gpt-3.5-turbo")

        test_streaming_complete_chat(self, fixture, verbose=False)

    def test_complete_chat_sse(self):
        """
        Tests that the StreamingOpenAIGPTWrapper can be used to perform streaming chat completion.
        """
        fixture = StreamingOpenAIGPTWrapper(
            openai_api_key, model="gpt-3.5-turbo", format_sse=True, append_stop_token=False
        )

        test_streaming_complete_chat_sse(self, fixture, check_stop=False, verbose=False)

    def test_complete_chat_sse_with_stop(self):
        """
        Tests that the StreamingOpenAIGPTWrapper can be used to perform streaming chat completion with a stop token.
        """
        fixture = StreamingOpenAIGPTWrapper(
            openai_api_key, model="gpt-3.5-turbo", format_sse=True, append_stop_token=True
        )

        test_streaming_complete_chat_sse(self, fixture, check_stop=True, verbose=False)

    def test_text_completion_success(self):
        """
        Tests that the StreamingOpenAIGPTWrapper can be used to perform text completion.
        """
        fixture = StreamingOpenAIGPTWrapper(openai_api_key, model="text-davinci-003")

        test_streaming_text_completion_success(self, fixture, verbose=False)

    def test_text_completion_failure(self):
        """
        Tests that the StreamingOpenAIGPTWrapper raises an exception when a chat model is used for text completion.
        """
        fixture = StreamingOpenAIGPTWrapper(openai_api_key, model="gpt-3.5-turbo")

        test_streaming_text_completion_failure(self, fixture, verbose=False)

    def test_text_completion_sse(self):
        """
        Tests that the StreamingOpenAIGPTWrapper can be used to perform streaming text completion.
        """
        fixture = StreamingOpenAIGPTWrapper(
            openai_api_key, model="text-davinci-003", format_sse=True, append_stop_token=False
        )

        test_streaming_text_completion_sse(self, fixture, check_stop=False, verbose=False)

    def test_text_completion_sse_with_stop(self):
        """
        Tests that the StreamingOpenAIGPTWrapper can be used to perform streaming text completion with a stop token.
        """
        fixture = StreamingOpenAIGPTWrapper(
            openai_api_key, model="text-davinci-003", format_sse=True, append_stop_token=True
        )

        test_streaming_text_completion_sse(self, fixture, check_stop=True, verbose=False)


class E2ETestStreamingClaudeWrapper(TestCase):

    def test_complete_chat(self):
        """
        Tests that the StreamingClaudeWrapper can be used to perform chat completion.
        """
        fixture = StreamingClaudeWrapper(anthropic_api_key, model="claude-v1")

        test_streaming_complete_chat(self, fixture, verbose=False)

    def test_complete_chat_sse(self):
        """
        Tests that the StreamingClaudeWrapper can be used to perform streaming chat completion.
        """
        fixture = StreamingClaudeWrapper(anthropic_api_key, model="claude-v1", format_sse=True, append_stop_token=False)

        test_streaming_complete_chat_sse(self, fixture, check_stop=False, verbose=False)

    def test_complete_chat_sse_with_stop(self):
        """
        Tests that the StreamingClaudeWrapper can be used to perform streaming chat completion with a stop token.
        """
        fixture = StreamingClaudeWrapper(anthropic_api_key, model="claude-v1", format_sse=True, append_stop_token=True)

        test_streaming_complete_chat_sse(self, fixture, check_stop=True, verbose=False)

    def test_text_completion(self):
        """
        Tests that the StreamingClaudeWrapper can be used to perform text completion.
        """
        fixture = StreamingClaudeWrapper(anthropic_api_key, model="claude-v1")

        test_streaming_text_completion_success(self, fixture, verbose=False)

    def test_text_completion_sse(self):
        """
        Tests that the StreamingClaudeWrapper can be used to perform streaming text completion.
        """
        fixture = StreamingClaudeWrapper(anthropic_api_key, model="claude-v1", format_sse=True, append_stop_token=False)

        test_streaming_text_completion_sse(self, fixture, check_stop=False, verbose=False)

    def test_text_completion_sse_with_stop(self):
        """
        Tests that the StreamingClaudeWrapper can be used to perform streaming text completion with a stop token.
        """
        fixture = StreamingClaudeWrapper(anthropic_api_key, model="claude-v1", format_sse=True, append_stop_token=True)

        test_streaming_text_completion_sse(self, fixture, check_stop=True, verbose=False)


class E2ETestChatBot(TestCase):

    def test_openai_gpt_chat(self):
        llm = OpenAIGPTWrapper(openai_api_key, model="gpt-3.5-turbo")
        fixture = ChatBot(llm)

        test_chatbot_chat(self, fixture)

    def test_openai_gpt_resend(self):
        llm = OpenAIGPTWrapper(openai_api_key, model="gpt-3.5-turbo")
        fixture = ChatBot(llm)

        test_chatbot_resend(self, fixture)

    def test_claude_chat(self):
        llm = ClaudeWrapper(anthropic_api_key, model="claude-v1")
        fixture = ChatBot(llm)

        test_chatbot_chat(self, fixture)

    def test_claude_resend(self):
        llm = ClaudeWrapper(anthropic_api_key, model="claude-v1")
        fixture = ChatBot(llm)

        test_chatbot_resend(self, fixture)

    def test_hugging_face_chat(self):
        llm = HuggingFaceInferenceWrapper(
            hugging_face_api_key,
            model_url="https://api-inference.huggingface.co/models/bigscience/bloom"
        )
        fixture = ChatBot(llm)

        test_chatbot_chat(self, fixture)

    def test_hugging_face_resend(self):
        llm = HuggingFaceInferenceWrapper(
            hugging_face_api_key,
            model_url="https://api-inference.huggingface.co/models/bigscience/bloom"
        )
        fixture = ChatBot(llm)

        test_chatbot_resend(self, fixture)

    def test_bloom_chat(self):
        llm = BloomWrapper(hugging_face_api_key)
        fixture = ChatBot(llm)

        test_chatbot_chat(self, fixture)

    def test_bloom_resend(self):
        llm = BloomWrapper(hugging_face_api_key)
        fixture = ChatBot(llm)

        test_chatbot_resend(self, fixture)

    def test_gpt2_chat(self):
        llm = GPT2Wrapper()
        fixture = ChatBot(llm)

        test_chatbot_chat(self, fixture)

    def test_gpt2_resend(self):
        llm = GPT2Wrapper()
        fixture = ChatBot(llm)

        test_chatbot_resend(self, fixture)

    def test_dolly_chat(self):
        llm = DollyWrapper()
        fixture = ChatBot(llm)

        test_chatbot_chat(self, fixture)

    def test_dolly_resend(self):
        llm = DollyWrapper()
        fixture = ChatBot(llm)

        test_chatbot_resend(self, fixture)

    def test_cohere_chat(self):
        llm = CohereWrapper(cohere_api_key, model="xlarge")
        fixture = ChatBot(llm)

        test_chatbot_chat(self, fixture)

    def test_cohere_resend(self):
        llm = CohereWrapper(cohere_api_key, model="xlarge")
        fixture = ChatBot(llm)

        test_chatbot_resend(self, fixture)


class E2ETestStreamingChatBot(TestCase):

    def test_openai_gpt_streaming_chat(self):
        llm = StreamingOpenAIGPTWrapper(openai_api_key, model="gpt-3.5-turbo")
        fixture = ChatBot(llm)

        test_streaming_chatbot_chat(self, fixture=fixture, chunk_time_seconds_threshold=0.2, verbose=False)

    def test_openai_gpt_streaming_resend(self):
        llm = StreamingOpenAIGPTWrapper(openai_api_key, model="gpt-3.5-turbo")
        fixture = ChatBot(llm)

        test_streaming_chatbot_resend(self, fixture=fixture, verbose=False)

    def test_claude_streaming_chat(self):
        llm = StreamingClaudeWrapper(anthropic_api_key, model="claude-v1")
        fixture = ChatBot(llm)

        test_streaming_chatbot_chat(self, fixture=fixture, chunk_time_seconds_threshold=0.2, verbose=False)

    def test_claude_streaming_resend(self):
        llm = StreamingClaudeWrapper(anthropic_api_key, model="claude-v1")
        fixture = ChatBot(llm)

        test_streaming_chatbot_resend(self, fixture=fixture, verbose=False)
