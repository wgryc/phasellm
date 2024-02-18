import os
import gc

from unittest import TestCase

from dotenv import load_dotenv

from phasellm.configurations import AzureAPIConfiguration

from phasellm.llms import \
    HuggingFaceInferenceWrapper, \
    BloomWrapper, \
    OpenAIGPTWrapper, StreamingOpenAIGPTWrapper, \
    ClaudeWrapper, StreamingClaudeWrapper, \
    VertexAIWrapper, StreamingVertexAIWrapper, \
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
from tests.e2e.llms.utils import test_streaming_chatbot_chat, test_streaming_chatbot_chat_sse, \
    test_streaming_chatbot_resend

load_dotenv()
azure_api_key = os.getenv("AZURE_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
cohere_api_key = os.getenv("COHERE_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
hugging_face_api_key = os.getenv("HUGGING_FACE_API_KEY")

# Enable skipping local testing local models (most machines can't run them)
skip_local_models = os.getenv("SKIP_LOCAL_MODELS")
skip_local_models = skip_local_models is not None and skip_local_models.lower() == "true"


class E2ETestHuggingFaceInferenceWrapper(TestCase):

    def setUp(self) -> None:
        gc.collect()

    def test_complete_chat(self):
        fixture = HuggingFaceInferenceWrapper(
            hugging_face_api_key,
            model_url="https://api-inference.huggingface.co/models/bigscience/bloom"
        )
        test_complete_chat(self, fixture, check_last_response_header=True, verbose=False)

    def test_complete_chat_kwargs(self):
        fixture = HuggingFaceInferenceWrapper(
            hugging_face_api_key,
            model_url="https://api-inference.huggingface.co/models/bigscience/bloom",
            temperature=0.9,
            top_k=2
        )
        test_complete_chat(self, fixture, check_last_response_header=True, verbose=False)

    def test_text_completion_success(self):
        fixture = HuggingFaceInferenceWrapper(
            hugging_face_api_key,
            model_url="https://api-inference.huggingface.co/models/bigscience/bloom"
        )
        test_text_completion_success(self, fixture, check_last_response_header=True, verbose=False)

    def test_text_completion_success_kwargs(self):
        fixture = HuggingFaceInferenceWrapper(
            hugging_face_api_key,
            temperature=0.9,
            top_k=0.9,
            model_url="https://api-inference.huggingface.co/models/bigscience/bloom"
        )
        test_text_completion_success(self, fixture, check_last_response_header=True, verbose=False)


class E2ETestBloomWrapper(TestCase):
    # TODO remove this if we decide to remove the BloomWrapper in favor of only having the HuggingFaceInferenceWrapper

    def setUp(self) -> None:
        gc.collect()

    def test_complete_chat(self):
        fixture = BloomWrapper(hugging_face_api_key)
        test_complete_chat(self, fixture, check_last_response_header=True, verbose=False)

    def test_text_completion_success(self):
        fixture = BloomWrapper(hugging_face_api_key)
        test_text_completion_success(self, fixture, check_last_response_header=True, verbose=False)


class E2ETestOpenAIGPTWrapper(TestCase):

    def setUp(self) -> None:
        gc.collect()

    def test_complete_chat(self):
        fixture = OpenAIGPTWrapper(openai_api_key, model="gpt-3.5-turbo")
        test_complete_chat(self, fixture, check_last_response_header=True, verbose=False)

    def test_complete_chat_azure(self):
        fixture = OpenAIGPTWrapper(api_config=AzureAPIConfiguration(
            api_key=azure_api_key,
            base_url=f'https://val-gpt4.openai.azure.com/openai/deployments/gpt-4',
            api_version='2023-05-15',
            deployment_id='gpt-4'
        ))
        test_complete_chat(self, fixture, check_last_response_header=True, verbose=False)

    def test_complete_chat_azure_pre_openai_v1(self):
        fixture = OpenAIGPTWrapper(api_config=AzureAPIConfiguration(
            api_key=azure_api_key,
            base_url=f'https://val-gpt4.openai.azure.com/gpt-4',
            api_version='2023-05-15',
            deployment_id='gpt-4'
        ))
        test_complete_chat(self, fixture, check_last_response_header=True, verbose=False)

    def test_complete_chat_kwargs(self):
        fixture = OpenAIGPTWrapper(openai_api_key, model="gpt-3.5-turbo", temperature=0.9, frequency_penalty=2)
        test_complete_chat(self, fixture, check_last_response_header=True, verbose=False)

    def test_text_completion_success(self):
        fixture = OpenAIGPTWrapper(openai_api_key, model="text-davinci-003")
        test_text_completion_success(self, fixture, check_last_response_header=True, verbose=False)

    def test_text_completion_success_kwargs(self):
        fixture = OpenAIGPTWrapper(openai_api_key, model="text-davinci-003", temperature=0.9, presence_penalty=2)
        test_text_completion_success(self, fixture, check_last_response_header=True, verbose=False)

    def test_text_completion_failure(self):
        """
        Tests that the OpenAIGPTWrapper raises an exception when a chat model is used for text completion.
        """
        fixture = OpenAIGPTWrapper(openai_api_key, model="gpt-3.5-turbo")
        test_text_completion_failure(self, fixture, verbose=False)


class E2ETestVertexAIWrapper(TestCase):

    def setUp(self) -> None:
        gc.collect()

    def test_complete_chat_text(self):
        fixture = VertexAIWrapper(model="text-bison@002")
        test_complete_chat(self, fixture, check_last_response_header=True, verbose=False)

    def test_complete_chat_chat(self):
        fixture = VertexAIWrapper(model="chat-bison@002")
        test_complete_chat(self, fixture, check_last_response_header=True, verbose=False)

    def test_complete_chat_generative(self):
        fixture = VertexAIWrapper(model="gemini-1.0-pro")
        test_complete_chat(self, fixture, check_last_response_header=True, verbose=False)

    def test_complete_chat_kwargs(self):
        fixture = VertexAIWrapper(temperature=0.9, top_k=2)
        test_complete_chat(self, fixture, check_last_response_header=True, verbose=False)

    def test_text_completion_success(self):
        fixture = VertexAIWrapper()
        test_text_completion_success(self, fixture, check_last_response_header=True, verbose=False)

    def test_text_completion_success_kwargs(self):
        fixture = VertexAIWrapper(temperature=0.9, top_k=2)
        test_text_completion_success(self, fixture, check_last_response_header=True, verbose=False)


class E2ETestClaudeWrapper(TestCase):

    def setUp(self) -> None:
        gc.collect()

    def test_complete_chat(self):
        fixture = ClaudeWrapper(anthropic_api_key, model="claude-v1")
        test_complete_chat(self, fixture, check_last_response_header=True, verbose=False)

    def test_complete_chat_kwargs(self):
        fixture = ClaudeWrapper(anthropic_api_key, model="claude-v1", temperature=0.9, top_k=2)
        test_complete_chat(self, fixture, check_last_response_header=True, verbose=False)

    def test_text_completion_success(self):
        fixture = ClaudeWrapper(anthropic_api_key, model="claude-v1")
        test_text_completion_success(self, fixture, check_last_response_header=True, verbose=False)

    def test_text_completion_success_kwargs(self):
        fixture = ClaudeWrapper(anthropic_api_key, model="claude-v1", temperature=0.9, top_k=2)
        test_text_completion_success(self, fixture, check_last_response_header=True, verbose=False)


class E2ETestGPT2Wrapper(TestCase):

    def setUp(self) -> None:
        if skip_local_models:
            print("Skipping test_complete_chat for DollyWrapper as SKIP_LOCAL_MODELS is set to True")
            return

        # Dynamically import torch to avoid import errors for users who don't have phasellm[complete].
        import torch
        torch.cuda.empty_cache()

        gc.collect()

    def test_complete_chat(self):
        if skip_local_models:
            print("Skipping test_complete_chat for GPT2Wrapper as SKIP_LOCAL_MODELS is set to True")
            return
        fixture = GPT2Wrapper()
        test_complete_chat(self, fixture, verbose=False)

    def test_complete_chat_kwargs(self):
        if skip_local_models:
            print("Skipping test_complete_chat for GPT2Wrapper as SKIP_LOCAL_MODELS is set to True")
            return
        fixture = GPT2Wrapper(temperature=0.9, top_k=2)
        test_complete_chat(self, fixture, verbose=False)

    def test_text_completion_success(self):
        if skip_local_models:
            print("Skipping test_text_completion_success for GPT2Wrapper as SKIP_LOCAL_MODELS is set to True")
            return
        fixture = GPT2Wrapper()
        test_text_completion_success(self, fixture, verbose=False)

    def test_text_completion_success_kwargs(self):
        if skip_local_models:
            print("Skipping test_text_completion_success for GPT2Wrapper as SKIP_LOCAL_MODELS is set to True")
            return
        fixture = GPT2Wrapper(temperature=0.9, top_k=2)
        test_text_completion_success(self, fixture, verbose=False)


class E2ETestDollyWrapper(TestCase):
    def setUp(self) -> None:
        if skip_local_models:
            print("Skipping test_complete_chat for DollyWrapper as SKIP_LOCAL_MODELS is set to True")
            return

        # Dynamically import torch to avoid import errors for users who don't have phasellm[complete].
        import torch
        torch.cuda.empty_cache()

        gc.collect()

    def test_complete_chat(self):
        if skip_local_models:
            print("Skipping test_complete_chat for DollyWrapper as SKIP_LOCAL_MODELS is set to True")
            return
        fixture = DollyWrapper()
        test_complete_chat(self, fixture, verbose=False)

    def test_complete_chat_kwargs(self):
        if skip_local_models:
            print("Skipping test_complete_chat for DollyWrapper as SKIP_LOCAL_MODELS is set to True")
            return
        fixture = DollyWrapper(temperature=0.9, top_k=2)
        test_complete_chat(self, fixture, verbose=False)

    def test_text_completion_success(self):
        if skip_local_models:
            print("Skipping test_text_completion_success for DollyWrapper as SKIP_LOCAL_MODELS is set to True")
            return
        fixture = DollyWrapper()
        test_text_completion_success(self, fixture, verbose=False)

    def test_text_completion_success_kwargs(self):
        if skip_local_models:
            print("Skipping test_text_completion_success for DollyWrapper as SKIP_LOCAL_MODELS is set to True")
            return
        fixture = DollyWrapper(temperature=0.9, top_k=2)
        test_text_completion_success(self, fixture, verbose=False)


class E2ETestCohereWrapper(TestCase):

    def setUp(self) -> None:
        gc.collect()

    def test_complete_chat(self):
        fixture = CohereWrapper(cohere_api_key, model="xlarge")
        test_complete_chat(self, fixture, verbose=False)

    def test_complete_chat_kwargs(self):
        # Note that the Cohere Client doesn't support top_k.
        fixture = CohereWrapper(cohere_api_key, model="xlarge", temperature=0.9)
        test_complete_chat(self, fixture, verbose=False)

    def test_text_completion_success(self):
        fixture = CohereWrapper(cohere_api_key, model="xlarge")
        test_text_completion_success(self, fixture, verbose=False)

    def test_text_completion_success_kwargs(self):
        fixture = CohereWrapper(cohere_api_key, model="xlarge", temperature=0.9)
        test_text_completion_success(self, fixture, verbose=False)


class E2ETestStreamingOpenAIGPTWrapper(TestCase):

    def setUp(self) -> None:
        gc.collect()

    def test_complete_chat(self):
        """
        Tests that the StreamingOpenAIGPTWrapper can be used to perform chat completion.
        """
        fixture = StreamingOpenAIGPTWrapper(openai_api_key, model="gpt-3.5-turbo")

        test_streaming_complete_chat(self, fixture, check_last_response_header=True, verbose=False)

    def test_complete_chat_azure(self):
        """
        Tests that the StreamingOpenAIGPTWrapper with azure configuration can be used to perform chat completion.
        """
        fixture = StreamingOpenAIGPTWrapper(api_config=AzureAPIConfiguration(
            api_key=azure_api_key,
            base_url=f'https://val-gpt4.openai.azure.com/openai/deployments/gpt-4',
            api_version='2023-05-15',
            deployment_id='gpt-4'
        ))

        test_streaming_complete_chat(self, fixture, check_last_response_header=True, verbose=False)

    def test_complete_chat_azure_pre_openai_v1(self):
        """
        Tests that the StreamingOpenAIGPTWrapper with azure configuration can be used to perform chat completion.
        """
        fixture = StreamingOpenAIGPTWrapper(api_config=AzureAPIConfiguration(
            api_key=azure_api_key,
            base_url=f'https://val-gpt4.openai.azure.com/gpt-4',
            api_version='2023-05-15',
            deployment_id='gpt-4'
        ))

        test_streaming_complete_chat(self, fixture, check_last_response_header=True, verbose=False)

    def test_complete_chat_kwargs(self):
        """
        Tests that the StreamingOpenAIGPTWrapper can be used to perform chat completion with kwargs.
        """
        fixture = StreamingOpenAIGPTWrapper(
            openai_api_key, model="gpt-3.5-turbo", temperature=0.9, presence_penalty=0.9, frequency_penalty=0.9
        )

        test_streaming_complete_chat(self, fixture, check_last_response_header=True, verbose=False)

    def test_complete_chat_sse(self):
        """
        Tests that the StreamingOpenAIGPTWrapper can be used to perform streaming chat completion.
        """
        fixture = StreamingOpenAIGPTWrapper(
            openai_api_key, model="gpt-3.5-turbo", format_sse=True, append_stop_token=False
        )

        test_streaming_complete_chat_sse(self, fixture, check_stop=False, verbose=False)

    def test_complete_chat_sse_kwargs(self):
        """
        Tests that the StreamingOpenAIGPTWrapper can be used to perform streaming chat completion with kwargs.
        """
        fixture = StreamingOpenAIGPTWrapper(
            openai_api_key, model="gpt-3.5-turbo", format_sse=True, append_stop_token=False, temperature=0.9,
            presence_penalty=0.9
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

    def test_complete_chat_sse_with_stop_kwargs(self):
        """
        Tests that the StreamingOpenAIGPTWrapper can be used to perform streaming chat completion with a stop token and
        kwargs.
        """
        fixture = StreamingOpenAIGPTWrapper(
            openai_api_key, model="gpt-3.5-turbo", format_sse=True, append_stop_token=True, temperature=0.9,
            presence_penalty=2
        )

        test_streaming_complete_chat_sse(self, fixture, check_stop=True, verbose=False)

    def test_text_completion_success(self):
        """
        Tests that the StreamingOpenAIGPTWrapper can be used to perform text completion.
        """
        fixture = StreamingOpenAIGPTWrapper(openai_api_key, model="text-davinci-003")

        test_streaming_text_completion_success(self, fixture, check_last_response_header=True, verbose=False)

    def test_text_completion_success_kwargs(self):
        """
        Tests that the StreamingOpenAIGPTWrapper can be used to perform text completion with kwargs.
        """
        fixture = StreamingOpenAIGPTWrapper(openai_api_key, model="text-davinci-003", temperature=0.9,
                                            presence_penalty=2)

        test_streaming_text_completion_success(self, fixture, check_last_response_header=True, verbose=False)

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

    def test_text_completion_sse_kwargs(self):
        """
        Tests that the StreamingOpenAIGPTWrapper can be used to perform streaming text completion with kwargs.
        """
        fixture = StreamingOpenAIGPTWrapper(
            openai_api_key, model="text-davinci-003", format_sse=True, append_stop_token=False, temperature=0.9,
            presence_penalty=2
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

    def test_text_completion_sse_with_stop_kwargs(self):
        """
        Tests that the StreamingOpenAIGPTWrapper can be used to perform streaming text completion with a stop token and
        kwargs.
        """
        fixture = StreamingOpenAIGPTWrapper(
            openai_api_key, model="text-davinci-003", format_sse=True, append_stop_token=True, temperature=0.9,
            presence_penalty=2
        )

        test_streaming_text_completion_sse(self, fixture, check_stop=True, verbose=False)


class E2ETestStreamingVertexAIWrapper(TestCase):

    def setUp(self) -> None:
        gc.collect()

    def test_complete_chat_chat(self):
        """
        Tests that the StreamingVertexAIWrapper can be used to perform chat completion.
        """
        fixture = StreamingVertexAIWrapper(model="chat-bison@002")

        test_streaming_complete_chat(self, fixture, check_last_response_header=False, verbose=False)

    def test_complete_chat_text(self):
        """
        Tests that the StreamingVertexAIWrapper can be used to perform chat completion.
        """
        fixture = StreamingVertexAIWrapper(model="text-bison@002")

        test_streaming_complete_chat(self, fixture, check_last_response_header=False, verbose=False)

    def test_complete_chat_generative(self):
        """
        Tests that the StreamingVertexAIWrapper can be used to perform chat completion.
        """
        fixture = StreamingVertexAIWrapper(model="gemini-1.0-pro")

        test_streaming_complete_chat(
            self, fixture, check_last_response_header=True, chunk_time_seconds_threshold=1, verbose=False
        )

    def test_complete_chat_kwargs(self):
        """
        Tests that the StreamingVertexAIWrapper can be used to perform chat completion with kwargs.
        """
        fixture = StreamingVertexAIWrapper(temperature=0.9, top_k=2)

        test_streaming_complete_chat(
            self, fixture, check_last_response_header=True, chunk_time_seconds_threshold=1, verbose=False
        )

    def test_complete_chat_sse(self):
        """
        Tests that the StreamingVertexAIWrapper can be used to perform streaming chat completion.
        """
        fixture = StreamingVertexAIWrapper(format_sse=True, append_stop_token=False)

        test_streaming_complete_chat_sse(self, fixture, check_stop=False, verbose=False)

    def test_complete_chat_sse_kwargs(self):
        """
        Tests that the StreamingVertexAIWrapper can be used to perform streaming chat completion with kwargs.
        """
        fixture = StreamingVertexAIWrapper(format_sse=True, append_stop_token=False, temperature=0.9, top_k=2)

        test_streaming_complete_chat_sse(self, fixture, check_stop=False, verbose=False)

    def test_complete_chat_sse_with_stop(self):
        """
        Tests that the StreamingVertexAIWrapper can be used to perform streaming chat completion with a stop token.
        """
        fixture = StreamingVertexAIWrapper(format_sse=True, append_stop_token=True)

        test_streaming_complete_chat_sse(self, fixture, check_stop=True, verbose=False)

    def test_complete_chat_sse_with_stop_kwargs(self):
        """
        Tests that the StreamingVertexAIWrapper can be used to perform streaming chat completion with a stop token and
        kwargs.
        """
        fixture = StreamingVertexAIWrapper(format_sse=True, append_stop_token=True, temperature=0.9, top_k=2)

        test_streaming_complete_chat_sse(self, fixture, check_stop=True, verbose=False)

    def test_text_completion(self):
        """
        Tests that the StreamingVertexAIWrapper can be used to perform text completion.
        """
        fixture = StreamingVertexAIWrapper()

        test_streaming_text_completion_success(
            self, fixture, check_last_response_header=True, verbose=False, prompt="Write 100 words of your choice."
        )

    def test_text_completion_kwargs(self):
        """
        Tests that the StreamingVertexAIWrapper can be used to perform text completion with kwargs.
        """
        fixture = StreamingVertexAIWrapper(temperature=0.9, top_k=2)

        test_streaming_text_completion_success(
            self, fixture, check_last_response_header=True, verbose=False, prompt="Write 100 words of your choice."
        )

    def test_text_completion_sse(self):
        """
        Tests that the StreamingVertexAIWrapper can be used to perform streaming text completion.
        """
        fixture = StreamingVertexAIWrapper(format_sse=True, append_stop_token=False)

        test_streaming_text_completion_sse(
            self, fixture, check_stop=False, verbose=False, prompt="Write 100 words of your choice."
        )

    def test_text_completion_sse_kwargs(self):
        """
        Tests that the StreamingVertexAIWrapper can be used to perform streaming text completion with kwargs.
        """
        fixture = StreamingVertexAIWrapper(format_sse=True, append_stop_token=False, temperature=0.9, top_k=2)

        test_streaming_text_completion_sse(
            self, fixture, check_stop=False, verbose=False, prompt="Write 100 words of your choice."
        )

    def test_text_completion_sse_with_stop(self):
        """
        Tests that the StreamingVertexAIWrapper can be used to perform streaming text completion with a stop token.
        """
        fixture = StreamingVertexAIWrapper(format_sse=True, append_stop_token=True)

        test_streaming_text_completion_sse(self, fixture, check_stop=True, verbose=False)

    def test_text_completion_sse_with_stop_kwargs(self):
        """
        Tests that the StreamingVertexAIWrapper can be used to perform streaming text completion with a stop token and
        kwargs.
        """
        fixture = StreamingVertexAIWrapper(format_sse=True, append_stop_token=True, temperature=0.9, top_k=2)

        test_streaming_text_completion_sse(self, fixture, check_stop=True, verbose=False)


class E2ETestStreamingClaudeWrapper(TestCase):

    def setUp(self) -> None:
        gc.collect()

    def test_complete_chat(self):
        """
        Tests that the StreamingClaudeWrapper can be used to perform chat completion.
        """
        fixture = StreamingClaudeWrapper(anthropic_api_key, model="claude-v1")

        test_streaming_complete_chat(self, fixture, check_last_response_header=True, verbose=False)

    def test_complete_chat_kwargs(self):
        """
        Tests that the StreamingClaudeWrapper can be used to perform chat completion with kwargs.
        """
        fixture = StreamingClaudeWrapper(anthropic_api_key, model="claude-v1", temperature=0.9, top_k=2)

        test_streaming_complete_chat(self, fixture, check_last_response_header=True, verbose=False)

    def test_complete_chat_sse(self):
        """
        Tests that the StreamingClaudeWrapper can be used to perform streaming chat completion.
        """
        fixture = StreamingClaudeWrapper(anthropic_api_key, model="claude-v1", format_sse=True, append_stop_token=False)

        test_streaming_complete_chat_sse(self, fixture, check_stop=False, verbose=False)

    def test_complete_chat_sse_kwargs(self):
        """
        Tests that the StreamingClaudeWrapper can be used to perform streaming chat completion with kwargs.
        """
        fixture = StreamingClaudeWrapper(anthropic_api_key, model="claude-v1", format_sse=True, append_stop_token=False,
                                         temperature=0.9, top_k=2)

        test_streaming_complete_chat_sse(self, fixture, check_stop=False, verbose=False)

    def test_complete_chat_sse_with_stop(self):
        """
        Tests that the StreamingClaudeWrapper can be used to perform streaming chat completion with a stop token.
        """
        fixture = StreamingClaudeWrapper(anthropic_api_key, model="claude-v1", format_sse=True, append_stop_token=True)

        test_streaming_complete_chat_sse(self, fixture, check_stop=True, verbose=False)

    def test_complete_chat_sse_with_stop_kwargs(self):
        """
        Tests that the StreamingClaudeWrapper can be used to perform streaming chat completion with a stop token and
        kwargs.
        """
        fixture = StreamingClaudeWrapper(anthropic_api_key, model="claude-v1", format_sse=True, append_stop_token=True,
                                         temperature=0.9, top_k=2)

        test_streaming_complete_chat_sse(self, fixture, check_stop=True, verbose=False)

    def test_text_completion(self):
        """
        Tests that the StreamingClaudeWrapper can be used to perform text completion.
        """
        fixture = StreamingClaudeWrapper(anthropic_api_key, model="claude-v1")

        test_streaming_text_completion_success(self, fixture, check_last_response_header=True, verbose=False)

    def test_text_completion_kwargs(self):
        """
        Tests that the StreamingClaudeWrapper can be used to perform text completion with kwargs.
        """
        fixture = StreamingClaudeWrapper(anthropic_api_key, model="claude-v1", temperature=0.9, top_k=2)

        test_streaming_text_completion_success(self, fixture, check_last_response_header=True, verbose=False)

    def test_text_completion_sse(self):
        """
        Tests that the StreamingClaudeWrapper can be used to perform streaming text completion.
        """
        fixture = StreamingClaudeWrapper(anthropic_api_key, model="claude-v1", format_sse=True, append_stop_token=False)

        test_streaming_text_completion_sse(self, fixture, check_stop=False, verbose=False)

    def test_text_completion_sse_kwargs(self):
        """
        Tests that the StreamingClaudeWrapper can be used to perform streaming text completion with kwargs.
        """
        fixture = StreamingClaudeWrapper(anthropic_api_key, model="claude-v1", format_sse=True, append_stop_token=False,
                                         temperature=0.9, top_k=2)

        test_streaming_text_completion_sse(self, fixture, check_stop=False, verbose=False)

    def test_text_completion_sse_with_stop(self):
        """
        Tests that the StreamingClaudeWrapper can be used to perform streaming text completion with a stop token.
        """
        fixture = StreamingClaudeWrapper(anthropic_api_key, model="claude-v1", format_sse=True, append_stop_token=True)

        test_streaming_text_completion_sse(self, fixture, check_stop=True, verbose=False)

    def test_text_completion_sse_with_stop_kwargs(self):
        """
        Tests that the StreamingClaudeWrapper can be used to perform streaming text completion with a stop token and
        kwargs.
        """
        fixture = StreamingClaudeWrapper(anthropic_api_key, model="claude-v1", format_sse=True, append_stop_token=True,
                                         temperature=0.9, top_k=2)

        test_streaming_text_completion_sse(self, fixture, check_stop=True, verbose=False)


class E2ETestChatBot(TestCase):

    def setUp(self) -> None:
        gc.collect()

    def test_openai_gpt_chat(self):
        llm = OpenAIGPTWrapper(openai_api_key, model="gpt-3.5-turbo")
        fixture = ChatBot(llm)

        test_chatbot_chat(self, fixture)

    def test_openai_gpt_chat_azure(self):
        llm = OpenAIGPTWrapper(api_config=AzureAPIConfiguration(
            api_key=azure_api_key,
            base_url=f'https://val-gpt4.openai.azure.com/openai/deployments/gpt-4',
            api_version='2023-05-15',
            deployment_id='gpt-4'
        ))
        fixture = ChatBot(llm)

        test_chatbot_chat(self, fixture)

    def test_openai_gpt_chat_azure_pre_openai_v1(self):
        llm = OpenAIGPTWrapper(api_config=AzureAPIConfiguration(
            api_key=azure_api_key,
            base_url=f'https://val-gpt4.openai.azure.com/gpt-4',
            api_version='2023-05-15',
            deployment_id='gpt-4'
        ))
        fixture = ChatBot(llm)

        test_chatbot_chat(self, fixture)

    def test_openai_gpt_resend(self):
        llm = OpenAIGPTWrapper(openai_api_key, model="gpt-3.5-turbo")
        fixture = ChatBot(llm)

        test_chatbot_resend(self, fixture)

    def test_claude_chat_2023_01_01(self):
        llm = ClaudeWrapper(anthropic_api_key, model="claude-v1", anthropic_version="2023-01-01")
        fixture = ChatBot(llm)

        test_chatbot_chat(self, fixture)

    def test_claude_chat_2023_06_01(self):
        llm = ClaudeWrapper(anthropic_api_key, model="claude-v1", anthropic_version="2023-06-01")
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
        if skip_local_models:
            print("Skipping test_gpt2_chat as SKIP_LOCAL_MODELS is set to True")
            return

        # Dynamically import torch to avoid import errors for users who don't have phasellm[complete].
        import torch
        torch.cuda.empty_cache()

        llm = GPT2Wrapper()
        fixture = ChatBot(llm)

        test_chatbot_chat(self, fixture)

    def test_gpt2_resend(self):
        if skip_local_models:
            print("Skipping test_gpt2_resend as SKIP_LOCAL_MODELS is set to True")
            return

        # Dynamically import torch to avoid import errors for users who don't have phasellm[complete].
        import torch
        torch.cuda.empty_cache()

        llm = GPT2Wrapper()
        fixture = ChatBot(llm)

        test_chatbot_resend(self, fixture)

    def test_dolly_chat(self):
        if skip_local_models:
            print("Skipping test_dolly_chat as SKIP_LOCAL_MODELS is set to True")
            return

        # Dynamically import torch to avoid import errors for users who don't have phasellm[complete].
        import torch
        torch.cuda.empty_cache()

        llm = DollyWrapper()
        fixture = ChatBot(llm)

        test_chatbot_chat(self, fixture)

    def test_dolly_resend(self):
        if skip_local_models:
            print("Skipping test_dolly_resend as SKIP_LOCAL_MODELS is set to True")
            return

        # Dynamically import torch to avoid import errors for users who don't have phasellm[complete].
        import torch
        torch.cuda.empty_cache()

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

    def setUp(self) -> None:
        gc.collect()

    def test_openai_gpt_streaming_chat(self):
        llm = StreamingOpenAIGPTWrapper(openai_api_key, model="gpt-3.5-turbo")
        fixture = ChatBot(llm)

        test_streaming_chatbot_chat(self, fixture=fixture, chunk_time_seconds_threshold=0.5, verbose=False)

    def test_openai_gpt_streaming_chat_azure(self):
        llm = StreamingOpenAIGPTWrapper(api_config=AzureAPIConfiguration(
            api_key=azure_api_key,
            base_url=f'https://val-gpt4.openai.azure.com/openai/deployments/gpt-4',
            api_version='2023-05-15',
            deployment_id='gpt-4'
        ))
        fixture = ChatBot(llm)

        test_streaming_chatbot_chat(self, fixture=fixture, chunk_time_seconds_threshold=0.5, verbose=False)

    def test_openai_gpt_streaming_chat_azure_pre_openai_v1(self):
        llm = StreamingOpenAIGPTWrapper(api_config=AzureAPIConfiguration(
            api_key=azure_api_key,
            base_url=f'https://val-gpt4.openai.azure.com/gpt-4',
            api_version='2023-05-15',
            deployment_id='gpt-4'
        ))
        fixture = ChatBot(llm)

        test_streaming_chatbot_chat(self, fixture=fixture, chunk_time_seconds_threshold=0.5, verbose=False)

    def test_openai_gpt_streaming_chat_sse(self):
        llm = StreamingOpenAIGPTWrapper(openai_api_key, model="gpt-3.5-turbo", format_sse=True, append_stop_token=True)
        fixture = ChatBot(llm)

        test_streaming_chatbot_chat_sse(self, fixture=fixture, verbose=False)

    def test_openai_gpt_streaming_resend(self):
        llm = StreamingOpenAIGPTWrapper(openai_api_key, model="gpt-3.5-turbo")
        fixture = ChatBot(llm)

        test_streaming_chatbot_resend(self, fixture=fixture, verbose=False)

    # TODO Consider adding test_openai_gpt_streaming_resend_sse()

    def test_claude_streaming_chat_2023_01_01(self):
        llm = StreamingClaudeWrapper(anthropic_api_key, model="claude-v1", anthropic_version="2023-01-01")
        fixture = ChatBot(llm)

        test_streaming_chatbot_chat(self, fixture=fixture, chunk_time_seconds_threshold=0.5, verbose=False)

    def test_claude_streaming_chat_2023_06_01(self):
        llm = StreamingClaudeWrapper(anthropic_api_key, model="claude-v1", anthropic_version="2023-06-01")
        fixture = ChatBot(llm)

        test_streaming_chatbot_chat(self, fixture=fixture, chunk_time_seconds_threshold=0.5, verbose=False)

    # TODO Consider adding test_claude_streaming_chat_sse()

    def test_claude_streaming_resend(self):
        llm = StreamingClaudeWrapper(anthropic_api_key, model="claude-v1")
        fixture = ChatBot(llm)

        test_streaming_chatbot_resend(self, fixture=fixture, verbose=False)

    # TODO Consider adding test_claude_streaming_resend_sse()
