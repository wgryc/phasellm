import time
import copy

from typing import Generator

from unittest import TestCase

from phasellm.llms import LanguageModelWrapper, StreamingLanguageModelWrapper, ChatBot

from dataclasses import dataclass


def common_chat_assertions(tester: TestCase, response: str, verbose: bool = False) -> None:
    """
    Helper function for common response completion assertions.
    """
    tester.assertTrue(
        isinstance(response, str),
        f"Expected a string. Got: {type(response)}"
    )
    tester.assertTrue(
        len(response) > 0,
        f"Chat is empty."
    )

    if verbose:
        print(f"Chat: {response}")


def common_text_assertions(tester: TestCase, response: str, verbose: bool = False) -> None:
    """
    Helper function for common text completion assertions.
    """
    tester.assertTrue(
        isinstance(response, str),
        f"Expected a string. Got: {type(response)}"
    )
    tester.assertTrue(
        len(response) > 0,
        f"Chat is empty."
    )

    if verbose:
        print(f"Chat: {response}")


@dataclass
class StreamingChatCompletionProbe:
    res: str
    chunk_times: list
    mean_chunk_time: float


def probe_streaming_chat_completion(generator: Generator) -> StreamingChatCompletionProbe:
    """
    Helper function for testing streaming chat completion.
    """
    res = ''
    chunk_times = []
    time_start = time.time()
    for chunk in generator:
        # Track the time it takes to generate each chunk.
        time_end = time.time()
        chunk_times.append(time_end - time_start)
        time_start = time_end

        # Print the response length so far.
        res += chunk

    # Compute the mean chunk time
    mean_chunk_time = sum(chunk_times) / len(chunk_times)

    return StreamingChatCompletionProbe(res=res, chunk_times=chunk_times, mean_chunk_time=mean_chunk_time)


def common_streaming_chat_assertions(
        tester: TestCase,
        probe: StreamingChatCompletionProbe,
        chunk_time_seconds_threshold: float,
        verbose: bool = False
) -> None:
    """
    Helper function for common streaming chat completion assertions.
    """
    tester.assertTrue(
        len(probe.res) > 0,
        "Expecting a non-empty response."
    )
    tester.assertTrue(
        len(probe.chunk_times) > 1,
        f"Expecting more than one chunk, got {len(probe.chunk_times)}"
    )
    tester.assertTrue(
        probe.mean_chunk_time < chunk_time_seconds_threshold,
        f"Expecting a mean chunk time of less than {chunk_time_seconds_threshold} seconds."
    )
    if verbose:
        print(f'Chunk count: {len(probe.chunk_times)}')
        print(f'Mean chunk time: {probe.mean_chunk_time} seconds.')
        print(f'Result:\n{probe.res}')


@dataclass
class StreamingTextCompletionProbe:
    res: str
    chunk_count: int


def probe_streaming_text_completion(generator: Generator) -> StreamingTextCompletionProbe:
    """
    Helper function for testing streaming text completion.
    """
    res = ''
    chunk_count = 0
    for chunk in generator:
        chunk_count += 1
        res += chunk

    return StreamingTextCompletionProbe(res=res, chunk_count=chunk_count)


def common_streaming_text_assertions(
        tester: TestCase,
        probe: StreamingTextCompletionProbe,
        verbose: bool = False
) -> None:
    """
    Helper function for common streaming text completion assertions.
    """
    tester.assertTrue(
        len(probe.res) > 0,
        "Expecting a non-empty response."
    )
    tester.assertTrue(
        probe.chunk_count > 1,
        f"Expecting more than one chunk, got {probe.chunk_count}"
    )
    if verbose:
        print(f'Chunk count: {probe.chunk_count}')
        print(f'Result:\n{probe.res}')


@dataclass
class StreamingSSECompletionProbe:
    res: str
    chunk_count: int
    chunks_with_protocol: int


def probe_streaming_sse_completions(generator: Generator) -> StreamingSSECompletionProbe:
    """
    Helper function for testing streaming chat completion.
    """
    res = ''
    chunk_count = 0
    chunks_with_protocol = 0
    for chunk in generator:
        chunk_count += 1
        if chunk.startswith('data:') and chunk.endswith('\n\n'):
            chunks_with_protocol += 1
        res += chunk

    return StreamingSSECompletionProbe(res=res, chunk_count=chunk_count, chunks_with_protocol=chunks_with_protocol)


def common_streaming_sse_assertions(
        tester: TestCase,
        probe: StreamingSSECompletionProbe,
        verbose: bool = False
) -> None:
    """
    Helper function for common streaming chat completion assertions.
    """
    tester.assertTrue(
        len(probe.res) > 0,
        "Expecting a non-empty response."
    )
    tester.assertTrue(
        probe.chunk_count > 1,
        f"Expecting more than one chunk, got {probe.chunk_count}"
    )
    tester.assertTrue(
        probe.chunks_with_protocol > 1,
        f"Expecting more than one chunk with protocol, got {probe.chunks_with_protocol}"
    )
    tester.assertTrue(
        probe.chunks_with_protocol == probe.chunk_count,
        f"Expecting all chunks to have protocol, got {probe.chunks_with_protocol} out of {probe.chunk_count}"
    )
    if verbose:
        print(f'Chunk count: {probe.chunk_count}')
        print(f'Chunks with protocol: {probe.chunks_with_protocol}')
        print(f'Result:\n{probe.res}')


def common_primary_chatbot_assertions(
        tester: TestCase,
        fixture: ChatBot,
        response: str
):
    """
    Helper function for common primary chatbot assertions.
    """
    # Check the state of the ChatBot.
    tester.assertTrue(
        len(fixture.messages) > 1,
        "Expecting more than one message in the stack."
    )
    tester.assertTrue(
        fixture.messages[-1]['role'] == 'assistant',
        "Expecting the last message to be from the assistant."
    )

    # Check that ChatBot.chat() stores the result in the stack.
    tester.assertTrue(
        len(fixture.messages[-1]['content']) != 0,
        "Expecting the last message to have content."
    )
    tester.assertTrue(
        fixture.messages[-1]['content'] == response,
    )


def common_secondary_chatbot_assertions(
        tester: TestCase,
        fixture: ChatBot
):
    """
    Helper function for common secondary chatbot assertions.
    """
    # Ensure there are 5 messages in the stack. (1 system, 2 user, 2 assistant)
    tester.assertTrue(
        len(fixture.messages) == 5,
        "Expecting 5 messages in the stack (1 system, 2 user, 2 assistant)."
    )
    # Check that the messages are in the correct order.
    tester.assertTrue(
        fixture.messages[0]['role'] == 'system',
        "Expecting the first message to be from the system."
    )
    tester.assertTrue(
        fixture.messages[1]['role'] == 'user',
        "Expecting the first message to be from the user."
    )
    tester.assertTrue(
        fixture.messages[2]['role'] == 'assistant',
        "Expecting the second message to be from the assistant."
    )
    tester.assertTrue(
        fixture.messages[3]['role'] == 'user',
        "Expecting the third message to be from the user."
    )
    tester.assertTrue(
        fixture.messages[4]['role'] == 'assistant',
        "Expecting the last message to be from the assistant."
    )

    print(f'ChatBot messages:\n{fixture.messages}')


#######################################################################################################################
# Reusable tests
#######################################################################################################################

# LLM Wrapper tests ---------------------------------------------------------------------------------------------------

def test_complete_chat(
    tester: TestCase,
    fixture: LanguageModelWrapper,
    verbose: bool = False
) -> None:
    messages = [{"role": "user", "content": "What should I eat for lunch today?"}]
    response = fixture.complete_chat(messages, append_role='assistant')

    common_chat_assertions(tester=tester, response=response, verbose=verbose)


def test_text_completion_success(
    tester: TestCase,
    fixture: LanguageModelWrapper,
    verbose: bool = False
) -> None:
    prompt = "Three countries in North America are: "
    response = fixture.text_completion(prompt)

    common_text_assertions(tester=tester, response=response, verbose=verbose)


def test_text_completion_failure(
    tester: TestCase,
    fixture: LanguageModelWrapper,
    verbose: bool = False
) -> None:
    exception = None
    try:
        prompt = "The capital of France is: "
        response = fixture.text_completion(prompt)
        if verbose:
            print(f'Response:\n{response}')
    except Exception as e:
        exception = e

    tester.assertTrue(exception is not None, "Expecting an exception.")


# Streaming LLM wrapper tests -----------------------------------------------------------------------------------------

def test_streaming_complete_chat(
        tester: TestCase,
        fixture: StreamingLanguageModelWrapper,
        verbose: bool = False
) -> None:
    """
    Test the complete_chat() method of the StreamingLanguageModelWrapper.
    """
    messages = [{"role": "user", "content": "What should I eat for lunch today?"}]
    generator = fixture.complete_chat(messages, append_role='assistant')

    results: StreamingChatCompletionProbe = probe_streaming_chat_completion(generator)

    common_streaming_chat_assertions(tester=tester, probe=results, chunk_time_seconds_threshold=0.2, verbose=verbose)


def test_streaming_complete_chat_sse(
        tester: TestCase,
        fixture: StreamingLanguageModelWrapper,
        verbose: bool = False
) -> None:
    messages = [{"role": "user", "content": "What should I eat for lunch today?"}]
    generator = fixture.complete_chat(messages, append_role='assistant')

    tester.assertTrue(isinstance(generator, Generator), "Expecting a generator.")

    results: StreamingSSECompletionProbe = probe_streaming_sse_completions(generator)

    common_streaming_sse_assertions(tester=tester, probe=results, verbose=verbose)


def test_streaming_text_completion_success(
    tester: TestCase,
    fixture: StreamingLanguageModelWrapper,
    verbose: bool = False
) -> None:
    prompt = "Three countries in North America are: "
    generator = fixture.text_completion(prompt)

    result: StreamingTextCompletionProbe = probe_streaming_text_completion(generator)

    common_streaming_text_assertions(tester=tester, probe=result, verbose=verbose)


def test_streaming_text_completion_failure(
    tester: TestCase,
    fixture: StreamingLanguageModelWrapper,
    verbose: bool = False
) -> None:
    prompt = "The capital of Canada is"
    generator = fixture.text_completion(prompt)

    tester.assertTrue(isinstance(generator, Generator), "Expecting a generator.")

    exception = None
    try:
        # Convert the generator to a list to evaluate it.
        list(generator)
    except Exception as e:
        exception = e

    if verbose:
        print(f'Exception:\n{exception}')

    tester.assertTrue(exception is not None, "Expecting an exception.")


def test_streaming_text_completion_sse(
    tester: TestCase,
    fixture: StreamingLanguageModelWrapper,
    verbose: bool = False
) -> None:
    prompt = "Three countries in North America are: "
    generator = fixture.text_completion(prompt)

    tester.assertTrue(isinstance(generator, Generator), "Expecting a generator.")

    results: StreamingSSECompletionProbe = probe_streaming_sse_completions(generator)

    common_streaming_sse_assertions(tester=tester, probe=results, verbose=verbose)


# ChatBot tests -------------------------------------------------------------------------------------------------------

def test_chatbot_chat(
    tester: TestCase,
    fixture: ChatBot,
    verbose: bool = False
) -> None:
    """
    Test the chat() method of the ChatBot for non-streaming wrappers.
    """
    response = fixture.chat('Who are you?')

    tester.assertTrue(isinstance(response, str), f"Expecting a string, got {type(response)}.")

    if verbose:
        print(f'ChatBot response: {response}')

    # Check that the ChatBot is in the correct state.
    common_primary_chatbot_assertions(tester, fixture, response)

    response = fixture.chat('What is your name?')

    tester.assertTrue(isinstance(response, str), f"Expecting a string, got {type(response)}.")

    if verbose:
        print(f'ChatBot response: {response}')


def test_chatbot_resend(
    tester: TestCase,
    fixture: ChatBot,
    verbose: bool = False
) -> None:
    """
    Test the resend() method of the ChatBot for non-streaming wrappers.
    """
    m = [{'role': 'system', 'content': "You are a robot that adds 'YO!' to the end of every sentence."},
         {'role': 'user', 'content': 'Tell me about Poland.'}]
    fixture.messages = copy.deepcopy(m)

    response = fixture.resend()

    tester.assertTrue(isinstance(response, str), f"Expecting a string, got {type(response)}.")

    if verbose:
        print(f'ChatBot response: {response}')

    # Check that the ChatBot is in the correct state.
    common_primary_chatbot_assertions(tester, fixture, response)

    tester.assertTrue(len(fixture.messages) == 3, "Expecting 3 messages.")
    tester.assertTrue(
        fixture.messages[:1] == m,
        f"Expecting first two messages to be \n{m}, got \n{fixture.messages[:1]}"
    )


# Streaming ChatBot tests ---------------------------------------------------------------------------------------------

def test_streaming_chatbot_chat(
    tester: TestCase,
    fixture: ChatBot,
    chunk_time_seconds_threshold: float = 0.2,
    verbose: bool = False
) -> None:
    """
    Test the chat() method of the ChatBot for streaming wrappers.
    """
    generator = fixture.chat('Who are you')

    tester.assertTrue(isinstance(generator, Generator), "Expecting a generator.")

    # Check the results of the generator.
    results: StreamingChatCompletionProbe = probe_streaming_chat_completion(generator)
    common_streaming_chat_assertions(
        tester=tester,
        probe=results,
        chunk_time_seconds_threshold=chunk_time_seconds_threshold,
        verbose=verbose
    )

    common_primary_chatbot_assertions(tester, fixture=fixture, response=results.res)

    # Make another call to ChatBot.chat() to ensure it is capable of receiving a new message.
    generator = fixture.chat('Where do you come from?')

    # Check the results of the generator.
    results: StreamingChatCompletionProbe = probe_streaming_chat_completion(generator)
    common_streaming_chat_assertions(
        tester=tester,
        probe=results,
        chunk_time_seconds_threshold=chunk_time_seconds_threshold,
        verbose=verbose
    )

    common_secondary_chatbot_assertions(tester, fixture=fixture)


def test_streaming_chatbot_resend(
    tester: TestCase,
    fixture: ChatBot,
    verbose: bool = False
) -> None:
    """
    Test the resend() method of the ChatBot for streaming wrappers.
    """
    m = [{'role': 'system', 'content': "You are a robot that adds 'YO!' to the end of every sentence."},
         {'role': 'user', 'content': 'Tell me about Poland.'}]
    fixture.messages = copy.deepcopy(m)

    generator = fixture.resend()

    tester.assertTrue(isinstance(generator, Generator), "Expecting a generator.")

    # Check the results of the generator.
    results: StreamingChatCompletionProbe = probe_streaming_chat_completion(generator)
    common_streaming_chat_assertions(
        tester=tester,
        probe=results,
        chunk_time_seconds_threshold=0.2,
        verbose=verbose
    )

    common_primary_chatbot_assertions(tester, fixture=fixture, response=results.res)

    tester.assertTrue(len(fixture.messages) == 3, "Expecting 3 messages.")
    tester.assertTrue(
        fixture.messages[:2] == m,
        f"Expecting first two messages to be \n{m}, got \n{fixture.messages[:2]}"
    )
