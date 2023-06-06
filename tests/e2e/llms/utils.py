import time

from unittest import TestCase

from types import GeneratorType

from dataclasses import dataclass


@dataclass
class StreamingChatCompletionProbe:
    res: str
    chunk_times: list
    mean_chunk_time: float


def probe_streaming_chat_completion(generator: GeneratorType):
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
        res = res + chunk

    # Compute the mean chunk time
    mean_chunk_time = sum(chunk_times) / len(chunk_times)

    return StreamingChatCompletionProbe(res=res, chunk_times=chunk_times, mean_chunk_time=mean_chunk_time)


def common_streaming_chat_assertions(
        tester: TestCase,
        probe: StreamingChatCompletionProbe,
        chunk_time_seconds_threshold: float,
        verbose: bool = False
):
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


def probe_streaming_text_completion(generator: GeneratorType):
    """
    Helper function for testing streaming text completion.
    """
    res = ''
    chunk_count = 0
    for chunk in generator:
        chunk_count += 1
        res = res + chunk

    return StreamingTextCompletionProbe(res=res, chunk_count=chunk_count)


def common_streaming_text_assertions(
        tester: TestCase,
        probe: StreamingTextCompletionProbe,
        verbose: bool = False
):
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
