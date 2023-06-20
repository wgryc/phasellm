import os
import re
import unittest
import requests
import sseclient

from typing import Generator

from unittest import TestCase

from dotenv import load_dotenv

from flask import Flask, Response

from multiprocessing import Process

from phasellm.llms import StreamingOpenAIGPTWrapper, _format_sse

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")


def mock_generator_failure() -> Generator:
    """
    Mock generator for a failure mode of sse streaming.

    Desired output on client side is:
        '''
        123

        456
        78
        '''
    Returns:

    """
    yield "data: 1\n\n"
    yield "data: 2\n\n"
    yield "data: 3\n\n4\n\n"
    yield "data: 5\n\n"
    yield "data: 6\n\n\n"
    yield "data: 7\n\n"
    yield "data: 8\n\n\n\n\n"
    yield "data: 9\n\n"
    yield "data: <|END|>\n\n"


def mock_generator_success() -> Generator:
    """
    Mock generator for a success mode of sse streaming.

    Desired output on client side is:
        '''
        123

        456
        78
        '''
    Returns:

    """
    yield "data: 1\n\n"
    yield "data: 2\n\n"
    yield "data: 3\ndata:\ndata:4\n\n"
    yield "data: 5\n\n"
    yield "data: 6\ndata:\n\n"
    yield "data: 7\n\n"
    yield "data: 8\ndata:\ndata:\ndata:\n\n"
    yield "data: 9\n\n"
    yield "data: <|END|>\n\n"


def mock_generator_success_format_sse() -> Generator:
    """
    Mock generator for a success mode of sse streaming.

    Desired output on client side is:
        '''
        123

        456
        78
        '''
    Returns:

    """
    yield _format_sse("1")
    yield _format_sse("2")
    yield _format_sse("3\n\n4")
    yield _format_sse("5")
    yield _format_sse("6\n")
    yield _format_sse("7")
    yield _format_sse("8\n\n\n")
    yield _format_sse("9")
    yield _format_sse("<|END|>")


def server_mock(generator: Generator):
    """
    SSE test server.
    Returns:

    """
    app = Flask(__name__)

    @app.route('/stream')
    def stream():
        return Response(generator, mimetype="text/event-stream")

    app.run(debug=False, port=5000, host='0.0.0.0')


def process_stream() -> str:
    url = 'http://localhost:5000/stream'
    headers = {'Accept': 'text/event-stream'}

    res = requests.get(url, headers=headers, stream=True)
    client = sseclient.SSEClient(res)
    data = []
    for event in client.events():
        if event.data == "<|END|>":
            break
        else:
            data.append(event.data)
    client.close()
    res = ''.join(data)
    return res


def server_success_mock():
    print(''.join(mock_generator_success()))
    server_mock(mock_generator_success())


def server_failure_mock():
    print(''.join(mock_generator_failure()))
    server_mock(mock_generator_failure())


def print_intercept_generator(generator: Generator) -> Generator:
    res = []
    for item in generator:
        res.append(item)
        yield item
    print(''.join(res))


def server_llm():
    llm = StreamingOpenAIGPTWrapper(
        apikey=openai_api_key, model='text-davinci-003', format_sse=True, append_stop_token=True
    )
    generator: Generator = llm.text_completion(
        "List two countries with two new line characters between them. "
        "Example:\n"
        "USA\n\nCanada\n\n"
    )

    # Line below is for debugging purposes.
    # generator: Generator = print_intercept_generator(generator)

    server_mock(generator)


class TestSSE(TestCase):

    def test_sse_client_server_mock_success(self):
        """
        Test SSE success mode using a mock generator.
        Returns:

        """
        # Start test server
        process = Process(target=server_success_mock)
        process.start()

        res = process_stream()

        self.assertEqual(res, "123\n\n456\n78\n\n\n9")

        self.tearDown()

        process.terminate()
        process.join()

    def test_sse_client_server_mock_failure(self):
        """
        Test SSE failure mode using a mock generator.
        Returns:

        """
        # Start test server
        process = Process(target=server_failure_mock)
        process.start()

        res = process_stream()

        # Notice the missing 4. Notice the lack of newlines.
        self.assertEqual(res, "12356789")

        process.terminate()
        process.join()

    def test_sse_client_server_llm(self):
        """
        Test SSE success mode using an LLM wrapper.
        Returns:

        """
        # Start test server
        process = Process(target=server_llm)
        process.start()

        res = process_stream()

        print(repr(res))

        matches = re.findall(r'\w+\n\n\w+', res)
        self.assertTrue(len(matches) > 0, "Expected a word followed by two newlines, followed by a word.")

        process.terminate()
        process.join()

    def test_success_generator_equality(self):
        """
        Test equality of success generators.
        Returns:

        """
        self.assertEqual(list(mock_generator_success()), list(mock_generator_success_format_sse()))


if __name__ == '__main__':
    unittest.main()
