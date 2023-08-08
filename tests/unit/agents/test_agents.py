import itertools

from unittest import TestCase

from tests.utils import Timeout

from unittest.mock import patch

from phasellm.agents import CodeExecutionAgent, SandboxedCodeExecutionAgent, EmailSenderAgent, NewsSummaryAgent, \
    WebpageAgent, RSSAgent


class TestCodeExecutionAgent(TestCase):

    def test_execute_code(self):
        """
        Tests that the CodeExecutionAgent can execute code.
        Returns:

        """
        fixture = CodeExecutionAgent()

        code = (
            "print('Hello, world!')\n"
        )

        expected = 'Hello, world!\n'

        actual = fixture.execute_code(code)

        self.assertTrue(actual == expected, f"{actual}\n!=\n{expected}")


class TestSandboxedCodeExecutionAgent(TestCase):

    def setUp(self) -> None:
        self.fixture = SandboxedCodeExecutionAgent()
        self.modules_to_include = [
            "os",
            "sys",
            "numpy",
            "pandas",
            "scipy",
            "sklearn",
            "matplotlib",
            "seaborn",
            "statsmodels",
            "tensorflow",
            "torch"
        ]
        self.packages_to_include = [
            "numpy",
            "pandas",
            "scipy",
            "scikit-learn",
            "matplotlib",
            "seaborn",
            "statsmodels",
            "tensorflow",
            "torch"
        ]

    def tearDown(self) -> None:
        self.fixture.close()

    def test_modules_to_packages_import_format(self):
        """
        Tests that modules contained within a code string are converted to packages (if they are in the whitelist).

        focuses on the "import {module}" format.
        """
        code = "\n".join([f"import {package}" for package in self.modules_to_include]) + \
               "\n\nprint('Hello, world!')\n"

        packages = self.fixture._modules_to_packages(code=code)

        self.assertTrue("os" not in packages)

        for package in self.packages_to_include:
            self.assertTrue(package in packages, f"Package: {package} not in packages: {packages}")

    def test_modules_to_packages_from_format(self):
        """
        Tests that modules contained within a code string are converted to packages (if they are in the whitelist).

        Focuses on the "from {module} import {thing}" format.

        Returns:

        """
        code = "\n".join([f"from {package} import *" for package in self.modules_to_include]) + \
               "\n\nprint('Hello, world!')\n"

        packages = self.fixture._modules_to_packages(code=code)

        self.assertTrue("os" not in packages)

        for package in self.packages_to_include:
            self.assertTrue(package in packages, f"Package: {package} not in packages: {packages}")

    def test_modules_to_packages_from_format_with_alias(self):
        """
        Tests that modules contained within a code string are converted to packages (if they are in the whitelist).

        Focuses on the "from {module} import {thing} as {alias}" format.

        Returns:

        """
        code = "\n".join([f"from {package} import * as {package}_alias" for package in self.modules_to_include]) + \
               "\n\nprint('Hello, world!')\n"

        packages = self.fixture._modules_to_packages(code=code)

        self.assertTrue("os" not in packages)

        for package in self.packages_to_include:
            self.assertTrue(package in packages, f"Package: {package} not in packages: {packages}")


class TestEmailSenderAgent(TestCase):

    @patch('phasellm.agents.EmailSenderAgent.send_plain_email')
    def test_send_plain_email_deprecated(self, send_plain_email_mock):
        """
        Test that a plain email can be sent using the deprecated method.
        Args:
            send_plain_email_mock: A mock of the send_plain_email method.

        Returns:

        """
        fixture = EmailSenderAgent(
            sender_name='test',
            smtp='test',
            sender_address='test',
            password='test',
            port=0,
            name='Test Agent'
        )

        fixture.sendPlainEmail(recipient_email='test', subject='test', content='test')

        self.assertTrue(
            send_plain_email_mock.called_with(recipient_email='test', subject='test', content='test'),
            "send_plain_email was not called with the correct arguments."
        )

    def test_send_plain_email(self):
        # TODO consider mocking networking calls and making assertions on s.send_message()
        pass


class TestNewsSummaryAgent(TestCase):

    @patch('phasellm.agents.NewsSummaryAgent.get_query')
    def test_get_query_deprecated(self, get_query_mock):
        """
        Test that the a query can be retrieved using the deprecated method.
        Args:
            get_query_mock: A mock of the get_query method.

        Returns:

        """
        fixture = NewsSummaryAgent(
            apikey='test',
            name='Test Agent'
        )

        fixture.getQuery(query='test', days_back=0, include_descriptions=True, max_articles=1)

        self.assertTrue(
            get_query_mock.called_with(query='test', days_back=0, include_descriptions=True, max_articles=1),
            "get_query was not called with the correct arguments."
        )

    def test_get_query(self):
        # TODO consider mocking networking calls and making assertions on string output.
        pass


class TestWebpageAgent(TestCase):
    test_html_str = (
        "<html>"
        "<head>"
        "<meta charset=\"utf-8\">"
        "<script src=\"https://test.com\"></script>"
        "<link rel=\"stylesheet\" href=\"https://test.com\">"
        "<style> body { background-color: #000000; } </style>"
        "<title>Test</title>"
        "</head>"
        "<body>"
        "<p>Hello, world!</p>"
        "</body>"
        "</html>"
    )

    def setUp(self) -> None:
        self.fixture = WebpageAgent()

    def test_parse_html(self):
        actual = self.fixture._parse_html(
            html=self.test_html_str,
            text_only=False,
            body_only=False
        )
        expected = self.test_html_str
        self.assertEqual(actual, expected)

    def test_parse_html_text_only(self):
        actual = self.fixture._parse_html(
            html=self.test_html_str,
            text_only=True,
            body_only=False
        )
        expected = "TestHello, world!"
        self.assertEqual(actual, expected)

    def test_parse_html_body_only(self):
        actual = self.fixture._parse_html(
            html=self.test_html_str,
            text_only=False,
            body_only=True
        )
        expected = "<body><p>Hello, world!</p></body>"
        self.assertEqual(actual, expected)

    def test_parse_html_text_only_body_only(self):
        actual = self.fixture._parse_html(
            html=self.test_html_str,
            text_only=True,
            body_only=True
        )
        expected = "Hello, world!"
        self.assertEqual(actual, expected)

    def test_prep_headers(self):
        actual = self.fixture._prep_headers(
            headers={
                'test': 'test'
            }
        )
        expected = {
            'test': 'test',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            # 'User-Agent': UserAgent().chrome, // This is here for reference only. It is added by _prep_headers.
            'Referrer': 'https://www.google.com/',
            'Accept-Language': '*',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'max-age=0'
        }
        self.assertEqual(actual['test'], expected['test'])
        self.assertEqual(actual['Accept'], expected['Accept'])
        self.assertEqual(actual['Referrer'], expected['Referrer'])
        self.assertEqual(actual['Accept-Language'], expected['Accept-Language'])
        self.assertEqual(actual['Accept-Encoding'], expected['Accept-Encoding'])
        self.assertEqual(actual['Connection'], expected['Connection'])
        self.assertEqual(actual['Upgrade-Insecure-Requests'], expected['Upgrade-Insecure-Requests'])
        self.assertEqual(actual['Cache-Control'], expected['Cache-Control'])

        self.assertTrue('User-Agent' in actual)
        self.assertTrue('Chrome' in actual['User-Agent'])


class TestRSSAgent(TestCase):

    def setUp(self) -> None:
        self.fixture = RSSAgent(url='test')

    @patch('phasellm.agents.RSSAgent.read')
    def test_poll(self, read_mock):
        # Mock the read return values to simulate differing RSS feed responses.
        read_mock.side_effect = itertools.chain([
            [{'title': 'test 1'}],
            [{'title': 'test 2'}, {'title': 'test 1'}]
        ],
            itertools.repeat([{'title': 'test 2'}, {'title': 'test 1'}])
        )

        results = []

        # Define a timeout thread to ensure the test does not hang.
        timeout = Timeout(seconds=5)
        timeout.start()

        # Execute the poller for 2 iterations.
        with self.fixture.poll(interval=1) as poller:
            for data in poller():
                results.extend(data)
                if len(results) >= 2:
                    timeout.stop()
                    break
                timeout.check()

        self.assertTrue(len(results) == 2, f'{len(results)} != 2')
        self.assertEqual(results[0]['title'], 'test 1')
        self.assertEqual(results[1]['title'], 'test 2')
