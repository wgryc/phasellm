from unittest import TestCase

from unittest.mock import patch

from phasellm.agents import CodeExecutionAgent, SandboxedCodeExecutionAgent, EmailSenderAgent, NewsSummaryAgent


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
