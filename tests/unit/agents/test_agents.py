from unittest import TestCase

from phasellm.agents import SandboxedCodeExecutionAgent


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