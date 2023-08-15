import openai

from importlib import reload

from abc import ABC, abstractmethod

from azure.identity import DefaultAzureCredential


class APIConfiguration(ABC):

    def __init__(self, model: str = 'gpt-3.5-turbo'):
        self.model = model

    @abstractmethod
    def __call__(self):
        """
        Abstract method to initialize the API configuration.

        Returns:

        """
        pass

    @abstractmethod
    def get_base_api_kwargs(self):
        """
        Abstract method for the base API kwargs for the API configuration.

        Returns:

        """
        pass


class OpenAIConfiguration(APIConfiguration):
    name = 'openai'

    def __init__(
            self,
            api_key: str,
            organization: str = None,
            model: str = 'gpt-3.5-turbo'
    ):
        """
        Initializes the OpenAI API configuration.

        Args:
            api_key: The OpenAI API key.
            organization: The OpenAI organization.
            model: The model to use.

        """
        super().__init__(model=model)

        self.api_key = api_key
        self.organization = organization

    def __call__(self):
        """
        Calls the OpenAI API configuration to initialize the OpenAI API.

        Returns:

        """
        # Reset the OpenAI API configuration.
        reload(openai)

        # Set the OpenAI API configuration.
        openai.api_key = self.api_key
        openai.organization = self.organization

    def get_base_api_kwargs(self):
        """
        Returns the base API kwargs for the OpenAI API configuration.

        Returns:
            A Dict of the base API kwargs for the OpenAI API configuration.

        """
        return {
            'model': self.model
        }


class AzureAPIConfiguration(APIConfiguration):
    name = 'azure'

    def __init__(
            self,
            api_key: str,
            api_base: str,
            api_version: str = '2023-05-15',
            deployment_id: str = 'gpt-3.5-turbo'
    ):
        """
        Initializes the Azure API configuration.

        Args:
            api_key: The Azure API key.
            api_base: The Azure API base.
            api_version: The Azure API version.
            deployment_id: The model deployment ID.

        """
        super().__init__(model=deployment_id)

        self.api_key = api_key
        self.api_base = api_base
        self.api_version = api_version

        self.deployment_id = deployment_id

    def __call__(self):
        """
        Calls the Azure API configuration to initialize the Azure API.

        Returns:

        """
        # Reset the OpenAI API configuration.
        reload(openai)

        # Set the OpenAI API configuration.
        openai.api_key = self.api_key
        openai.api_type = self.name
        openai.api_base = self.api_base
        openai.api_version = self.api_version

    def get_base_api_kwargs(self):
        """
        Returns the base API kwargs for the Azure API configuration.

        Returns:
            A Dict of the base API kwargs for the Azure API configuration.

        """
        return {
            'deployment_id': self.deployment_id
        }


class AzureActiveDirectoryConfiguration:
    name = 'azure_ad'

    def __init__(
            self,
            api_base: str,
            api_version: str = '2023-05-15',
            deployment_id: str = 'gpt-3.5-turbo'
    ):
        """
        Initializes the Azure Active Directory API configuration.

        Learn more:
        https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/switching-endpoints#azure-active-directory-authentication

        Args:
            api_base: The Azure Active Directory API base.
            api_version: The API version. https://learn.microsoft.com/en-us/azure/ai-services/openai/reference
            deployment_id: The model deployment ID

        """
        super().__init__(model=deployment_id)

        self.api_base = api_base
        self.api_version = api_version

        self.deployment_id = deployment_id

    def __call__(self):
        """
        Calls the Azure Active Directory API configuration to initialize the Azure Active Directory API.

        Returns:

        """
        # Reset the OpenAI API configuration.
        reload(openai)

        # Set the OpenAI API configuration.
        credential = DefaultAzureCredential()
        token = credential.get_token('https://cognitiveservices.azure.com/.default')

        openai.api_type = self.name
        openai.api_key = token.token
        openai.api_base = self.api_base
        openai.api_version = self.api_version

    def get_base_api_kwargs(self):
        """
        Returns the base API kwargs for the Azure Active Directory API configuration.

        Returns:
            A Dict containing the base API kwargs for the Azure Active Directory API configuration.

        """
        return {
            'deployment_id': self.deployment_id
        }
