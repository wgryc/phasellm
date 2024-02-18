import re

from warnings import warn


def coerce_azure_base_url(url: str) -> str:
    """
    This function coerces the base URL to the proper format for the Azure OpenAI API. This is used for backwards
    compatibility of base_url and api_base arguments.
    Args:
        url: The url to coerce.

    Returns:
        The coerced URL.

    """
    match = re.match(r'https:\/\/.*\.openai\.azure.com\/openai\/deployments\/.*', url)
    if not match:
        # Ensure proper format of the base URL.
        res = re.search(r'https:\/\/.*\.openai\.azure.com(?!\/openai\/deployments\/)', url)
        if res.group():
            # see https://github.com/openai/openai-python/blob/v1/examples/azure.py
            warn('The base_url argument must be in the format:'
                 'https://{resource}.openai.azure.com/openai/deployments/{model}\n'
                 'Attempting to coerce base_url to the proper format.')
            url = f"{url[:res.end()]}/openai/deployments{url[res.end():]}"
            warn(f'Coerced url to: {url}')
            return url
    return url
