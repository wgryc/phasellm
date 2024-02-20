def extract_vertex_ai_kwargs(kwargs: dict) -> dict:
    """
    Extracts the Vertex AI kwargs from the kwargs dictionary.
    Args:
        kwargs: The kwargs dictionary.

    Returns:
        The Vertex AI kwargs.

    """

    return {
        'max_output_tokens': kwargs['max_output_tokens'] if 'max_output_tokens' in kwargs else None,
        'candidate_count': kwargs['candidate_count'] if 'candidate_count' in kwargs else None,
        'top_p': kwargs['top_p'] if 'top_p' in kwargs else None,
        'top_k': kwargs['top_k'] if 'top_k' in kwargs else None,
        'logprobs': kwargs['logprobs'] if 'logprobs' in kwargs else None,
        'presence_penalty': kwargs['presence_penalty'] if 'presence_penalty' in kwargs else None,
        'frequency_penalty': kwargs['frequency_penalty'] if 'frequency_penalty' in kwargs else None,
        'logit_bias': kwargs['logit_bias'] if 'logit_bias' in kwargs else None
    }


def extract_vertex_ai_response_metadata(response) -> dict:
    last_response_header = {}
    if hasattr(response, '_raw_response'):
        last_response_header = {
            **last_response_header,
            **response._raw_response.PromptFeedback.to_dict(response._raw_response.prompt_feedback),
            **response._raw_response.UsageMetadata.to_dict(response._raw_response.usage_metadata)
        }
    if hasattr(response, '_prediction_response'):
        last_response_header = {
            **last_response_header,
            **response._prediction_response.metadata
        }
    if hasattr(response, 'safety_attributes'):
        last_response_header = {
            **last_response_header,
            **response.safety_attributes
        }
    return last_response_header
