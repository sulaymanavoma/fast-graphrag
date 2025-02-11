__all__ = ['DefaultLLMService', 'DefaultEmbeddingService']

from ._llm_openai import OpenAIEmbeddingService, OpenAILLMService, OpenAILLMServiceStrong


class DefaultLLMService(OpenAILLMService):
    pass
class DefaultEmbeddingService(OpenAIEmbeddingService):
    pass
class DefaultLLMServiceStrong(OpenAILLMServiceStrong):
    pass