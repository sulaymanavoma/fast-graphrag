"""LLM Services module."""

from dataclasses import dataclass, field
from typing import Optional

import instructor
from openai import AsyncOpenAI

from fast_graphrag._utils import logger

from ._llm_openai import OpenAIEmbeddingService, OpenAILLMService


@dataclass
class OllamaAILLMService(OpenAILLMService):
    """LLM Service for OpenAI LLMs."""

    model: Optional[str] = field(default=None)
    base_url: Optional[str] = field(default="http://localhost:11434/v1")

    def __post_init__(self):
        # Patch the OpenAI client with instructor
        ollama_client = AsyncOpenAI(base_url=self.base_url, api_key="ollama")
        self.llm_async_client: instructor.AsyncInstructor = instructor.from_openai(ollama_client)
        logger.debug("Initialized OllamaAILLMService with patched OpenAI client.")


@dataclass
class OllamaAIEmbeddingService(OpenAIEmbeddingService):
    """Base class for Language Model implementations."""

    embedding_dim: int = field(default=1536)
    model: Optional[str] = field(default=None)
    base_url: Optional[str] = field(default="http://localhost:11434/v1")

    def __post_init__(self):
        self.embedding_async_client: AsyncOpenAI = AsyncOpenAI(base_url=self.base_url, api_key="ollama")
        logger.debug("Initialized OllamaAIEmbeddingService with OpenAI client.")
