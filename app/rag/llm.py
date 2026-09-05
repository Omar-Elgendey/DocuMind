import logging
import os
from abc import ABC, abstractmethod
from typing import Optional

from groq import Groq


logger = logging.getLogger(__name__)


class LLMGenerationError(RuntimeError):
    """Raised when LLM generation fails or returns invalid output."""
    pass


class BaseLLMProvider(ABC):
    """Interface for LLM providers."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate a response from the LLM."""
        pass


class GroqProvider(BaseLLMProvider):
    """LLM provider implementation using the Groq API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "openai/gpt-oss-20b",
    ):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")

        if not self.api_key:
            raise LLMGenerationError(
                "GROQ_API_KEY environment variable is missing."
            )

        self.model_name = model_name

        try:
            self.client = Groq(api_key=self.api_key)
        except Exception as exc:
            logger.exception("Failed to initialize Groq client.")
            raise LLMGenerationError(
                "Failed to initialize Groq client."
            ) from exc

    def generate(self, prompt: str) -> str:
        """Generate a response using the configured Groq model."""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
            )

            content = response.choices[0].message.content

            if not content:
                raise LLMGenerationError(
                    "Groq returned an empty response."
                )

            return content

        except LLMGenerationError:
            raise

        except Exception as exc:
            logger.exception(
                "Groq API generation failed using model '%s'.",
                self.model_name,
            )
            raise LLMGenerationError(
                f"Groq API generation failed for model '{self.model_name}'."
            ) from exc


class LLMGenerator:
    """Orchestrates LLM generation through an injected provider."""

    def __init__(self, provider: BaseLLMProvider):
        if not isinstance(provider, BaseLLMProvider):
            raise TypeError(
                "provider must be an instance of BaseLLMProvider."
            )

        self.provider = provider

    def generate(self, prompt: str) -> str:
        """Validate input, generate a response, and validate the output."""

        if not prompt or not prompt.strip():
            raise ValueError(
                "Prompt cannot be empty or whitespace-only."
            )

        try:
            response = self.provider.generate(prompt)

        except LLMGenerationError:
            raise

        except Exception as exc:
            logger.exception("Unexpected provider error.")
            raise LLMGenerationError(
                "Unexpected error during LLM generation."
            ) from exc

        if not response or not response.strip():
            raise LLMGenerationError(
                "LLM provider returned an empty response."
            )

        return response.strip()