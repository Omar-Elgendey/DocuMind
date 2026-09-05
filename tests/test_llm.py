import pytest
from unittest.mock import MagicMock
from app.rag.llm import (
    LLMGenerator,
    BaseLLMProvider,
    GroqProvider,
    LLMGenerationError,
)


@pytest.fixture
def mock_provider():
    """Fixture providing a mock implementation of BaseLLMProvider."""
    return MagicMock(spec=BaseLLMProvider)


# --- Tests for GroqProvider Initialization ---

def test_groq_provider_missing_api_key_raises_error(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(LLMGenerationError, match="GROQ_API_KEY environment variable is missing."):
        GroqProvider(api_key=None)


# --- Tests for LLMGenerator ---

def test_llm_generator_invalid_provider_raises_type_error():
    with pytest.raises(TypeError, match="provider must be an instance of BaseLLMProvider."):
        LLMGenerator(provider="invalid_provider")


def test_generate_happy_path(mock_provider):
    mock_provider.generate.return_value = "DocuMind is a modular RAG system."
    generator = LLMGenerator(provider=mock_provider)
    prompt = "What is DocuMind?"

    result = generator.generate(prompt)

    assert result == "DocuMind is a modular RAG system."
    mock_provider.generate.assert_called_once_with(prompt)


@pytest.mark.parametrize("invalid_prompt", ["", "   ", "\n\t"])
def test_generate_invalid_prompt_raises_value_error(mock_provider, invalid_prompt):
    generator = LLMGenerator(provider=mock_provider)

    with pytest.raises(ValueError, match="Prompt cannot be empty or whitespace-only."):
        generator.generate(invalid_prompt)

    mock_provider.generate.assert_not_called()


def test_generate_provider_failure_raises_llm_generation_error(mock_provider):
    mock_provider.generate.side_effect = LLMGenerationError("API timeout error")
    generator = LLMGenerator(provider=mock_provider)

    with pytest.raises(LLMGenerationError, match="API timeout error"):
        generator.generate("Valid prompt string")


def test_generate_unexpected_provider_error_raises_llm_generation_error(mock_provider):
    mock_provider.generate.side_effect = Exception("Unexpected network crash")
    generator = LLMGenerator(provider=mock_provider)

    with pytest.raises(LLMGenerationError, match="Unexpected error during LLM generation."):
        generator.generate("Valid prompt string")


@pytest.mark.parametrize("invalid_response", [None, "", "   ", "\n"])
def test_generate_invalid_response_raises_llm_generation_error(mock_provider, invalid_response):
    mock_provider.generate.return_value = invalid_response
    generator = LLMGenerator(provider=mock_provider)

    with pytest.raises(LLMGenerationError, match="LLM provider returned an empty response."):
        generator.generate("Valid prompt string")


def test_generate_preserves_prompt_integrity(mock_provider):
    mock_provider.generate.return_value = "Valid answer."
    generator = LLMGenerator(provider=mock_provider)
    exact_prompt = "Context: [Chunk 1]\nUser Query: Hello World?\nAnswer:"

    generator.generate(exact_prompt)

    mock_provider.generate.assert_called_once_with(exact_prompt)