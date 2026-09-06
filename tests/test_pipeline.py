from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from app.rag.pipeline import RAGPipeline
from app.rag.prompt import build_prompt, extract_sources
from app.rag.llm import LLMGenerationError


@pytest.fixture
def sample_documents():
    return [
        Document(
            page_content="Cancellation is allowed within 14 days.",
            metadata={"document_id": "doc-1", "page": 2},
        ),
        Document(
            page_content="Refunds are processed within 5 business days.",
            metadata={"document_id": "doc-1", "page": 3},
        ),
    ]


@pytest.fixture
def mock_retriever(sample_documents):
    retriever = MagicMock(name="MockDocuMindRetriever")
    effective_retriever = MagicMock(name="MockEffectiveRetriever")
    effective_retriever.invoke.return_value = sample_documents
    retriever.model_copy.return_value = effective_retriever
    return retriever


@pytest.fixture
def mock_generator():
    generator = MagicMock(name="MockLLMGenerator")
    generator.generate.return_value = "Cancellation is allowed within 14 days."
    return generator


@pytest.fixture
def pipeline(mock_retriever, mock_generator):
    return RAGPipeline(retriever=mock_retriever, generator=mock_generator)


class TestRunHappyPath:
    def test_returns_expected_shape(self, pipeline, sample_documents):
        result = pipeline.run("What are the cancellation terms?")

        assert set(result.keys()) == {"answer", "sources"}
        assert result["answer"] == "Cancellation is allowed within 14 days."
        assert result["sources"] == extract_sources(sample_documents)

    def test_creates_retriever_copy_with_requested_settings(self, pipeline, mock_retriever):
        pipeline.run("cancellation terms", top_k=8, filter={"document_id": "doc-1"})

        mock_retriever.model_copy.assert_called_once_with(
            update={"top_k": 8, "metadata_filter": {"document_id": "doc-1"}}
        )

    def test_uses_default_top_k_and_filter_when_not_provided(self, pipeline, mock_retriever):
        pipeline.run("cancellation terms")

        mock_retriever.model_copy.assert_called_once_with(
            update={"top_k": 5, "metadata_filter": None}
        )

    def test_invokes_effective_retriever_with_plain_query_string(self, pipeline, mock_retriever):
        pipeline.run("cancellation terms")

        effective_retriever = mock_retriever.model_copy.return_value
        effective_retriever.invoke.assert_called_once_with("cancellation terms")

    def test_generator_receives_the_actual_built_prompt(
        self, pipeline, mock_generator, sample_documents
    ):
        query = "What are the cancellation terms?"
        expected_prompt = build_prompt(query, sample_documents)

        pipeline.run(query)

        mock_generator.generate.assert_called_once_with(expected_prompt)


class TestRunInvalidInputPropagation:
    def test_propagates_value_error_from_retriever_unchanged(self, pipeline, mock_retriever):
        effective_retriever = mock_retriever.model_copy.return_value
        effective_retriever.invoke.side_effect = ValueError(
            "Search query cannot be empty or whitespace-only."
        )

        with pytest.raises(ValueError, match="cannot be empty"):
            pipeline.run("")


class TestRunOperationFailure:
    def test_propagates_retriever_runtime_error_unchanged(self, pipeline, mock_retriever):
        effective_retriever = mock_retriever.model_copy.return_value
        effective_retriever.invoke.side_effect = RuntimeError("Vector store unreachable.")

        with pytest.raises(RuntimeError, match="Vector store unreachable"):
            pipeline.run("cancellation terms")

    def test_propagates_generator_llm_generation_error_unchanged(self, pipeline, mock_generator):
        mock_generator.generate.side_effect = LLMGenerationError("Groq API generation failed.")

        with pytest.raises(LLMGenerationError, match="Groq API generation failed"):
            pipeline.run("cancellation terms")

    def test_does_not_wrap_exceptions_in_a_different_type(self, pipeline, mock_generator):
        mock_generator.generate.side_effect = LLMGenerationError("boom")

        try:
            pipeline.run("cancellation terms")
        except Exception as exc:
            assert isinstance(exc, LLMGenerationError)
            assert exc.__cause__ is None  # not re-raised via `from exc`


class TestRunEdgeCases:
    def test_empty_documents_returns_fallback_answer_and_no_sources(
        self, pipeline, mock_retriever, mock_generator
    ):
        effective_retriever = mock_retriever.model_copy.return_value
        effective_retriever.invoke.return_value = []
        mock_generator.generate.return_value = (
            "I don't have enough information from the provided documents to answer that."
        )

        result = pipeline.run("something unrelated to the document")

        assert result["sources"] == []
        assert "don't have enough information" in result["answer"]

    def test_empty_documents_still_builds_a_fallback_prompt_for_the_generator(
        self, pipeline, mock_retriever, mock_generator
    ):
        effective_retriever = mock_retriever.model_copy.return_value
        effective_retriever.invoke.return_value = []

        query = "something unrelated to the document"
        expected_prompt = build_prompt(query, [])

        pipeline.run(query)

        mock_generator.generate.assert_called_once_with(expected_prompt)