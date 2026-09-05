from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document
from pydantic import ValidationError

from app.rag.retriever import DocuMindRetriever


@pytest.fixture
def mock_vector_store():
    return MagicMock(name="MockChromaVectorStore")


@pytest.fixture
def sample_results():
    return [
        (
            Document(
                page_content="Cancellation is allowed within 14 days.",
                metadata={"document_id": "doc-1", "page": 2},
            ),
            0.92,
        ),
        (
            Document(
                page_content="Refunds are processed within 5 business days.",
                metadata={"document_id": "doc-1", "page": 3},
            ),
            0.81,
        ),
    ]


@pytest.fixture
def retriever(mock_vector_store):
    return DocuMindRetriever(
        vector_store=mock_vector_store,
        top_k=5,
    )


class TestGetRelevantDocuments:

    @patch("app.rag.retriever.similarity_search")
    def test_returns_documents_without_scores(
        self,
        mock_similarity_search,
        retriever,
        sample_results,
    ):
        mock_similarity_search.return_value = sample_results

        result = retriever._get_relevant_documents(
            "What are the cancellation terms?"
        )

        assert result == [doc for doc, _ in sample_results]
        assert all(isinstance(doc, Document) for doc in result)

    @patch("app.rag.retriever.similarity_search")
    def test_calls_similarity_search_with_expected_arguments(
        self,
        mock_similarity_search,
        mock_vector_store,
        sample_results,
    ):
        mock_similarity_search.return_value = sample_results

        retriever = DocuMindRetriever(
            vector_store=mock_vector_store,
            top_k=3,
            metadata_filter={"document_id": "doc-1"},
        )

        retriever._get_relevant_documents("cancellation terms")

        mock_similarity_search.assert_called_once_with(
            query="cancellation terms",
            top_k=3,
            filter={"document_id": "doc-1"},
            vector_store=mock_vector_store,
        )

    @patch("app.rag.retriever.similarity_search")
    def test_uses_default_top_k(
        self,
        mock_similarity_search,
        mock_vector_store,
        sample_results,
    ):
        mock_similarity_search.return_value = sample_results

        retriever = DocuMindRetriever(
            vector_store=mock_vector_store
        )

        retriever._get_relevant_documents("cancellation terms")

        _, kwargs = mock_similarity_search.call_args

        assert kwargs["top_k"] == 5

    @patch("app.rag.retriever.similarity_search")
    def test_preserves_result_order(
        self,
        mock_similarity_search,
        retriever,
        sample_results,
    ):
        mock_similarity_search.return_value = sample_results

        result = retriever._get_relevant_documents(
            "cancellation terms"
        )

        assert result[0].page_content == sample_results[0][0].page_content
        assert result[1].page_content == sample_results[1][0].page_content

    @patch("app.rag.retriever.similarity_search")
    def test_invoke_returns_documents(
        self,
        mock_similarity_search,
        retriever,
        sample_results,
    ):
        mock_similarity_search.return_value = sample_results

        result = retriever.invoke(
            "What are the cancellation terms?"
        )

        assert result == [doc for doc, _ in sample_results]
        assert all(isinstance(doc, Document) for doc in result)


class TestGetRelevantDocumentsValidation:

    @patch("app.rag.retriever.similarity_search")
    def test_empty_query_raises_value_error(
        self,
        mock_similarity_search,
        retriever,
    ):
        with pytest.raises(ValueError, match="cannot be empty"):
            retriever._get_relevant_documents("")

        mock_similarity_search.assert_not_called()

    @patch("app.rag.retriever.similarity_search")
    def test_whitespace_query_raises_value_error(
        self,
        mock_similarity_search,
        retriever,
    ):
        with pytest.raises(ValueError, match="cannot be empty"):
            retriever._get_relevant_documents("   \n\t")

        mock_similarity_search.assert_not_called()

    def test_missing_vector_store_raises_validation_error(self):
        with pytest.raises(ValidationError):
            DocuMindRetriever(top_k=5)

    def test_zero_top_k_raises_validation_error(
        self,
        mock_vector_store,
    ):
        with pytest.raises(ValidationError):
            DocuMindRetriever(
                vector_store=mock_vector_store,
                top_k=0,
            )

    def test_negative_top_k_raises_validation_error(
        self,
        mock_vector_store,
    ):
        with pytest.raises(ValidationError):
            DocuMindRetriever(
                vector_store=mock_vector_store,
                top_k=-1,
            )


class TestGetRelevantDocumentsErrors:

    @patch("app.rag.retriever.similarity_search")
    def test_similarity_search_failure_raises_runtime_error(
        self,
        mock_similarity_search,
        retriever,
    ):
        original_exc = ConnectionError("ChromaDB unreachable")
        mock_similarity_search.side_effect = original_exc

        with pytest.raises(
            RuntimeError,
            match="Retriever failed to fetch",
        ):
            retriever._get_relevant_documents(
                "cancellation terms"
            )

    @patch("app.rag.retriever.similarity_search")
    def test_runtime_error_preserves_original_exception(
        self,
        mock_similarity_search,
        retriever,
    ):
        original_exc = ConnectionError("ChromaDB unreachable")
        mock_similarity_search.side_effect = original_exc

        with pytest.raises(RuntimeError) as exc_info:
            retriever._get_relevant_documents(
                "cancellation terms"
            )

        assert exc_info.value.__cause__ is original_exc


class TestGetRelevantDocumentsEdgeCases:

    @patch("app.rag.retriever.similarity_search")
    def test_no_results_returns_empty_list(
        self,
        mock_similarity_search,
        retriever,
    ):
        mock_similarity_search.return_value = []

        result = retriever._get_relevant_documents(
            "something with no matches"
        )

        assert result == []

    @patch("app.rag.retriever.similarity_search")
    def test_default_metadata_filter_is_none(
        self,
        mock_similarity_search,
        mock_vector_store,
        sample_results,
    ):
        mock_similarity_search.return_value = sample_results

        retriever = DocuMindRetriever(
            vector_store=mock_vector_store
        )

        retriever._get_relevant_documents("cancellation terms")

        _, kwargs = mock_similarity_search.call_args

        assert kwargs["filter"] is None

    @patch("app.rag.retriever.similarity_search")
    def test_query_whitespace_is_preserved(
        self,
        mock_similarity_search,
        retriever,
        sample_results,
    ):
        mock_similarity_search.return_value = sample_results

        retriever._get_relevant_documents(
            "  cancellation terms  "
        )

        _, kwargs = mock_similarity_search.call_args

        assert kwargs["query"] == "  cancellation terms  "