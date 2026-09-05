import pytest
from langchain_core.documents import Document

from app.rag.prompt import build_prompt, extract_sources


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


class TestBuildPromptHappyPath:
    def test_includes_query_and_context(self, sample_documents):
        result = build_prompt("What are the cancellation terms?", sample_documents)

        assert "What are the cancellation terms?" in result
        assert "Cancellation is allowed within 14 days." in result
        assert "Refunds are processed within 5 business days." in result

    def test_returns_a_string(self, sample_documents):
        result = build_prompt("cancellation terms", sample_documents)
        assert isinstance(result, str)

    def test_preserves_chunk_order(self, sample_documents):
        result = build_prompt("cancellation terms", sample_documents)
        first_pos = result.find("Cancellation is allowed within 14 days.")
        second_pos = result.find("Refunds are processed within 5 business days.")
        assert first_pos < second_pos

    def test_strips_surrounding_whitespace_from_query_in_output(self, sample_documents):
        result = build_prompt("  cancellation terms  ", sample_documents)
        assert "User Query: cancellation terms" in result


class TestBuildPromptInvalidInput:
    def test_empty_query_raises_value_error(self, sample_documents):
        with pytest.raises(ValueError, match="cannot be empty"):
            build_prompt("", sample_documents)

    def test_whitespace_only_query_raises_value_error(self, sample_documents):
        with pytest.raises(ValueError, match="cannot be empty"):
            build_prompt("   \n\t  ", sample_documents)


class TestBuildPromptEdgeCases:
    def test_empty_documents_does_not_raise(self):
        result = build_prompt("cancellation terms", [])
        assert isinstance(result, str)

    def test_empty_documents_includes_query_and_no_context_message(self):
        result = build_prompt("cancellation terms", [])
        assert "cancellation terms" in result
        assert "No relevant document context was found" in result

    def test_single_document(self):
        docs = [Document(page_content="Only one chunk.", metadata={"document_id": "doc-1"})]
        result = build_prompt("a question", docs)
        assert "Only one chunk." in result


class TestExtractSourcesHappyPath:
    def test_returns_expected_metadata(self, sample_documents):
        result = extract_sources(sample_documents)

        assert result == [
            {"document_id": "doc-1", "page": 2},
            {"document_id": "doc-1", "page": 3},
        ]

    def test_preserves_input_order(self, sample_documents):
        result = extract_sources(sample_documents)
        assert result[0]["page"] == 2
        assert result[1]["page"] == 3

    def test_returns_a_list_of_dicts(self, sample_documents):
        result = extract_sources(sample_documents)
        assert isinstance(result, list)
        assert all(isinstance(item, dict) for item in result)


class TestExtractSourcesOperationFailure:
    def test_missing_document_id_raises_runtime_error(self):
        docs = [Document(page_content="No id here.", metadata={"page": 1})]

        with pytest.raises(RuntimeError, match="missing required 'document_id'"):
            extract_sources(docs)

    def test_empty_string_document_id_raises_runtime_error(self):
        docs = [Document(page_content="Empty id.", metadata={"document_id": "", "page": 1})]

        with pytest.raises(RuntimeError, match="missing required 'document_id'"):
            extract_sources(docs)

    def test_failure_on_second_document_still_raises(self, sample_documents):
        bad_doc = Document(page_content="No id.", metadata={"page": 9})
        docs = [sample_documents[0], bad_doc]

        with pytest.raises(RuntimeError):
            extract_sources(docs)


class TestExtractSourcesEdgeCases:
    def test_empty_documents_returns_empty_list(self):
        result = extract_sources([])
        assert result == []

    def test_missing_page_defaults_to_none(self):
        docs = [Document(page_content="No page info.", metadata={"document_id": "doc-2"})]

        result = extract_sources(docs)

        assert result == [{"document_id": "doc-2", "page": None}]