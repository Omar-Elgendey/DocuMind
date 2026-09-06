from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from app.rag.splitter import split_documents


MODULE_PATH = "app.rag.splitter"


def _make_documents(n=1, content="some content here", **metadata):
    return [
        Document(page_content=content, metadata=dict(metadata))
        for _ in range(n)
    ]


class TestSplitDocumentsInvalidInput:

    def test_empty_document_list_returns_empty_list(self):
        result = split_documents([])

        assert result == []

    def test_chunk_size_zero_raises_value_error(self):
        docs = _make_documents()

        with pytest.raises(ValueError, match="chunk_size must be greater than 0"):
            split_documents(docs, chunk_size=0)

    def test_negative_chunk_size_raises_value_error(self):
        docs = _make_documents()

        with pytest.raises(ValueError, match="chunk_size must be greater than 0"):
            split_documents(docs, chunk_size=-10)

    def test_negative_chunk_overlap_raises_value_error(self):
        docs = _make_documents()

        with pytest.raises(ValueError, match="chunk_overlap must be >= 0"):
            split_documents(docs, chunk_overlap=-1)

    def test_chunk_overlap_equal_to_chunk_size_raises_value_error(self):
        docs = _make_documents()

        with pytest.raises(ValueError, match="chunk_overlap must be >= 0"):
            split_documents(docs, chunk_size=500, chunk_overlap=500)

    def test_chunk_overlap_greater_than_chunk_size_raises_value_error(self):
        docs = _make_documents()

        with pytest.raises(ValueError, match="chunk_overlap must be >= 0"):
            split_documents(docs, chunk_size=500, chunk_overlap=600)

    def test_empty_document_id_raises_value_error(self):
        docs = _make_documents()

        with pytest.raises(ValueError, match="document_id must not be empty"):
            split_documents(docs, document_id="")

    def test_whitespace_only_document_id_raises_value_error(self):
        docs = _make_documents()

        with pytest.raises(ValueError, match="document_id must not be empty"):
            split_documents(docs, document_id="   ")


class TestSplitDocumentsHappyPath:

    def test_splits_long_document_into_multiple_chunks(self):
        long_text = "word " * 500
        docs = _make_documents(content=long_text, source="report.pdf")

        result = split_documents(docs, chunk_size=200, chunk_overlap=20)

        assert len(result) > 1
        assert all(isinstance(chunk, Document) for chunk in result)

    def test_short_document_returns_single_chunk(self):
        docs = _make_documents(content="short text", source="note.txt")

        result = split_documents(docs, chunk_size=1000, chunk_overlap=200)

        assert len(result) == 1
        assert result[0].page_content == "short text"

    def test_preserves_original_metadata(self):
        docs = _make_documents(content="short text", source="note.txt", page=2)

        result = split_documents(docs, chunk_size=1000, chunk_overlap=200)

        assert result[0].metadata["source"] == "note.txt"
        assert result[0].metadata["page"] == 2

    def test_filters_out_empty_or_whitespace_only_chunks(self):
        docs = [
            Document(page_content="real content", metadata={}),
            Document(page_content="   ", metadata={}),
            Document(page_content="", metadata={}),
        ]

        result = split_documents(docs, chunk_size=1000, chunk_overlap=200)

        assert len(result) == 1
        assert result[0].page_content == "real content"


class TestSplitDocumentsWithDocumentId:

    def test_document_id_is_stamped_on_every_chunk(self):
        long_text = "word " * 500
        docs = _make_documents(content=long_text, source="report.pdf")

        result = split_documents(
            docs, chunk_size=200, chunk_overlap=20, document_id="doc-123"
        )

        assert len(result) > 1
        assert all(chunk.metadata["document_id"] == "doc-123" for chunk in result)

    def test_document_id_does_not_overwrite_existing_metadata(self):
        docs = _make_documents(content="short text", source="note.txt", page=2)

        result = split_documents(docs, document_id="doc-123")

        assert result[0].metadata["source"] == "note.txt"
        assert result[0].metadata["page"] == 2
        assert result[0].metadata["document_id"] == "doc-123"

    def test_no_document_id_key_when_not_provided(self):
        docs = _make_documents(content="short text", source="note.txt")

        result = split_documents(docs)

        assert "document_id" not in result[0].metadata

    def test_document_id_not_stamped_on_filtered_out_chunks(self):
        docs = [
            Document(page_content="real content", metadata={}),
            Document(page_content="   ", metadata={}),
        ]

        result = split_documents(docs, document_id="doc-123")

        assert len(result) == 1
        assert result[0].metadata["document_id"] == "doc-123"


class TestSplitDocumentsOperationFailure:

    def test_unexpected_error_is_wrapped_in_runtime_error(self):
        docs = _make_documents()

        with patch(
            f"{MODULE_PATH}.RecursiveCharacterTextSplitter.split_documents",
            side_effect=Exception("splitter exploded"),
        ):
            with pytest.raises(RuntimeError, match="Document splitting failed"):
                split_documents(docs)

    def test_runtime_error_is_chained_from_original_exception(self):
        docs = _make_documents()
        original_exc = Exception("splitter exploded")

        with patch(
            f"{MODULE_PATH}.RecursiveCharacterTextSplitter.split_documents",
            side_effect=original_exc,
        ):
            with pytest.raises(RuntimeError) as exc_info:
                split_documents(docs)

        assert exc_info.value.__cause__ is original_exc