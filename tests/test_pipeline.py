from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from app.rag.pipeline import RAGPipeline


MODULE_PATH = "app.rag.pipeline"


def _make_pipeline(vector_store=None):
    retriever = MagicMock()
    generator = MagicMock()
    return RAGPipeline(retriever=retriever, generator=generator, vector_store=vector_store)


def _fake_documents(n=1, source="doc.pdf"):
    return [
        Document(page_content=f"content {i}", metadata={"source": source})
        for i in range(n)
    ]


class TestIngestDocumentInvalidInput:

    def test_empty_file_path_raises_value_error(self):
        pipeline = _make_pipeline()

        with pytest.raises(ValueError, match="file_path must not be empty"):
            pipeline.ingest_document("")

    def test_whitespace_only_file_path_raises_value_error(self):
        pipeline = _make_pipeline()

        with pytest.raises(ValueError, match="file_path must not be empty"):
            pipeline.ingest_document("   ")


class TestIngestDocumentHappyPath:

    def test_generates_document_id_when_not_provided(self):
        pipeline = _make_pipeline(vector_store=MagicMock())
        raw_docs = _fake_documents(2)
        chunks = _fake_documents(3)

        with patch(f"{MODULE_PATH}.UniversalLoader") as mock_loader_class, \
             patch(f"{MODULE_PATH}.split_documents", return_value=chunks) as mock_split, \
             patch(f"{MODULE_PATH}.add_documents", return_value=["id_0", "id_1", "id_2"]) as mock_add:
            mock_loader_class.return_value.load.return_value = raw_docs

            result = pipeline.ingest_document("report.pdf")

        assert result["document_id"]
        assert result["chunks_count"] == 3
        mock_split.assert_called_once()
        assert mock_split.call_args.kwargs["document_id"] == result["document_id"]
        mock_add.assert_called_once()

    def test_uses_provided_document_id(self):
        pipeline = _make_pipeline(vector_store=MagicMock())
        raw_docs = _fake_documents(1)
        chunks = _fake_documents(1)

        with patch(f"{MODULE_PATH}.UniversalLoader") as mock_loader_class, \
             patch(f"{MODULE_PATH}.split_documents", return_value=chunks) as mock_split, \
             patch(f"{MODULE_PATH}.add_documents", return_value=["doc-42_0"]) as mock_add:
            mock_loader_class.return_value.load.return_value = raw_docs

            result = pipeline.ingest_document("report.pdf", document_id="doc-42")

        assert result["document_id"] == "doc-42"
        assert result["chunks_count"] == 1
        mock_split.assert_called_once_with(
            raw_docs, chunk_size=1000, chunk_overlap=200, document_id="doc-42"
        )
        mock_add.assert_called_once_with(chunks, vector_store=pipeline._vector_store)

    def test_passes_through_chunk_size_and_overlap(self):
        pipeline = _make_pipeline(vector_store=MagicMock())
        raw_docs = _fake_documents(1)
        chunks = _fake_documents(1)

        with patch(f"{MODULE_PATH}.UniversalLoader") as mock_loader_class, \
             patch(f"{MODULE_PATH}.split_documents", return_value=chunks) as mock_split, \
             patch(f"{MODULE_PATH}.add_documents", return_value=["doc-42_0"]):
            mock_loader_class.return_value.load.return_value = raw_docs

            pipeline.ingest_document(
                "report.pdf", document_id="doc-42", chunk_size=500, chunk_overlap=50
            )

        mock_split.assert_called_once_with(
            raw_docs, chunk_size=500, chunk_overlap=50, document_id="doc-42"
        )

    def test_lazily_creates_vector_store_when_not_injected(self):
        pipeline = _make_pipeline(vector_store=None)
        raw_docs = _fake_documents(1)
        chunks = _fake_documents(1)
        fake_store = MagicMock()

        with patch(f"{MODULE_PATH}.UniversalLoader") as mock_loader_class, \
             patch(f"{MODULE_PATH}.split_documents", return_value=chunks), \
             patch(f"{MODULE_PATH}.add_documents", return_value=["doc-42_0"]) as mock_add, \
             patch(f"{MODULE_PATH}.get_vector_store", return_value=fake_store) as mock_get_store:
            mock_loader_class.return_value.load.return_value = raw_docs

            pipeline.ingest_document("report.pdf", document_id="doc-42")

        mock_get_store.assert_called_once()
        mock_add.assert_called_once_with(chunks, vector_store=fake_store)

    def test_reuses_injected_vector_store_across_calls(self):
        injected_store = MagicMock()
        pipeline = _make_pipeline(vector_store=injected_store)
        raw_docs = _fake_documents(1)
        chunks = _fake_documents(1)

        with patch(f"{MODULE_PATH}.UniversalLoader") as mock_loader_class, \
             patch(f"{MODULE_PATH}.split_documents", return_value=chunks), \
             patch(f"{MODULE_PATH}.add_documents", return_value=["id_0"]), \
             patch(f"{MODULE_PATH}.get_vector_store") as mock_get_store:
            mock_loader_class.return_value.load.return_value = raw_docs

            pipeline.ingest_document("report.pdf", document_id="doc-1")
            pipeline.ingest_document("report.pdf", document_id="doc-2")

        mock_get_store.assert_not_called()


class TestIngestDocumentEdgeCases:

    def test_no_chunks_produced_returns_zero_count_without_storing(self):
        pipeline = _make_pipeline(vector_store=MagicMock())
        raw_docs = _fake_documents(1)

        with patch(f"{MODULE_PATH}.UniversalLoader") as mock_loader_class, \
             patch(f"{MODULE_PATH}.split_documents", return_value=[]), \
             patch(f"{MODULE_PATH}.add_documents") as mock_add:
            mock_loader_class.return_value.load.return_value = raw_docs

            result = pipeline.ingest_document("empty.txt", document_id="doc-99")

        assert result == {"document_id": "doc-99", "chunks_count": 0}
        mock_add.assert_not_called()


class TestIngestDocumentOperationFailure:

    def test_loader_error_propagates(self):
        pipeline = _make_pipeline(vector_store=MagicMock())

        with patch(f"{MODULE_PATH}.UniversalLoader") as mock_loader_class:
            mock_loader_class.return_value.load.side_effect = FileNotFoundError(
                "missing file"
            )

            with pytest.raises(FileNotFoundError):
                pipeline.ingest_document("missing.pdf")

    def test_splitter_error_propagates(self):
        pipeline = _make_pipeline(vector_store=MagicMock())
        raw_docs = _fake_documents(1)

        with patch(f"{MODULE_PATH}.UniversalLoader") as mock_loader_class, \
             patch(
                 f"{MODULE_PATH}.split_documents",
                 side_effect=ValueError("chunk_size must be greater than 0"),
             ):
            mock_loader_class.return_value.load.return_value = raw_docs

            with pytest.raises(ValueError):
                pipeline.ingest_document("report.pdf", chunk_size=0)

    def test_unexpected_storage_error_is_wrapped_in_runtime_error(self):
        pipeline = _make_pipeline(vector_store=MagicMock())
        raw_docs = _fake_documents(1)
        chunks = _fake_documents(1)

        with patch(f"{MODULE_PATH}.UniversalLoader") as mock_loader_class, \
             patch(f"{MODULE_PATH}.split_documents", return_value=chunks), \
             patch(f"{MODULE_PATH}.add_documents", side_effect=Exception("chroma down")):
            mock_loader_class.return_value.load.return_value = raw_docs

            with pytest.raises(RuntimeError, match="Failed to store document chunks"):
                pipeline.ingest_document("report.pdf")

    def test_runtime_error_from_storage_is_chained(self):
        pipeline = _make_pipeline(vector_store=MagicMock())
        raw_docs = _fake_documents(1)
        chunks = _fake_documents(1)
        original_exc = Exception("chroma down")

        with patch(f"{MODULE_PATH}.UniversalLoader") as mock_loader_class, \
             patch(f"{MODULE_PATH}.split_documents", return_value=chunks), \
             patch(f"{MODULE_PATH}.add_documents", side_effect=original_exc):
            mock_loader_class.return_value.load.return_value = raw_docs

            with pytest.raises(RuntimeError) as exc_info:
                pipeline.ingest_document("report.pdf")

        assert exc_info.value.__cause__ is original_exc