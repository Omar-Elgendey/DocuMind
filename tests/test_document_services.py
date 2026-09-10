import pytest
from unittest.mock import MagicMock, patch
from sqlalchemy.orm import Session

from app.rag.pipeline import RAGPipeline
from app.services.document_service import DocumentService


MODULE_PATH = "app.services.document_service"


@pytest.fixture
def mock_pipeline():
    return MagicMock(spec=RAGPipeline)


@pytest.fixture
def mock_db():
    return MagicMock(spec=Session)


@pytest.fixture
def service(mock_pipeline):
    return DocumentService(pipeline=mock_pipeline)


class TestIngestDocumentSuccess:

    @patch(f"{MODULE_PATH}.repository")
    def test_ingest_document_success(
        self, mock_repo, service, mock_pipeline, mock_db
    ):
        document_id = "doc_123"
        filename = "paper.pdf"
        file_path = "data/uploads/paper.pdf"
        expected_chunks = 8

        mock_pipeline.ingest_document.return_value = {
            "document_id": document_id,
            "chunks_count": expected_chunks,
        }

        result = service.ingest_document(
            db=mock_db,
            document_id=document_id,
            filename=filename,
            file_path=file_path,
        )

        assert result == {
            "document_id": document_id,
            "chunks_count": expected_chunks,
        }

        mock_repo.create_pending_document.assert_called_once_with(
            db=mock_db,
            document_id=document_id,
            original_filename=filename,
            file_path=file_path,
        )
        mock_pipeline.ingest_document.assert_called_once_with(
            document_id=document_id,
            file_path=file_path,
        )
        mock_repo.mark_document_completed.assert_called_once_with(
            db=mock_db,
            document_id=document_id,
            chunks_count=expected_chunks,
        )
        mock_repo.mark_document_failed.assert_not_called()


class TestIngestDocumentPipelineFailure:

    @patch(f"{MODULE_PATH}.os.remove")
    @patch(f"{MODULE_PATH}.os.path.exists")
    @patch(f"{MODULE_PATH}.repository")
    def test_pipeline_failure_marks_failed_and_reraises(
        self, mock_repo, mock_exists, mock_remove, service, mock_pipeline, mock_db
    ):
        document_id = "doc_error_456"
        filename = "corrupt.pdf"
        file_path = "data/uploads/corrupt.pdf"
        error_text = "UniversalLoader failed to read document."

        mock_pipeline.ingest_document.side_effect = RuntimeError(error_text)
        mock_exists.return_value = True

        with pytest.raises(RuntimeError, match=error_text):
            service.ingest_document(
                db=mock_db,
                document_id=document_id,
                filename=filename,
                file_path=file_path,
            )

        mock_repo.create_pending_document.assert_called_once_with(
            db=mock_db,
            document_id=document_id,
            original_filename=filename,
            file_path=file_path,
        )
        mock_repo.mark_document_failed.assert_called_once_with(
            db=mock_db,
            document_id=document_id,
            error_message=error_text,
        )
        mock_repo.mark_document_completed.assert_not_called()
        mock_exists.assert_called_once_with(file_path)
        mock_remove.assert_called_once_with(file_path)


class TestIngestDocumentPendingCreationFailure:

    @patch(f"{MODULE_PATH}.repository")
    def test_pending_db_failure_prevents_pipeline_run(
        self, mock_repo, service, mock_pipeline, mock_db
    ):
        mock_repo.create_pending_document.side_effect = RuntimeError(
            "DB Connection lost"
        )

        with pytest.raises(RuntimeError, match="DB Connection lost"):
            service.ingest_document(
                db=mock_db,
                document_id="doc_789",
                filename="file.docx",
                file_path="data/uploads/file.docx",
            )

        mock_pipeline.ingest_document.assert_not_called()
        mock_repo.mark_document_failed.assert_not_called()


class TestIngestDocumentMarkCompletedFailure:

    @patch(f"{MODULE_PATH}.os.remove")
    @patch(f"{MODULE_PATH}.os.path.exists")
    @patch(f"{MODULE_PATH}.repository")
    def test_mark_completed_failure_triggers_mark_failed_and_reraises(
        self, mock_repo, mock_exists, mock_remove, service, mock_pipeline, mock_db
    ):
        document_id = "doc_999"
        filename = "report.pdf"
        file_path = "data/uploads/report.pdf"
        expected_chunks = 5

        mock_pipeline.ingest_document.return_value = {
            "document_id": document_id,
            "chunks_count": expected_chunks,
        }
        completion_error = RuntimeError("Failed to update document status to completed")
        mock_repo.mark_document_completed.side_effect = completion_error
        mock_exists.return_value = True

        with pytest.raises(RuntimeError, match="Failed to update document status to completed"):
            service.ingest_document(
                db=mock_db,
                document_id=document_id,
                filename=filename,
                file_path=file_path,
            )

        mock_repo.mark_document_completed.assert_called_once_with(
            db=mock_db,
            document_id=document_id,
            chunks_count=expected_chunks,
        )
        mock_repo.mark_document_failed.assert_called_once_with(
            db=mock_db,
            document_id=document_id,
            error_message=str(completion_error),
        )
        mock_exists.assert_called_once_with(file_path)
        mock_remove.assert_called_once_with(file_path)


class TestListDocumentsSuccess:

    @patch(f"{MODULE_PATH}.repository")
    def test_list_documents_returns_repository_result(
        self, mock_repo, service, mock_db
    ):
        expected_documents = [MagicMock(), MagicMock()]
        mock_repo.list_documents.return_value = expected_documents

        result = service.list_documents(db=mock_db)

        assert result == expected_documents
        mock_repo.list_documents.assert_called_once_with(
            db=mock_db,
            status=None,
            limit=50,
            offset=0,
        )

    @patch(f"{MODULE_PATH}.repository")
    def test_list_documents_passes_through_status_and_pagination(
        self, mock_repo, service, mock_db
    ):
        mock_repo.list_documents.return_value = []

        service.list_documents(
            db=mock_db,
            status="completed",
            limit=10,
            offset=20,
        )

        mock_repo.list_documents.assert_called_once_with(
            db=mock_db,
            status="completed",
            limit=10,
            offset=20,
        )

    @patch(f"{MODULE_PATH}.repository")
    def test_list_documents_empty_result(
        self, mock_repo, service, mock_db
    ):
        mock_repo.list_documents.return_value = []

        result = service.list_documents(db=mock_db)

        assert result == []


class TestListDocumentsFailure:

    @patch(f"{MODULE_PATH}.repository")
    def test_list_documents_invalid_status_propagates_value_error(
        self, mock_repo, service, mock_db
    ):
        mock_repo.list_documents.side_effect = ValueError(
            "Invalid status 'bogus'."
        )

        with pytest.raises(ValueError, match="Invalid status 'bogus'"):
            service.list_documents(db=mock_db, status="bogus")

    @patch(f"{MODULE_PATH}.repository")
    def test_list_documents_db_error_propagates_runtime_error(
        self, mock_repo, service, mock_db
    ):
        mock_repo.list_documents.side_effect = RuntimeError(
            "Failed to list documents from database."
        )

        with pytest.raises(RuntimeError, match="Failed to list documents from database."):
            service.list_documents(db=mock_db)


class TestGetDocumentSuccess:

    @patch(f"{MODULE_PATH}.repository")
    def test_get_document_returns_repository_result(
        self, mock_repo, service, mock_db
    ):
        expected_document = MagicMock()
        mock_repo.get_document.return_value = expected_document

        result = service.get_document(db=mock_db, document_id="doc-1")

        assert result == expected_document
        mock_repo.get_document.assert_called_once_with(
            db=mock_db,
            document_id="doc-1",
        )

    @patch(f"{MODULE_PATH}.repository")
    def test_get_document_not_found_returns_none(
        self, mock_repo, service, mock_db
    ):
        mock_repo.get_document.return_value = None

        result = service.get_document(db=mock_db, document_id="missing-id")

        assert result is None


class TestGetDocumentFailure:

    @patch(f"{MODULE_PATH}.repository")
    def test_get_document_empty_id_propagates_value_error(
        self, mock_repo, service, mock_db
    ):
        mock_repo.get_document.side_effect = ValueError(
            "document_id must not be empty."
        )

        with pytest.raises(ValueError, match="document_id must not be empty."):
            service.get_document(db=mock_db, document_id="")

    @patch(f"{MODULE_PATH}.repository")
    def test_get_document_db_error_propagates_runtime_error(
        self, mock_repo, service, mock_db
    ):
        mock_repo.get_document.side_effect = RuntimeError(
            "Failed to retrieve document for id=doc-1."
        )

        with pytest.raises(RuntimeError, match="Failed to retrieve document for id=doc-1."):
            service.get_document(db=mock_db, document_id="doc-1")


class TestDeleteDocumentSuccess:

    @patch(f"{MODULE_PATH}.os.remove")
    @patch(f"{MODULE_PATH}.os.path.exists")
    @patch(f"{MODULE_PATH}.delete_documents_by_id")
    @patch(f"{MODULE_PATH}.repository")
    def test_delete_document_success_with_existing_file(
        self, mock_repo, mock_delete_chroma, mock_exists, mock_remove, service, mock_db
    ):
        doc_id = "doc_del_123"
        file_path = "data/uploads/paper.pdf"

        mock_doc = MagicMock()
        mock_doc.file_path = file_path
        mock_repo.get_document.return_value = mock_doc
        mock_exists.return_value = True

        service.delete_document(db=mock_db, document_id=doc_id)

        mock_repo.get_document.assert_called_once_with(db=mock_db, document_id=doc_id)
        mock_delete_chroma.assert_called_once_with(document_id=doc_id)
        mock_exists.assert_called_once_with(file_path)
        mock_remove.assert_called_once_with(file_path)
        mock_repo.soft_delete_document.assert_called_once_with(db=mock_db, document_id=doc_id)

    @patch(f"{MODULE_PATH}.os.remove")
    @patch(f"{MODULE_PATH}.os.path.exists")
    @patch(f"{MODULE_PATH}.delete_documents_by_id")
    @patch(f"{MODULE_PATH}.repository")
    def test_delete_document_success_when_file_does_not_exist_on_disk(
        self, mock_repo, mock_delete_chroma, mock_exists, mock_remove, service, mock_db
    ):
        doc_id = "doc_del_456"
        file_path = "data/uploads/missing.pdf"

        mock_doc = MagicMock()
        mock_doc.file_path = file_path
        mock_repo.get_document.return_value = mock_doc
        mock_exists.return_value = False

        service.delete_document(db=mock_db, document_id=doc_id)

        mock_repo.get_document.assert_called_once_with(db=mock_db, document_id=doc_id)
        mock_delete_chroma.assert_called_once_with(document_id=doc_id)
        mock_exists.assert_called_once_with(file_path)
        mock_remove.assert_not_called()
        mock_repo.soft_delete_document.assert_called_once_with(db=mock_db, document_id=doc_id)


class TestDeleteDocumentFailure:

    @patch(f"{MODULE_PATH}.repository")
    def test_delete_document_raises_value_error_if_not_found(
        self, mock_repo, service, mock_db
    ):
        doc_id = "missing_doc"
        mock_repo.get_document.return_value = None

        with pytest.raises(ValueError, match=f"No active document found with id={doc_id}"):
            service.delete_document(db=mock_db, document_id=doc_id)

        mock_repo.get_document.assert_called_once_with(db=mock_db, document_id=doc_id)
        mock_repo.soft_delete_document.assert_not_called()

    @patch(f"{MODULE_PATH}.delete_documents_by_id")
    @patch(f"{MODULE_PATH}.repository")
    def test_delete_document_chroma_failure_propagates_and_aborts_db_deletion(
        self, mock_repo, mock_delete_chroma, service, mock_db
    ):
        doc_id = "doc_chroma_err"
        mock_doc = MagicMock()
        mock_doc.file_path = "data/uploads/file.pdf"
        mock_repo.get_document.return_value = mock_doc

        mock_delete_chroma.side_effect = RuntimeError("Chroma connection error")

        with pytest.raises(RuntimeError, match="Chroma connection error"):
            service.delete_document(db=mock_db, document_id=doc_id)

        mock_delete_chroma.assert_called_once_with(document_id=doc_id)
        mock_repo.soft_delete_document.assert_not_called()

    @patch(f"{MODULE_PATH}.os.remove")
    @patch(f"{MODULE_PATH}.os.path.exists")
    @patch(f"{MODULE_PATH}.delete_documents_by_id")
    @patch(f"{MODULE_PATH}.repository")
    def test_delete_document_file_removal_failure_propagates_and_aborts_db_deletion(
        self, mock_repo, mock_delete_chroma, mock_exists, mock_remove, service, mock_db
    ):
        doc_id = "doc_file_err"
        file_path = "data/uploads/locked.pdf"

        mock_doc = MagicMock()
        mock_doc.file_path = file_path
        mock_repo.get_document.return_value = mock_doc
        mock_exists.return_value = True
        mock_remove.side_effect = OSError("Permission denied")

        with pytest.raises(OSError, match="Permission denied"):
            service.delete_document(db=mock_db, document_id=doc_id)

        mock_delete_chroma.assert_called_once_with(document_id=doc_id)
        mock_remove.assert_called_once_with(file_path)
        mock_repo.soft_delete_document.assert_not_called()

    @patch(f"{MODULE_PATH}.os.remove")
    @patch(f"{MODULE_PATH}.os.path.exists")
    @patch(f"{MODULE_PATH}.delete_documents_by_id")
    @patch(f"{MODULE_PATH}.repository")
    def test_delete_document_final_soft_delete_failure_propagates(
        self, mock_repo, mock_delete_chroma, mock_exists, mock_remove, service, mock_db
    ):
        doc_id = "doc_softdel_err"
        file_path = "data/uploads/report.pdf"

        mock_doc = MagicMock()
        mock_doc.file_path = file_path
        mock_repo.get_document.return_value = mock_doc
        mock_exists.return_value = True
        mock_repo.soft_delete_document.side_effect = RuntimeError(
            "Failed to soft-delete document for id=doc_softdel_err."
        )

        with pytest.raises(RuntimeError, match="Failed to soft-delete document"):
            service.delete_document(db=mock_db, document_id=doc_id)

        mock_delete_chroma.assert_called_once_with(document_id=doc_id)
        mock_remove.assert_called_once_with(file_path)
        mock_repo.soft_delete_document.assert_called_once_with(db=mock_db, document_id=doc_id)