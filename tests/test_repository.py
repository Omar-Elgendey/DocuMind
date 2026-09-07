from datetime import datetime, timezone
from unittest.mock import MagicMock
import pytest

from app.db.models import Document, DocumentStatus
from app.db.repository import (
    create_pending_document,
    get_document,
    list_documents,
    mark_document_completed,
    mark_document_failed,
    soft_delete_document,
)


@pytest.fixture
def mock_db():
    """Fixture providing a mocked SQLAlchemy Session."""
    return MagicMock()


# ============================================================================
# 1. Tests for create_pending_document
# ============================================================================

def test_create_pending_document_success(mock_db):
    doc_id = "test-uuid-1234"
    filename = "report.pdf"
    file_path = "/uploads/report.pdf"

    result = create_pending_document(mock_db, doc_id, filename, file_path)

    assert isinstance(result, Document)
    assert result.id == doc_id
    assert result.original_filename == filename
    assert result.file_path == file_path
    assert result.status == DocumentStatus.PENDING.value

    mock_db.add.assert_called_once_with(result)
    mock_db.commit.assert_called_once()
    mock_db.refresh.assert_called_once_with(result)


@pytest.mark.parametrize("doc_id, filename, path", [
    ("", "valid.pdf", "/path"),
    ("   ", "valid.pdf", "/path"),
    ("id", "", "/path"),
    ("id", "   ", "/path"),
    ("id", "valid.pdf", ""),
    ("id", "valid.pdf", "   "),
])
def test_create_pending_document_validation_errors(mock_db, doc_id, filename, path):
    with pytest.raises(ValueError):
        create_pending_document(mock_db, doc_id, filename, path)


def test_create_pending_document_db_error(mock_db):
    mock_db.commit.side_effect = Exception("DB Connection Lost")

    with pytest.raises(RuntimeError) as exc_info:
        create_pending_document(mock_db, "id-123", "doc.pdf", "/path")

    assert "Failed to create document record" in str(exc_info.value)
    mock_db.rollback.assert_called_once()


# ============================================================================
# 2. Tests for mark_document_completed
# ============================================================================

def test_mark_document_completed_success(mock_db):
    mock_doc = Document(id="doc-1", status=DocumentStatus.PENDING.value)
    mock_db.scalars.return_value.first.return_value = mock_doc

    result = mark_document_completed(mock_db, "doc-1", chunks_count=10)

    assert result.status == DocumentStatus.COMPLETED.value
    assert result.chunks_count == 10
    mock_db.commit.assert_called_once()
    mock_db.refresh.assert_called_once_with(mock_doc)


def test_mark_document_completed_not_found(mock_db):
    mock_db.scalars.return_value.first.return_value = None

    with pytest.raises(ValueError, match="No document found with id=doc-1"):
        mark_document_completed(mock_db, "doc-1", chunks_count=5)


@pytest.mark.parametrize("invalid_chunks", [-1, -10, "5", True, False, 3.14])
def test_mark_document_completed_invalid_chunks(mock_db, invalid_chunks):
    with pytest.raises(ValueError, match="chunks_count must be a non-negative integer"):
        mark_document_completed(mock_db, "doc-1", chunks_count=invalid_chunks)


def test_mark_document_completed_db_error(mock_db):
    mock_doc = Document(id="doc-1", status=DocumentStatus.PENDING.value)
    mock_db.scalars.return_value.first.return_value = mock_doc
    mock_db.commit.side_effect = Exception("Commit Failed")

    with pytest.raises(RuntimeError) as exc_info:
        mark_document_completed(mock_db, "doc-1", chunks_count=5)

    assert "Failed to update document status to completed" in str(exc_info.value)
    mock_db.rollback.assert_called_once()


# ============================================================================
# 3. Tests for mark_document_failed
# ============================================================================

def test_mark_document_failed_success(mock_db):
    mock_doc = Document(id="doc-1", status=DocumentStatus.PENDING.value)
    mock_db.scalars.return_value.first.return_value = mock_doc

    result = mark_document_failed(mock_db, "doc-1", error_message="Parsing timeout")

    assert result.status == DocumentStatus.FAILED.value
    assert result.error_message == "Parsing timeout"
    mock_db.commit.assert_called_once()
    mock_db.refresh.assert_called_once_with(mock_doc)


@pytest.mark.parametrize("empty_err", ["", "   "])
def test_mark_document_failed_invalid_error_message(mock_db, empty_err):
    with pytest.raises(ValueError, match="error_message must not be empty"):
        mark_document_failed(mock_db, "doc-1", error_message=empty_err)


# ============================================================================
# 4. Tests for soft_delete_document
# ============================================================================

def test_soft_delete_document_success(mock_db):
    mock_doc = Document(id="doc-1", deleted_at=None)
    mock_db.scalars.return_value.first.return_value = mock_doc

    result = soft_delete_document(mock_db, "doc-1")

    assert result.deleted_at is not None
    assert isinstance(result.deleted_at, datetime)
    mock_db.commit.assert_called_once()
    mock_db.refresh.assert_called_once_with(mock_doc)


def test_soft_delete_document_already_deleted(mock_db):
    mock_doc = Document(id="doc-1", deleted_at=datetime.now(timezone.utc))
    mock_db.scalars.return_value.first.return_value = mock_doc

    with pytest.raises(ValueError, match="is already deleted"):
        soft_delete_document(mock_db, "doc-1")


def test_soft_delete_document_not_found(mock_db):
    mock_db.scalars.return_value.first.return_value = None

    with pytest.raises(ValueError, match="No document found with id=doc-1"):
        soft_delete_document(mock_db, "doc-1")


# ============================================================================
# 5. Tests for get_document
# ============================================================================

def test_get_document_success(mock_db):
    mock_doc = Document(id="doc-1", deleted_at=None)
    mock_db.scalars.return_value.first.return_value = mock_doc

    result = get_document(mock_db, "doc-1")

    assert result == mock_doc


def test_get_document_not_found(mock_db):
    mock_db.scalars.return_value.first.return_value = None

    result = get_document(mock_db, "non-existent")

    assert result is None


@pytest.mark.parametrize("empty_id", ["", "   "])
def test_get_document_invalid_id(mock_db, empty_id):
    with pytest.raises(ValueError, match="document_id must not be empty"):
        get_document(mock_db, empty_id)


def test_get_document_db_error(mock_db):
    mock_db.scalars.side_effect = Exception("Database error")

    with pytest.raises(RuntimeError) as exc_info:
        get_document(mock_db, "doc-1")

    assert "Failed to retrieve document" in str(exc_info.value)


# ============================================================================
# 6. Tests for list_documents
# ============================================================================

def test_list_documents_success(mock_db):
    mock_docs = [
        Document(id="doc-1"),
        Document(id="doc-2"),
    ]
    mock_db.scalars.return_value.all.return_value = mock_docs

    results = list_documents(mock_db, limit=10, offset=0)

    assert len(results) == 2
    assert results == mock_docs


def test_list_documents_with_valid_status(mock_db):
    mock_db.scalars.return_value.all.return_value = []

    results = list_documents(mock_db, status=DocumentStatus.COMPLETED.value)

    assert results == []


def test_list_documents_invalid_status(mock_db):
    with pytest.raises(ValueError, match="Invalid status 'invalid_status'"):
        list_documents(mock_db, status="invalid_status")


@pytest.mark.parametrize("limit, offset", [
    (0, 0),
    (-5, 0),
    (10, -1),
    (True, 0),
    (10, False),
])
def test_list_documents_invalid_pagination(mock_db, limit, offset):
    with pytest.raises(ValueError):
        list_documents(mock_db, limit=limit, offset=offset)


def test_list_documents_db_error(mock_db):
    mock_db.scalars.side_effect = Exception("Query Execution Error")

    with pytest.raises(RuntimeError, match="Failed to list documents from database"):
        list_documents(mock_db)