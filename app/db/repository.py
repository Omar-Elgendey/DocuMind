import logging
from datetime import datetime, timezone
from typing import Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Document, DocumentStatus

logger = logging.getLogger(__name__)


def create_pending_document(
    db: Session,
    document_id: str,
    original_filename: str,
    file_path: str,
) -> Document:
    """
    Insert a new document row with status='pending'.

    Args:
        db: SQLAlchemy database session.
        document_id: Unique identifier string for the document.
        original_filename: The original name of the uploaded file.
        file_path: File system storage path for the uploaded file.

    Returns:
        The newly created Document ORM object.

    Raises:
        ValueError: If document_id, original_filename, or file_path is empty or whitespace.
        RuntimeError: If a database error occurs during commit.
    """
    if not document_id or not document_id.strip():
        raise ValueError("document_id must not be empty.")
    if not original_filename or not original_filename.strip():
        raise ValueError("original_filename must not be empty.")
    if not file_path or not file_path.strip():
        raise ValueError("file_path must not be empty.")

    document = Document(
        id=document_id,
        original_filename=original_filename,
        file_path=file_path,
        status=DocumentStatus.PENDING.value,
    )

    try:
        db.add(document)
        db.commit()
        db.refresh(document)
        logger.info("Created pending document record for id=%s", document_id)
        return document
    except Exception as exc:
        db.rollback()
        logger.exception("Database error creating pending document for id=%s", document_id)
        raise RuntimeError(
            f"Failed to create document record for id={document_id}."
        ) from exc


def mark_document_completed(
    db: Session,
    document_id: str,
    chunks_count: int,
) -> Document:
    """
    Update document status to 'completed' and record its total chunk count.

    Args:
        db: SQLAlchemy database session.
        document_id: Unique identifier of the document to update.
        chunks_count: Non-negative integer representing total text chunks created.

    Returns:
        The updated Document ORM object.

    Raises:
        ValueError: If no document matches document_id, or if chunks_count is invalid.
        RuntimeError: If a database error occurs during commit.
    """
    if not document_id or not document_id.strip():
        raise ValueError("document_id must not be empty.")

    if isinstance(chunks_count, bool) or not isinstance(chunks_count, int) or chunks_count < 0:
        raise ValueError("chunks_count must be a non-negative integer.")

    try:
        stmt = select(Document).where(Document.id == document_id)
        document = db.scalars(stmt).first()
    except Exception as exc:
        logger.exception("Database query error retrieving document for id=%s", document_id)
        raise RuntimeError(
            f"Failed to retrieve document for id={document_id}."
        ) from exc

    if not document:
        raise ValueError(f"No document found with id={document_id}")

    document.status = DocumentStatus.COMPLETED.value
    document.chunks_count = chunks_count

    try:
        db.commit()
        db.refresh(document)
        logger.info("Marked document id=%s as completed (%d chunks)", document_id, chunks_count)
        return document
    except Exception as exc:
        db.rollback()
        logger.exception("Database error updating document id=%s to completed", document_id)
        raise RuntimeError(
            f"Failed to update document status to completed for id={document_id}."
        ) from exc


def mark_document_failed(
    db: Session,
    document_id: str,
    error_message: str,
) -> Document:
    """
    Update document status to 'failed' and record the error reason.

    Args:
        db: SQLAlchemy database session.
        document_id: Unique identifier of the document to update.
        error_message: Non-empty description of the error encountered.

    Returns:
        The updated Document ORM object.

    Raises:
        ValueError: If no document matches document_id, or if error_message is empty.
        RuntimeError: If a database error occurs during commit.
    """
    if not document_id or not document_id.strip():
        raise ValueError("document_id must not be empty.")
    if not error_message or not error_message.strip():
        raise ValueError("error_message must not be empty.")

    try:
        stmt = select(Document).where(Document.id == document_id)
        document = db.scalars(stmt).first()
    except Exception as exc:
        logger.exception("Database query error retrieving document for id=%s", document_id)
        raise RuntimeError(
            f"Failed to retrieve document for id={document_id}."
        ) from exc

    if not document:
        raise ValueError(f"No document found with id={document_id}")

    document.status = DocumentStatus.FAILED.value
    document.error_message = error_message

    try:
        db.commit()
        db.refresh(document)
        logger.info("Marked document id=%s as failed", document_id)
        return document
    except Exception as exc:
        db.rollback()
        logger.exception("Database error updating document id=%s to failed", document_id)
        raise RuntimeError(
            f"Failed to update document status to failed for id={document_id}."
        ) from exc


def soft_delete_document(
    db: Session,
    document_id: str,
) -> Document:
    """
    Perform a soft delete on a document row by populating deleted_at timestamp.

    Args:
        db: SQLAlchemy database session.
        document_id: Unique identifier of the document to soft-delete.

    Returns:
        The updated Document ORM object.

    Raises:
        ValueError: If document does not exist or has already been soft-deleted.
        RuntimeError: If a database error occurs during commit.
    """
    if not document_id or not document_id.strip():
        raise ValueError("document_id must not be empty.")

    try:
        stmt = select(Document).where(Document.id == document_id)
        document = db.scalars(stmt).first()
    except Exception as exc:
        logger.exception("Database query error retrieving document for id=%s", document_id)
        raise RuntimeError(
            f"Failed to retrieve document for id={document_id}."
        ) from exc

    if not document:
        raise ValueError(f"No document found with id={document_id}")

    if document.deleted_at is not None:
        raise ValueError(f"Document {document_id} is already deleted")

    document.deleted_at = datetime.now(timezone.utc)

    try:
        db.commit()
        db.refresh(document)
        logger.info("Soft-deleted document id=%s", document_id)
        return document
    except Exception as exc:
        db.rollback()
        logger.exception("Database error soft-deleting document id=%s", document_id)
        raise RuntimeError(
            f"Failed to soft-delete document for id={document_id}."
        ) from exc


def get_document(
    db: Session,
    document_id: str,
) -> Document | None:
    """
    Retrieve an active (non-deleted) document by its unique ID.

    Args:
        db: SQLAlchemy database session.
        document_id: Unique identifier of the document to retrieve.

    Returns:
        The Document ORM object if found and active, otherwise None.

    Raises:
        ValueError: If document_id is empty or whitespace.
        RuntimeError: If a database query or connection error occurs.
    """
    if not document_id or not document_id.strip():
        raise ValueError("document_id must not be empty.")

    try:
        stmt = select(Document).where(
            Document.id == document_id,
            Document.deleted_at.is_(None),
        )
        return db.scalars(stmt).first()
    except Exception as exc:
        logger.exception("Database error executing get_document for id=%s", document_id)
        raise RuntimeError(
            f"Failed to retrieve document for id={document_id}."
        ) from exc


def list_documents(
    db: Session,
    status: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[Document]:
    """
    List active (non-deleted) documents with optional status filtering and pagination.

    Args:
        db: SQLAlchemy database session.
        status: Optional status string to filter by (must be a valid DocumentStatus value).
        limit: Maximum number of records to return (default 50, must be > 0).
        offset: Number of records to skip (default 0, must be >= 0).

    Returns:
        A list of active Document ORM objects, ordered by creation date descending.

    Raises:
        ValueError: If status is invalid, or if limit/offset parameters are out of range.
        RuntimeError: If a database query or connection error occurs.
    """
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError("limit must be a positive integer.")

    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise ValueError("offset must be a non-negative integer.")

    if status is not None:
        valid_statuses = {s.value for s in DocumentStatus}
        if status not in valid_statuses:
            raise ValueError(
                f"Invalid status '{status}'. Must be one of: {', '.join(sorted(valid_statuses))}"
            )

    try:
        stmt = select(Document).where(Document.deleted_at.is_(None))

        if status is not None:
            stmt = stmt.where(Document.status == status)

        stmt = stmt.order_by(Document.created_at.desc()).offset(offset).limit(limit)

        results: Sequence[Document] = db.scalars(stmt).all()
        return list(results)
    except Exception as exc:
        logger.exception("Database error executing list_documents")
        raise RuntimeError("Failed to list documents from database.") from exc