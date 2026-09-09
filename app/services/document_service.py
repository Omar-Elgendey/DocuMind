import logging
import os
from typing import Any, Dict, List, Optional
from sqlalchemy.orm import Session

from app.db import repository
from app.db.models import Document
from app.rag.pipeline import RAGPipeline
from app.rag.vector_store import delete_documents_by_id

logger = logging.getLogger(__name__)


class DocumentService:
    """
    Orchestrates document ingestion, lifecycle management, and deletion by
    coordinating the MySQL repository, the RAG pipeline, and the filesystem.
    """

    def __init__(self, pipeline: RAGPipeline) -> None:
        """
        Initialize DocumentService with injected pipeline dependency.

        Args:
            pipeline: Injected RAGPipeline instance holding stateful retrieval and ingestion resources.
        """
        self.pipeline = pipeline

    def ingest_document(
        self,
        db: Session,
        document_id: str,
        filename: str,
        file_path: str,
    ) -> Dict[str, Any]:
        """
        Orchestrate the end-to-end ingestion flow for a document.

        State Flow:
        1. Create document record in MySQL with status 'PENDING'.
        2. Run RAG pipeline ingestion (loading, splitting, vector embedding).
        3. On Success: Update document status to 'COMPLETED' with chunk count.
        4. On Exception: Update document status to 'FAILED' with error details and re-raise.

        Args:
            db: Active SQLAlchemy session provided by the API layer.
            document_id: Unique UUID string representing the document.
            filename: Original file name uploaded by the user.
            file_path: Local disk path where the file is stored.

        Returns:
            Dict[str, Any]: Summary dictionary containing "document_id" and "chunks_count".

        Raises:
            ValueError: Propagated if any input parameter validation fails downstream.
            RuntimeError: Propagated if database operations or pipeline execution fails.
            Exception: Re-raises any unhandled pipeline/database exceptions after logging state.
        """
        logger.info("Registering document '%s' (ID: %s) as PENDING.", filename, document_id)

        repository.create_pending_document(
            db=db,
            document_id=document_id,
            original_filename=filename,
            file_path=file_path,
        )

        try:
            logger.info("Starting pipeline ingestion for document ID: %s", document_id)
            ingestion_result: Dict[str, Any] = self.pipeline.ingest_document(
                document_id=document_id,
                file_path=file_path,
            )

            chunks_count: int = ingestion_result.get("chunks_count", 0)

            repository.mark_document_completed(
                db=db,
                document_id=document_id,
                chunks_count=chunks_count,
            )
            logger.info(
                "Document ID %s successfully ingested (%d chunks) and marked COMPLETED.",
                document_id,
                chunks_count,
            )

            return {
                "document_id": document_id,
                "chunks_count": chunks_count,
            }

        except Exception as exc:
            error_message = str(exc) or "Unknown error occurred during document ingestion."
            logger.error(
                "Ingestion failed for document ID %s. Transitioning state to FAILED. Error: %s",
                document_id,
                error_message,
            )

            repository.mark_document_failed(
                db=db,
                document_id=document_id,
                error_message=error_message,
            )
            raise

    def list_documents(
        self,
        db: Session,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Document]:
        """
        Retrieve a paginated list of active (non-deleted) documents.

        Args:
            db: Active SQLAlchemy session provided by the API layer.
            status: Optional status filter (must be a valid DocumentStatus value).
            limit: Maximum number of records to return.
            offset: Number of records to skip.

        Returns:
            List[Document]: A list of active Document ORM objects.

        Raises:
            ValueError: Propagated if status/limit/offset validation fails.
            RuntimeError: Propagated if the database query fails.
        """
        return repository.list_documents(
            db=db,
            status=status,
            limit=limit,
            offset=offset,
        )

    def get_document(
        self,
        db: Session,
        document_id: str,
    ) -> Optional[Document]:
        """
        Retrieve a single active (non-deleted) document by its ID.

        Args:
            db: Active SQLAlchemy session provided by the API layer.
            document_id: Unique identifier of the document to retrieve.

        Returns:
            Optional[Document]: The Document ORM object if found, else None.

        Raises:
            ValueError: Propagated if document_id is empty or whitespace.
            RuntimeError: Propagated if the database query fails.
        """
        return repository.get_document(
            db=db,
            document_id=document_id,
        )

    def delete_document(
        self,
        db: Session,
        document_id: str,
    ) -> None:
        """
        Orchestrate the end-to-end deletion flow for a document.

        Deletion Flow:
        1. Fetch the document record to obtain its file_path.
        2. Delete all associated chunks from ChromaDB.
        3. Delete the file from the filesystem, if it still exists.
        4. Soft delete the document record in MySQL (final step).

        The order is intentional: MySQL is updated last so that, if any
        earlier step fails, the document remains visible and the deletion
        can safely be retried instead of silently losing track of data
        that still exists in Chroma or on disk.

        Args:
            db: Active SQLAlchemy session provided by the API layer.
            document_id: Unique identifier of the document to delete.

        Raises:
            ValueError: If no active document exists with the given ID.
            RuntimeError: Propagated if Chroma deletion or the final
                MySQL soft delete fails.
        """
        document = repository.get_document(db=db, document_id=document_id)

        if document is None:
            raise ValueError(f"No active document found with id={document_id}")

        logger.info("Deleting chunks from ChromaDB for document ID: %s", document_id)
        delete_documents_by_id(document_id=document_id)

        if os.path.exists(document.file_path):
            logger.info("Deleting file from filesystem: %s", document.file_path)
            os.remove(document.file_path)
        else:
            logger.warning(
                "File not found on filesystem for document ID %s at path '%s'; skipping.",
                document_id,
                document.file_path,
            )

        repository.soft_delete_document(db=db, document_id=document_id)

        logger.info("Document ID %s successfully deleted.", document_id)