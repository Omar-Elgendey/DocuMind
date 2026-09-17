import logging
import os
from typing import Any, Dict, List, Optional
from sqlalchemy.orm import Session

from app.db import repository
from app.db.models import Document, DocumentStatus
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
        session_id: str,
    ) -> Dict[str, Any]:
        """
        Orchestrate the end-to-end ingestion flow for a document.

        State Flow:
        1. Create document record in MySQL with status 'PENDING', tagged with session_id.
        2. Run RAG pipeline ingestion (loading, splitting, vector embedding),
           stamping every chunk with both document_id and session_id.
        3. On Success: Update document status to 'COMPLETED' with chunk count.
        4. On Exception: Update document status to 'FAILED' with error details and re-raise.

        Args:
            db: Active SQLAlchemy session provided by the API layer.
            document_id: Unique UUID string representing the document.
            filename: Original file name uploaded by the user.
            file_path: Local disk path where the file is stored.
            session_id: Identifier of the session that owns this document.

        Returns:
            Dict[str, Any]: Summary dictionary containing "document_id" and "chunks_count".

        Raises:
            ValueError: Propagated if any input parameter validation fails downstream.
            RuntimeError: Propagated if database operations or pipeline execution fails.
            Exception: Re-raises any unhandled pipeline/database exceptions after logging state.
        """
        logger.info(
            "Registering document '%s' (ID: %s, session: %s) as PENDING.",
            filename,
            document_id,
            session_id,
        )

        repository.create_pending_document(
            db=db,
            document_id=document_id,
            original_filename=filename,
            file_path=file_path,
            session_id=session_id,
        )

        try:
            logger.info("Starting pipeline ingestion for document ID: %s", document_id)
            ingestion_result: Dict[str, Any] = self.pipeline.ingest_document(
                document_id=document_id,
                file_path=file_path,
                session_id=session_id,
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

            if os.path.exists(file_path):
                logger.info("Cleaning up file after failed ingestion: %s", file_path)
                os.remove(file_path)
            raise

    def list_documents(
        self,
        db: Session,
        session_id: str,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Document]:
        """
        Retrieve a paginated list of active (non-deleted) documents owned by session_id.

        Args:
            db: Active SQLAlchemy session provided by the API layer.
            session_id: Identifier of the session whose documents should be listed.
            status: Optional status filter (must be a valid DocumentStatus value).
            limit: Maximum number of records to return.
            offset: Number of records to skip.

        Returns:
            List[Document]: A list of active Document ORM objects owned by session_id.

        Raises:
            ValueError: Propagated if session_id/status/limit/offset validation fails.
            RuntimeError: Propagated if the database query fails.
        """
        return repository.list_documents(
            db=db,
            session_id=session_id,
            status=status,
            limit=limit,
            offset=offset,
        )

    def get_document(
        self,
        db: Session,
        document_id: str,
        session_id: str,
    ) -> Optional[Document]:
        """
        Retrieve a single active (non-deleted) document by its ID, scoped to session_id.

        Args:
            db: Active SQLAlchemy session provided by the API layer.
            document_id: Unique identifier of the document to retrieve.
            session_id: Identifier of the session that must own the document.

        Returns:
            Optional[Document]: The Document ORM object if found and owned by session_id, else None.

        Raises:
            ValueError: Propagated if document_id or session_id is empty or whitespace.
            RuntimeError: Propagated if the database query fails.
        """
        return repository.get_document(
            db=db,
            document_id=document_id,
            session_id=session_id,
        )

    def delete_document(
        self,
        db: Session,
        document_id: str,
        session_id: str,
    ) -> None:
        """
        Orchestrate the end-to-end deletion flow for a document owned by session_id.

        Deletion Flow:
        1. Fetch the document record (scoped to session_id) to obtain its file_path.
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
            session_id: Identifier of the session that must own the document.

        Raises:
            ValueError: If no active document owned by session_id exists with the given ID.
            RuntimeError: Propagated if Chroma deletion or the final
                MySQL soft delete fails.
        """
        document = repository.get_document(db=db, document_id=document_id, session_id=session_id)

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

    def chat(
        self,
        db: Session,
        document_id: str,
        session_id: str,
        question: str,
        top_k: int = 5,
    ) -> dict:
        """
        Query a completed document using the RAG pipeline, scoped to session_id
        so a chat can never be answered from another session's document chunks.
        """
        document = repository.get_document(db=db, document_id=document_id, session_id=session_id)

        if document is None:
            raise ValueError(f"No active document found with id={document_id}")

        if document.status != DocumentStatus.COMPLETED.value:
            raise ValueError(
                f"Document is not ready for chat (status={document.status})"
            )

        return self.pipeline.run(
            query=question,
            top_k=top_k,
            filter={"document_id": document_id, "session_id": session_id},
        )