import logging
from typing import Any, Dict, List, Optional
from sqlalchemy.orm import Session

from app.db import repository
from app.db.models import Document
from app.rag.pipeline import RAGPipeline

logger = logging.getLogger(__name__)


class DocumentService:
    """
    Orchestrates document ingestion by managing processing state transitions
    in the MySQL repository and coordinating execution with the RAG pipeline.
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

        # 1. Register PENDING state in MySQL
        repository.create_pending_document(
            db=db,
            document_id=document_id,
            original_filename=filename,
            file_path=file_path,
        )

        try:
            # 2. Execute RAG Ingestion Pipeline
            logger.info("Starting pipeline ingestion for document ID: %s", document_id)
            ingestion_result: Dict[str, Any] = self.pipeline.ingest_document(
                document_id=document_id,
                file_path=file_path,
            )

            chunks_count: int = ingestion_result.get("chunks_count", 0)

            # 3. Transition state to COMPLETED
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

            # 4. Transition state to FAILED and re-raise exception for caller handling
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