from fastapi import Depends, Request
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.rag.pipeline import RAGPipeline
from app.services.document_service import DocumentService


def get_pipeline(request: Request) -> RAGPipeline:
    """
    Retrieve the singleton RAGPipeline instance built once at
    application startup and stored on app.state.
    """
    return request.app.state.pipeline


def get_document_service(
    pipeline: RAGPipeline = Depends(get_pipeline),
) -> DocumentService:
    """
    Build a DocumentService instance using the shared pipeline singleton.
    """
    return DocumentService(pipeline=pipeline)