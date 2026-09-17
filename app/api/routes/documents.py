import os
import uuid
from pathlib import Path

from typing import Optional
from fastapi import APIRouter, Depends, Header, HTTPException, UploadFile, File, Query, status
from sqlalchemy.orm import Session

from app.api.deps import get_document_service
from app.db.session import get_db
from app.rag.loaders import UniversalLoader
from app.services.document_service import DocumentService

from app.schemas.document import (
    DocumentUploadResponse,
    DocumentResponse,
    ChatRequest,
    ChatResponse,
)


router = APIRouter(prefix="/documents", tags=["documents"])

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./data/uploads")
MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024


def require_session_id(
    x_session_id: Optional[str] = Header(None, alias="X-Session-ID"),
) -> str:
    """
    Extract and validate the X-Session-ID header sent by the frontend.

    Every document-scoped endpoint depends on this to make sure requests
    are always tied to a specific browser session, so one session can
    never see or act on another session's documents.

    Raises:
        HTTPException: 400 if the header is missing or empty.
    """
    if not x_session_id or not x_session_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing X-Session-ID header",
        )
    return x_session_id


@router.post("", status_code=201)
def upload_document(
    file: UploadFile = File(...),
    session_id: str = Depends(require_session_id),
    db: Session = Depends(get_db),
    service: DocumentService = Depends(get_document_service),
) -> DocumentUploadResponse:
    """
    Upload a document file, ingest it into the RAG pipeline, and
    persist its metadata and processing status in MySQL, scoped to
    the requesting session.
    """
    extension = Path(file.filename).suffix.lower()

    if extension not in UniversalLoader.SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file extension: {extension}",
        )

    contents = file.file.read()

    if len(contents) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds the maximum allowed size of "
                    f"{MAX_FILE_SIZE_BYTES // (1024 * 1024)}MB.",
        )

    document_id = str(uuid.uuid4())

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    file_path = os.path.join(UPLOAD_DIR, f"{document_id}{extension}")

    with open(file_path, "wb") as f:
        f.write(contents)

    try:
        return service.ingest_document(
            db=db,
            document_id=document_id,
            filename=file.filename,
            file_path=file_path,
            session_id=session_id,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("")
def list_documents(
    session_id: str = Depends(require_session_id),
    db: Session = Depends(get_db),
    service: DocumentService = Depends(get_document_service),
    status_filter: Optional[str] = Query(None, alias="status", description="Filter by document status"),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
) -> list[DocumentResponse]:
    """
    List documents owned by the requesting session, with optional
    filtering by status and pagination.
    """
    try:
        documents = service.list_documents(
            db=db,
            session_id=session_id,
            status=status_filter,
            limit=limit,
            offset=offset,
        )
        return [DocumentResponse.model_validate(doc) for doc in documents]
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve),
        ) from ve
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc


@router.get("/{document_id}")
def get_document(
    document_id: str,
    session_id: str = Depends(require_session_id),
    db: Session = Depends(get_db),
    service: DocumentService = Depends(get_document_service),
) -> DocumentResponse:
    """
    Retrieve details of a single document by its ID, scoped to the
    requesting session.
    """
    try:
        document = service.get_document(db=db, document_id=document_id, session_id=session_id)
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve),
        ) from ve
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )

    return DocumentResponse.model_validate(document)


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(
    document_id: str,
    session_id: str = Depends(require_session_id),
    db: Session = Depends(get_db),
    service: DocumentService = Depends(get_document_service),
) -> None:
    try:
        service.delete_document(db=db, document_id=document_id, session_id=session_id)
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(ve)) from ve
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc


@router.post("/{document_id}/chat")
def chat_with_document(
    document_id: str,
    request: ChatRequest,
    session_id: str = Depends(require_session_id),
    db: Session = Depends(get_db),
    service: DocumentService = Depends(get_document_service),
) -> ChatResponse:
    """
    Query a completed document using the RAG pipeline, scoped to the
    requesting session.
    """
    try:
        result = service.chat(
            db=db,
            document_id=document_id,
            session_id=session_id,
            question=request.question,
            top_k=request.top_k,
        )

        return {
            "answer": result.get("answer", ""),
            "document_id": document_id,
            "sources": result.get("sources", []),
        }

    except ValueError as ve:
        error_msg = str(ve)
        if "No active document" in error_msg:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=error_msg,
            ) from ve

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error_msg,
        ) from ve

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc