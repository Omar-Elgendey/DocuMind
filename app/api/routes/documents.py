import os
import uuid
from pathlib import Path

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File,Query,status
from sqlalchemy.orm import Session

from app.api.deps import get_document_service
from app.db.session import get_db
from app.db.models import Document, DocumentStatus
from app.rag.loaders import UniversalLoader
from app.services.document_service import DocumentService


router = APIRouter(prefix="/documents", tags=["documents"])

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./data/uploads")
MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024  


@router.post("", status_code=201)
def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    service: DocumentService = Depends(get_document_service),
) -> dict:
    """
    Upload a document file, ingest it into the RAG pipeline, and
    persist its metadata and processing status in MySQL.
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
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

@router.get("")
def list_documents(
    db: Session = Depends(get_db),
    service: DocumentService = Depends(get_document_service),
    status_filter: Optional[str] = Query(None, alias="status", description="Filter by document status"),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
) -> list[dict]:
    """
    List documents with optional filtering by status and pagination.
    """
    try:
        documents = service.list_documents(
            db=db,
            status=status_filter,
            limit=limit,
            offset=offset,
        )
        return [
            {
                "id": doc.id,
                "original_filename": doc.original_filename,
                "status": doc.status,
                "chunks_count": doc.chunks_count,
                "created_at": doc.created_at,
            }
            for doc in documents
        ]
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
    db: Session = Depends(get_db),
    service: DocumentService = Depends(get_document_service),
) -> dict:
    """
    Retrieve details of a single document by its ID.
    """
    try:
        document = service.get_document(db=db, document_id=document_id)
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

    return {
        "id": document.id,
        "original_filename": document.original_filename,
        "status": document.status,
        "chunks_count": document.chunks_count,
        "created_at": document.created_at,
    }
    
    
@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(
    document_id: str,
    db: Session = Depends(get_db),
    service: DocumentService = Depends(get_document_service),
) -> None:
    try:
        service.delete_document(db=db, document_id=document_id)
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(ve)) from ve
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc
      

@router.post("/{document_id}/chat")
def chat_with_document(
    document_id: str,
    request: dict,  
    db: Session = Depends(get_db),
    service: DocumentService = Depends(get_document_service),
) -> dict:
    """
    Query a completed document using the RAG pipeline.
    """
    question = request.get("question")
    if not question or not str(question).strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question field is required and must not be empty.",
        )

    try:
        result = service.chat(
            db=db,
            document_id=document_id,
            question=question,
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
        
        return result