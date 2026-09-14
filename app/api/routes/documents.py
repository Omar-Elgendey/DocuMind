import os
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session

from app.api.deps import get_document_service
from app.db.session import get_db
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
        result = service.ingest_document(
            db=db,
            document_id=document_id,
            filename=file.filename,
            file_path=file_path,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return result