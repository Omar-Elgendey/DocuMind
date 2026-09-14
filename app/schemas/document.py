from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field

from app.db.models import DocumentStatus


class HealthResponse(BaseModel):
    status: str = Field(default="ok", example="ok")


class DocumentBase(BaseModel):
    original_filename: str = Field(..., example="lecture 5.pdf")


class DocumentUploadResponse(BaseModel):
    document_id: str = Field(..., example="efe82c8d-8610-4212-9cfd-866b574f1455")
    chunks_count: int = Field(..., example=33)


class DocumentResponse(DocumentBase):
    id: str = Field(..., example="efe82c8d-8610-4212-9cfd-866b574f1455")
    status: DocumentStatus
    chunks_count: Optional[int] = Field(default=None, example=33)
    created_at: datetime

    class Config:
        from_attributes = True


class ChatRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        example="What are the main topics covered in lecture 5?",
        description="The query string to search within the document.",
    )
    top_k: Optional[int] = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of top relevant chunks to retrieve.",
    )


class SourceChunk(BaseModel):
    document_id: str = Field(..., example="efe82c8d-8610-4212-9cfd-866b574f1455")
    page: Optional[int] = Field(default=None, example=14)


class ChatResponse(BaseModel):
    answer: str = Field(
        ...,
        example="The document covers Data Integration, Redundancy, and Normalization...",
    )
    document_id: str = Field(..., example="efe82c8d-8610-4212-9cfd-866b574f1455")
    sources: List[SourceChunk] = Field(default_factory=list)