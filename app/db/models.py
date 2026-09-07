import enum
from datetime import datetime, timezone

from sqlalchemy import CHAR, Column, DateTime, Index, Integer, String, Text
from app.db.session import Base


class DocumentStatus(str, enum.Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"


class Document(Base):
    """Document entity representing uploaded files and their processing state."""

    __tablename__ = "documents"

    id = Column(
        CHAR(36),
        primary_key=True,
        nullable=False,
    )

    original_filename = Column(
        String(255),
        nullable=False,
    )

    file_path = Column(
        String(512),
        nullable=False,
    )

    status = Column(
        String(20),
        nullable=False,
        default=DocumentStatus.PENDING.value,
    )

    chunks_count = Column(
        Integer,
        nullable=True,
    )

    error_message = Column(
        Text,
        nullable=True,
    )

    created_at = Column(
        DateTime,
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    updated_at = Column(
        DateTime,
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    deleted_at = Column(
        DateTime,
        nullable=True,
    )

    __table_args__ = (
        Index(
            "idx_active_documents",
            "deleted_at",
            "created_at",
        ),
        Index(
            "idx_documents_status",
            "status",
        ),
    )
