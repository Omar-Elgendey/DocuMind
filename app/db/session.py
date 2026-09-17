import os
from typing import Generator

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session


load_dotenv()


DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_NAME = os.getenv("DB_NAME")

DATABASE_URL = os.getenv("DATABASE_URL")


if not DATABASE_URL:
    if not all([DB_USER, DB_PASSWORD, DB_NAME]):
        raise ValueError(
            "Missing database configuration. Set DATABASE_URL "
            "or DB_USER, DB_PASSWORD, and DB_NAME."
        )

    DATABASE_URL = (
        f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}"
        f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )


DB_SSL_CA_PATH = os.getenv("DB_SSL_CA_PATH")

connect_args = {}
if DB_SSL_CA_PATH:
    connect_args = {"ssl": {"ca": DB_SSL_CA_PATH}}

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=3600,
    echo=False,
    connect_args=connect_args,
)


SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
)


Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    """Provide a database session for FastAPI endpoints."""
    db = SessionLocal()

    try:
        yield db
    finally:
        db.close()