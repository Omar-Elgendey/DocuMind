import logging
import os

from dotenv import load_dotenv
import fastapi
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import health
from app.api.routes import documents
from app.db.session import Base, engine
import app.db.models  # noqa: F401
from app.rag.embedding import get_embedding_model
from app.rag.llm import GroqProvider, LLMGenerator
from app.rag.pipeline import RAGPipeline
from app.rag.retriever import DocuMindRetriever
from app.rag.vector_store import get_vector_store


load_dotenv()

logger = logging.getLogger(__name__)

app = FastAPI(
    title="DocuMind API",
    description="Document Retrieval-Augmented Generation (RAG) System Engine",
    version="1.0.0",
)

# Parse CORS_ORIGINS from Railway variables
cors_origins_raw = os.getenv("CORS_ORIGINS", "")
origins = [origin.strip() for origin in cors_origins_raw.split(",") if origin.strip()]

# Fallback to wildcard if no specific origins configured
if not origins:
    origins = ["*"]

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Routers with v1 prefix
app.include_router(health.router, prefix="/api/v1")
app.include_router(documents.router, prefix="/api/v1")


@app.on_event("startup")
def startup_event() -> None:
    """
    Initialize the database and build the shared RAGPipeline
    once at application startup.
    """

    # Create database tables if they do not already exist.
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables initialized successfully.")

    embedding_model = get_embedding_model()
    vector_store = get_vector_store(embedding_model=embedding_model)
    retriever = DocuMindRetriever(vector_store=vector_store)
    generator = LLMGenerator(provider=GroqProvider())

    app.state.pipeline = RAGPipeline(
        retriever=retriever,
        generator=generator,
        vector_store=vector_store,
    )

    logger.info(
        "DocuMind RAG Pipeline successfully loaded and attached to app.state."
    )


@app.on_event("shutdown")
def shutdown_event() -> None:
    """Clean up resources upon application shutdown."""
    logger.info("Shutting down DocuMind API server...")