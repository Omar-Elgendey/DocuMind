import logging

from dotenv import load_dotenv
import fastapi
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import health
from app.api.routes import documents
from app.rag.embedding import get_embedding_model
from app.rag.llm import GroqProvider, LLMGenerator
from app.rag.pipeline import RAGPipeline
from app.rag.retriever import DocuMindRetriever
from app.rag.vector_store import get_vector_store


logger = logging.getLogger(__name__)

app = FastAPI(
    title="DocuMind API",
    description="Document Retrieval-Augmented Generation (RAG) System Engine",
    version="1.0.0",
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
    Build the shared RAGPipeline instance once, at application startup,
    and store it on app.state so every request reuses the same
    embedding model, vector store connection, and LLM client instead
    of recreating them on every call.
    """
    embedding_model = get_embedding_model()
    vector_store = get_vector_store(embedding_model=embedding_model)
    retriever = DocuMindRetriever(vector_store=vector_store)
    generator = LLMGenerator(provider=GroqProvider())

    app.state.pipeline = RAGPipeline(
        retriever=retriever,
        generator=generator,
        vector_store=vector_store,
    )
    logger.info("DocuMind RAG Pipeline successfully loaded and attached to app.state.")
    
@app.on_event("shutdown")
def shutdown_event() -> None:
    """Clean up resources upon application shutdown."""
    logger.info("Shutting down DocuMind API server...")