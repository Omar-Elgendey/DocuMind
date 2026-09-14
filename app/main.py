from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI

from app.api.routes import health
from app.api.routes import documents
from app.rag.embedding import get_embedding_model
from app.rag.llm import GroqProvider, LLMGenerator
from app.rag.pipeline import RAGPipeline
from app.rag.retriever import DocuMindRetriever
from app.rag.vector_store import get_vector_store


app = FastAPI(title="DocuMind API")

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