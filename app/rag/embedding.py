import os
from functools import lru_cache
from typing import List, Optional

from langchain_community.embeddings import HuggingFaceEmbeddings

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=None)
def _load_embedding_model(model_name: str) -> HuggingFaceEmbeddings:
    """
    Load and cache a HuggingFaceEmbeddings instance for a given model name.
    """
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def get_embedding_model(
    model_name: Optional[str] = None,
) -> HuggingFaceEmbeddings:
    """
    Initialize and return the configured Hugging Face embedding model.
    """
    effective_model_name = model_name or os.getenv(
        "EMBEDDING_MODEL_NAME",
        DEFAULT_EMBEDDING_MODEL,
    )

    try:
        return _load_embedding_model(effective_model_name)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to initialize embedding model '{effective_model_name}': {exc}"
        ) from exc


def embed_documents(
    texts: List[str],
    model: Optional[HuggingFaceEmbeddings] = None,
) -> List[List[float]]:
    """
    Generate embeddings for multiple document chunks.
    """
    if not texts:
        return []

    model = model or get_embedding_model()

    try:
        return model.embed_documents(texts)
    except Exception as exc:
        raise RuntimeError(f"Failed to generate document embeddings: {exc}") from exc


def embed_query(
    text: str,
    model: Optional[HuggingFaceEmbeddings] = None,
) -> List[float]:
    """
    Generate an embedding for a user query.
    """
    if not text or not text.strip():
        raise ValueError("Query text cannot be empty or whitespace-only.")

    model = model or get_embedding_model()

    try:
        return model.embed_query(text)
    except Exception as exc:
        raise RuntimeError(f"Failed to generate query embedding: {exc}") from exc