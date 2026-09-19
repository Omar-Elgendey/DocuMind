import os
from functools import lru_cache
from typing import List, Optional

from langchain_community.embeddings import FastEmbedEmbeddings

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=None)
def _load_embedding_model(model_name: str) -> FastEmbedEmbeddings:
    """
    Load and cache a FastEmbedEmbeddings instance for a given model name.

    FastEmbed uses ONNX Runtime instead of PyTorch, making it
    significantly faster and lighter for CPU-only inference on
    resource-constrained environments.
    """
    return FastEmbedEmbeddings(model_name=model_name)


def get_embedding_model(
    model_name: Optional[str] = None,
) -> FastEmbedEmbeddings:
    """
    Initialize and return the configured FastEmbed embedding model.
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
    model: Optional[FastEmbedEmbeddings] = None,
) -> List[List[float]]:
    if not texts:
        return []

    model = model or get_embedding_model()

    try:
        return model.embed_documents(texts)
    except Exception as exc:
        raise RuntimeError(f"Failed to generate document embeddings: {exc}") from exc


def embed_query(
    text: str,
    model: Optional[FastEmbedEmbeddings] = None,
) -> List[float]:
    if not text or not text.strip():
        raise ValueError("Query text cannot be empty or whitespace-only.")

    model = model or get_embedding_model()

    try:
        return model.embed_query(text)
    except Exception as exc:
        raise RuntimeError(f"Failed to generate query embedding: {exc}") from exc