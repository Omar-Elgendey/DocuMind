import os
from functools import lru_cache
from typing import List, Optional

from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _load_embedding_model(model_name: str) -> HuggingFaceInferenceAPIEmbeddings:
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_TOKEN environment variable is missing!")

    return HuggingFaceInferenceAPIEmbeddings(
        api_key=hf_token,
        model_name=model_name,
    )


def get_embedding_model(
    model_name: Optional[str] = None,
) -> HuggingFaceInferenceAPIEmbeddings:
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
    model: Optional[HuggingFaceInferenceAPIEmbeddings] = None,
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
    model: Optional[HuggingFaceInferenceAPIEmbeddings] = None,
) -> List[float]:
    if not text or not text.strip():
        raise ValueError("Query text cannot be empty or whitespace-only.")

    model = model or get_embedding_model()

    try:
        return model.embed_query(text)
    except Exception as exc:
        raise RuntimeError(f"Failed to generate query embedding: {exc}") from exc