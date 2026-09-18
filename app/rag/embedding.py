import os
from functools import lru_cache

from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-small"


@lru_cache(maxsize=None)
def _load_embedding_model(model_name: str) -> HuggingFaceInferenceAPIEmbeddings:
    """
    Load and cache a HuggingFaceInferenceAPIEmbeddings instance for a given model name.

    This offloads model execution to Hugging Face Inference API, saving CPU and RAM
    usage on the hosting server.
    """
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_TOKEN environment variable is missing!")

    return HuggingFaceInferenceAPIEmbeddings(
        api_key=hf_token,
        model_name=model_name,
    )


def get_embedding_model(
    model_name: str | None = None,
) -> HuggingFaceInferenceAPIEmbeddings:
    """
    Initialize and return the configured Hugging Face Inference API embedding model.

    The underlying model is cached in-process (see _load_embedding_model),
    so repeated calls with the same effective model name are cheap and
    do not reload the model instance.

    Args:
        model_name: Optional embedding model name. If not provided,
            the EMBEDDING_MODEL_NAME environment variable is used.
            Otherwise, the default model is intfloat/multilingual-e5-small.

    Returns:
        An initialized HuggingFaceInferenceAPIEmbeddings instance.

    Raises:
        RuntimeError: If the embedding model cannot be initialized.
    """
    effective_model_name = model_name or os.getenv(
        "EMBEDDING_MODEL_NAME",
        DEFAULT_EMBEDDING_MODEL,
    )

    try:
        return _load_embedding_model(effective_model_name)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to initialize embedding model '{effective_model_name}'."
        ) from exc


def embed_documents(
    texts: list[str],
    model: HuggingFaceInferenceAPIEmbeddings | None = None,
) -> list[list[float]]:
    """
    Generate embeddings for multiple document chunks with E5 passage prefix.

    Args:
        texts: List of document chunks to embed.
        model: Optional pre-initialized embedding model.

    Returns:
        A list of embedding vectors in the same order as the input chunks.

    Raises:
        RuntimeError: If embedding generation fails.
    """
    if not texts:
        return []

    model = model or get_embedding_model()

    # E5 models require 'passage: ' prefix for document chunks
    formatted_texts = [f"passage: {t}" for t in texts]

    try:
        return model.embed_documents(formatted_texts)
    except Exception as exc:
        raise RuntimeError("Failed to generate document embeddings.") from exc


def embed_query(
    text: str,
    model: HuggingFaceInferenceAPIEmbeddings | None = None,
) -> list[float]:
    """
    Generate an embedding for a user query with E5 query prefix.

    Args:
        text: User query text.
        model: Optional pre-initialized embedding model.

    Returns:
        A single embedding vector.

    Raises:
        ValueError: If the query is empty or whitespace-only.
        RuntimeError: If embedding generation fails.
    """
    if not text or not text.strip():
        raise ValueError("Query text cannot be empty or whitespace-only.")

    model = model or get_embedding_model()

    # E5 models require 'query: ' prefix for input queries
    formatted_text = f"query: {text}"

    try:
        return model.embed_query(formatted_text)
    except Exception as exc:
        raise RuntimeError("Failed to generate query embedding.") from exc