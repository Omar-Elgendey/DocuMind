import os

from langchain_huggingface import HuggingFaceEmbeddings


DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-small"


def get_embedding_model(
    model_name: str | None = None,
) -> HuggingFaceEmbeddings:
    """
    Initialize and return the configured Hugging Face embedding model.

    Args:
        model_name: Optional embedding model name. If not provided,
            the EMBEDDING_MODEL_NAME environment variable is used.
            Otherwise, the default model is intfloat/multilingual-e5-small.
    Returns:
        An initialized HuggingFaceEmbeddings instance.

    Raises:
        RuntimeError: If the embedding model cannot be initialized.
    """
    model_name = model_name or os.getenv(
        "EMBEDDING_MODEL_NAME",
        DEFAULT_EMBEDDING_MODEL,
    )

    try:
        return HuggingFaceEmbeddings(
            model_name=model_name,
            encode_kwargs={"normalize_embeddings": True},
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to initialize embedding model '{model_name}'."
        ) from exc


def embed_documents(
    texts: list[str],
    model: HuggingFaceEmbeddings | None = None,
) -> list[list[float]]:
    """
    Generate embeddings for multiple document chunks.

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

    try:
        return model.embed_documents(texts)
    except Exception as exc:
        raise RuntimeError("Failed to generate document embeddings.") from exc


def embed_query(
    text: str,
    model: HuggingFaceEmbeddings | None = None,
) -> list[float]:
    """
    Generate an embedding for a user query.

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

    try:
        return model.embed_query(text)
    except Exception as exc:
        raise RuntimeError("Failed to generate query embedding.") from exc