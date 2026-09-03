import os

from langchain_core.documents import Document
from langchain_chroma import Chroma

from app.rag.embedding import get_embedding_model


DEFAULT_PERSIST_DIRECTORY = os.getenv(
    "CHROMA_PERSIST_DIRECTORY",
    "./data/chroma",
)

DEFAULT_COLLECTION_NAME = os.getenv(
    "CHROMA_COLLECTION_NAME",
    "documind_collection",
)


def get_vector_store(
    embedding_model=None,
    persist_directory: str = DEFAULT_PERSIST_DIRECTORY,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> Chroma:
    """
    Initialize or load the persistent Chroma vector store.

    Args:
        embedding_model: Optional pre-initialized embedding model.
        persist_directory: Local path where ChromaDB persists data.
        collection_name: Name of the Chroma collection.

    Returns:
        An initialized Chroma vector store.

    Raises:
        RuntimeError: If the vector store cannot be initialized.
    """
    if embedding_model is None:
        embedding_model = get_embedding_model()

    try:
        return Chroma(
            collection_name=collection_name,
            embedding_function=embedding_model,
            persist_directory=persist_directory,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to initialize vector store collection "
            f"'{collection_name}'."
        ) from exc


def _prepare_document_ids(
    documents: list[Document],
) -> tuple[list[Document], list[str]]:
    """
    Validate document metadata and generate deterministic chunk IDs.

    Args:
        documents: Document chunks to prepare.

    Returns:
        A tuple containing the documents and their generated IDs.

    Raises:
        ValueError: If documents are empty or document_id is missing.
    """
    if not documents:
        raise ValueError(
            "Cannot add an empty document list to the vector store."
        )

    document_ids = []
    prepared_documents = []

    for index, document in enumerate(documents):
        document_id = document.metadata.get("document_id")

        if not document_id or not str(document_id).strip():
            raise ValueError(
                "Each document must contain a non-empty "
                "'document_id' in its metadata."
            )

        chunk_id = f"{document_id}_{index}"

        prepared_documents.append(document)
        document_ids.append(chunk_id)

    return prepared_documents, document_ids


def add_documents(
    documents: list[Document],
    vector_store: Chroma | None = None,
    embedding_model=None,
    persist_directory: str = DEFAULT_PERSIST_DIRECTORY,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> list[str]:
    """
    Add document chunks to the Chroma vector store.

    Args:
        documents: Document chunks with text and metadata.
        vector_store: Optional pre-initialized vector store.
        embedding_model: Optional pre-initialized embedding model.
        persist_directory: Local Chroma persistence path.
        collection_name: Chroma collection name.

    Returns:
        List of generated chunk IDs.

    Raises:
        ValueError: If documents are empty or missing document_id.
        RuntimeError: If documents cannot be added.
    """
    documents, ids = _prepare_document_ids(documents)

    if vector_store is None:
        vector_store = get_vector_store(
            embedding_model=embedding_model,
            persist_directory=persist_directory,
            collection_name=collection_name,
        )

    try:
        vector_store.add_documents(
            documents=documents,
            ids=ids,
        )

        return ids

    except Exception as exc:
        raise RuntimeError(
            "Failed to add documents to the vector store."
        ) from exc


def similarity_search(
    query: str,
    top_k: int = 4,
    filter: dict | None = None,
    vector_store: Chroma | None = None,
    embedding_model=None,
    persist_directory: str = DEFAULT_PERSIST_DIRECTORY,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> list[tuple[Document, float]]:
    """
    Perform semantic similarity search on the vector store.

    Args:
        query: User search query.
        top_k: Number of results to return.
        filter: Optional metadata filter.
        vector_store: Optional pre-initialized vector store.
        embedding_model: Optional pre-initialized embedding model.
        persist_directory: Local Chroma persistence path.
        collection_name: Chroma collection name.

    Returns:
        A list of documents and their similarity scores.

    Raises:
        ValueError: If query is empty or top_k is invalid.
        RuntimeError: If the search fails.
    """
    if not query or not query.strip():
        raise ValueError(
            "Search query cannot be empty or whitespace-only."
        )

    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        raise ValueError(
            "top_k must be a positive integer greater than zero."
        )

    if vector_store is None:
        vector_store = get_vector_store(
            embedding_model=embedding_model,
            persist_directory=persist_directory,
            collection_name=collection_name,
        )

    try:
        return vector_store.similarity_search_with_score(
            query=query,
            k=top_k,
            filter=filter,
        )

    except Exception as exc:
        raise RuntimeError(
            "Failed to execute similarity search."
        ) from exc


def delete_documents_by_id(
    document_id: str,
    vector_store: Chroma | None = None,
    persist_directory: str = DEFAULT_PERSIST_DIRECTORY,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> None:
    """
    Delete all chunks associated with a document ID.

    Args:
        document_id: Unique document identifier.
        vector_store: Optional pre-initialized vector store.
        persist_directory: Local Chroma persistence path.
        collection_name: Chroma collection name.

    Raises:
        ValueError: If document_id is empty.
        RuntimeError: If deletion fails.
    """
    if not document_id or not document_id.strip():
        raise ValueError(
            "document_id cannot be empty or whitespace-only."
        )

    if vector_store is None:
        vector_store = get_vector_store(
            persist_directory=persist_directory,
            collection_name=collection_name,
        )

    try:
        results = vector_store.get(
            where={"document_id": document_id}
        )

        ids_to_delete = results.get("ids", [])

        if ids_to_delete:
            vector_store.delete(ids=ids_to_delete)

    except Exception as exc:
        raise RuntimeError(
            f"Failed to delete documents for document_id "
            f"'{document_id}'."
        ) from exc


def upsert_document(
    document_id: str,
    documents: list[Document],
    vector_store: Chroma | None = None,
    embedding_model=None,
    persist_directory: str = DEFAULT_PERSIST_DIRECTORY,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> list[str]:
    """
    Replace all existing chunks for a document with new chunks.

    Args:
        document_id: Unique document identifier.
        documents: New document chunks.
        vector_store: Optional pre-initialized vector store.
        embedding_model: Optional pre-initialized embedding model.
        persist_directory: Local Chroma persistence path.
        collection_name: Chroma collection name.

    Returns:
        List of newly inserted chunk IDs.

    Raises:
        ValueError: If document_id or documents are invalid.
        RuntimeError: If deletion or insertion fails.
    """
    if not document_id or not document_id.strip():
        raise ValueError(
            "document_id cannot be empty or whitespace-only."
        )

    if not documents:
        raise ValueError(
            "Cannot upsert an empty document list."
        )

    for document in documents:
        metadata_document_id = document.metadata.get("document_id")

        if metadata_document_id != document_id:
            raise ValueError(
                "All document chunks must have a 'document_id' "
                "matching the provided document_id."
            )

    if vector_store is None:
        vector_store = get_vector_store(
            embedding_model=embedding_model,
            persist_directory=persist_directory,
            collection_name=collection_name,
        )

    delete_documents_by_id(
        document_id=document_id,
        vector_store=vector_store,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )

    return add_documents(
        documents=documents,
        vector_store=vector_store,
        embedding_model=embedding_model,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )