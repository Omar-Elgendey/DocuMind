from langchain_core.documents import Document


def build_prompt(query: str, documents: list[Document]) -> str:
    """
    Build a structured prompt combining retrieved document context
    and the user's query.

    Args:
        query: The user's question.
        documents: Retrieved document chunks.

    Returns:
        A formatted prompt string ready for LLM consumption.

    Raises:
        ValueError: If the query is empty or contains only whitespace.
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty or whitespace-only.")

    if not documents:
        return (
            "You are an AI assistant answering questions based on provided documents.\n\n"
            "Context Information:\n"
            "No relevant document context was found for this query.\n\n"
            f"User Query: {query.strip()}\n\n"
            "Instruction:\n"
            "Inform the user that you do not have enough information from the "
            "provided documents to answer the question. "
            "Do not guess or use outside knowledge."
        )

    formatted_context = []

    for index, document in enumerate(documents, start=1):
        formatted_context.append(
            f"[Chunk {index}]\n{document.page_content.strip()}"
        )

    context = "\n\n".join(formatted_context)

    return (
        "You are an AI assistant answering questions based on provided documents.\n"
        "Answer the question based strictly on the provided context below. "
        "If the answer cannot be determined from the context, state that you "
        "do not have enough information.\n\n"
        f"Context:\n{context}\n\n"
        f"User Query: {query.strip()}\n\n"
        "Answer:"
    )


def extract_sources(
    documents: list[Document],
) -> list[dict[str, str | int | None]]:
    """
    Extract source metadata from retrieved document chunks.

    Args:
        documents: Retrieved document chunks.

    Returns:
        A list containing document IDs and page numbers.

    Raises:
        RuntimeError: If a document is missing the required document_id.
    """
    if not documents:
        return []

    sources = []

    for document in documents:
        document_id = document.metadata.get("document_id")

        if not document_id:
            raise RuntimeError(
                "Document is missing required 'document_id' in metadata."
            )

        sources.append(
            {
                "document_id": document_id,
                "page": document.metadata.get("page"),
            }
        )

    return sources