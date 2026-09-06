import logging
from typing import List, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def split_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    document_id: Optional[str] = None,
) -> List[Document]:
    """
    Split a list of LangChain Document objects into smaller chunks using
    RecursiveCharacterTextSplitter while preserving metadata and preventing empty chunks.

    Args:
        documents (List[Document]): The documents to split.
        chunk_size (int): The maximum number of characters for each chunk.
        chunk_overlap (int): The number of overlapping characters between chunks.
        document_id (Optional[str]): If provided, stamped into the metadata of
            every resulting chunk under the "document_id" key.

    Returns:
        List[Document]: A list of valid, chunked Document objects.

    Raises:
        ValueError: If chunk_size, chunk_overlap, or document_id is invalid.
        RuntimeError: If an unexpected error occurs during splitting.
    """

    if not documents:
        logger.warning(
            "No documents provided for splitting. Returning an empty list."
        )
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be >= 0 and smaller than chunk_size."
        )

    if document_id is not None and not document_id.strip():
        raise ValueError("document_id must not be empty or whitespace.")

    try:
        logger.info(
            "Starting chunking process for %d document(s) | "
            "chunk_size=%d, chunk_overlap=%d",
            len(documents),
            chunk_size,
            chunk_overlap
        )

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n",". "," ", ""]
        )

        chunks = text_splitter.split_documents(documents)

        valid_chunks: List[Document] = []

        for chunk in chunks:
            if chunk.page_content and chunk.page_content.strip():
                if document_id is not None:
                    chunk.metadata["document_id"] = document_id
                valid_chunks.append(chunk)
            else:
                logger.warning(
                    "Filtered out an empty or whitespace-only chunk."
                )

        logger.info(
            "Successfully split documents into %d valid chunk(s).",
            len(valid_chunks)
        )

        return valid_chunks

    except Exception as e:
        logger.exception(
            "Unexpected error occurred during document splitting"
        )
        raise RuntimeError(
            f"Document splitting failed: {e}"
        ) from e