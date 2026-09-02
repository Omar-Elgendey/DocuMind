import logging
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Configure logging for execution tracking
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def split_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[Document]:
    """
    Split a list of LangChain Document objects into smaller chunks using
    RecursiveCharacterTextSplitter while preserving metadata and preventing empty chunks.

    Args:
        documents (List[Document]): The documents to split.
        chunk_size (int): The maximum number of characters for each chunk.
        chunk_overlap (int): The number of overlapping characters between chunks.

    Returns:
        List[Document]: A list of valid, chunked Document objects.

    Raises:
        ValueError: If chunk_size or chunk_overlap is invalid.
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

        # Filter out empty or whitespace-only chunks
        valid_chunks: List[Document] = []

        for chunk in chunks:
            if chunk.page_content and chunk.page_content.strip():
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
