import logging
import uuid
from typing import Any

from langchain_core.documents import Document

from app.rag.llm import LLMGenerator
from app.rag.loaders import UniversalLoader
from app.rag.prompt import build_prompt, extract_sources
from app.rag.retriever import DocuMindRetriever
from app.rag.splitter import split_documents
from app.rag.vector_store import add_documents, get_vector_store


logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Orchestrates the complete RAG pipeline by coordinating ingestion,
    retrieval, prompt building, generation, and source extraction.
    """

    def __init__(
        self,
        retriever: DocuMindRetriever,
        generator: LLMGenerator,
        vector_store: Any | None = None,
    ) -> None:
        """
        Initialize the pipeline with injected dependencies.

        Args:
            retriever: Injected retriever instance for fetching context documents.
            generator: Injected LLM generator instance for producing answers.
            vector_store: Injected Chroma vector store used for ingestion.
                If not provided, a default instance is obtained lazily via
                get_vector_store() the first time ingest_document() is called.
        """
        self.retriever = retriever
        self.generator = generator
        self._vector_store = vector_store

    def _get_vector_store(self) -> Any:
        if self._vector_store is None:
            self._vector_store = get_vector_store()
        return self._vector_store

    def ingest_document(
        self,
        file_path: str,
        document_id: str | None = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> dict[str, Any]:
        """
        Load a file, split it into chunks tagged with a document_id,
        and add those chunks to the vector store.

        Args:
            file_path: Path to the file to ingest (pdf, docx, pptx, or txt).
            document_id: Optional identifier for the document. If not
                provided, a new UUID4 is generated.
            chunk_size: Maximum number of characters per chunk.
            chunk_overlap: Number of overlapping characters between chunks.

        Returns:
            A dictionary containing 'document_id' (str) and
            'chunks_count' (int).

        Raises:
            ValueError: If file_path or the resulting chunk_size/chunk_overlap
                combination is invalid.
            RuntimeError: If loading, splitting, or storing the document fails.
        """

        if not file_path or not file_path.strip():
            raise ValueError("file_path must not be empty.")

        effective_document_id = document_id or str(uuid.uuid4())

        logger.info(
            "Starting ingestion for '%s' with document_id=%s",
            file_path,
            effective_document_id,
        )

        raw_documents: list[Document] = UniversalLoader(file_path).load()

        chunks: list[Document] = split_documents(
            raw_documents,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            document_id=effective_document_id,
        )

        if not chunks:
            logger.warning(
                "No chunks were produced for '%s'; nothing was added to the vector store.",
                file_path,
            )
            return {"document_id": effective_document_id, "chunks_count": 0}

        try:
            stored_ids: list[str] = add_documents(
                chunks, vector_store=self._get_vector_store()
            )
        except (ValueError, RuntimeError):
            raise
        except Exception as e:
            logger.exception(
                "Unexpected error while storing chunks for document_id=%s",
                effective_document_id,
            )
            raise RuntimeError(
                f"Failed to store document chunks: {e}"
            ) from e

        logger.info(
            "Successfully ingested '%s' as document_id=%s (%d chunks).",
            file_path,
            effective_document_id,
            len(stored_ids),
        )

        return {"document_id": effective_document_id, "chunks_count": len(stored_ids)}

    def run(
        self,
        query: str,
        top_k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Execute the full RAG orchestration flow.

        Args:
            query: The user's question string.
            top_k: Maximum number of relevant documents to retrieve. Defaults to 5.
            filter: Optional metadata filter dict passed to the retriever.

        Returns:
            A dictionary containing 'answer' (str) and 'sources' (list[dict]).

        Raises:
            ValueError: Propagated if query validation fails in downstream components.
            RuntimeError: Propagated if retrieval or source extraction fails.
            LLMGenerationError: Propagated if LLM generation fails.
        """

        effective_retriever = self.retriever.model_copy(
            update={"top_k": top_k, "metadata_filter": filter}
        )
        documents: list[Document] = effective_retriever.invoke(query)

        prompt: str = build_prompt(query, documents)
        answer: str = self.generator.generate(prompt)
        sources: list[dict[str, Any]] = extract_sources(documents)

        return {
            "answer": answer,
            "sources": sources,
        }