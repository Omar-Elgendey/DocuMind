import logging
import time
from typing import Any

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

from app.rag.embedding import embed_query, get_embedding_model
from app.rag.vector_store import similarity_search_by_vector

logger = logging.getLogger(__name__)


class DocuMindRetriever(BaseRetriever):
    """
    Retriever for fetching relevant document chunks
    using semantic similarity search.
    """

    vector_store: Any = Field(
        ...,
        description="Initialized Chroma vector store."
    )

    top_k: int = Field(
        default=5,
        gt=0,
        description="Number of relevant document chunks to retrieve."
    )

    metadata_filter: dict | None = Field(
        default=None,
        description="Optional metadata filter for retrieval."
    )

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager=None,
    ) -> list[Document]:
        """
        Retrieve relevant document chunks for the given query.
        """

        if not query or not query.strip():
            raise ValueError(
                "Query cannot be empty or whitespace-only."
            )

        try:
            logger.info(
                "Retrieving documents with top_k=%d",
                self.top_k,
            )

            formatted_filter = None
            if self.metadata_filter:
                clean_filter = {
                    k: v for k, v in self.metadata_filter.items() if v is not None
                }

                if len(clean_filter) == 1:
                    formatted_filter = clean_filter
                elif len(clean_filter) > 1:
                    formatted_filter = {
                        "$and": [{k: v} for k, v in clean_filter.items()]
                    }

            # --- DIAGNOSTIC TIMING (temporary) ---
            t0 = time.perf_counter()
            query_embedding = embed_query(query, model=get_embedding_model())
            t1 = time.perf_counter()
            logger.info("EMBEDDING took %.3f seconds", t1 - t0)

            results = similarity_search_by_vector(
                query_embedding=query_embedding,
                top_k=self.top_k,
                filter=formatted_filter,
                vector_store=self.vector_store,
            )
            t2 = time.perf_counter()
            logger.info("CHROMA SEARCH took %.3f seconds", t2 - t1)
            # --- END DIAGNOSTIC TIMING ---

            logger.info(
                "Retrieval finished, got %d results",
                len(results),
            )

            documents = [
                document
                for document, _score in results
            ]

            return documents

        except Exception as exc:
            logger.exception(
                "Failed to retrieve relevant documents."
            )
            raise RuntimeError(
                "Retriever failed to fetch relevant documents."
            ) from exc