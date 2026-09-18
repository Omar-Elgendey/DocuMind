import logging
from typing import Any

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

from app.rag.vector_store import similarity_search

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

            # --- بداية التعديل المطلوب لـ ChromaDB ---
            formatted_filter = None
            if self.metadata_filter:
                # 1. استبعاد أي عناصر بقيمة None أو فارغة
                clean_filter = {
                    k: v for k, v in self.metadata_filter.items() if v is not None
                }

                # 2. تغليف الفلتر ليكون متوافقاً مع قواعد ChromaDB
                if len(clean_filter) == 1:
                    formatted_filter = clean_filter
                elif len(clean_filter) > 1:
                    formatted_filter = {
                        "$and": [{k: v} for k, v in clean_filter.items()]
                    }
            # --- نهاية التعديل ---

            results = similarity_search(
                query=query,
                top_k=self.top_k,
                filter=formatted_filter,  # تم تمرير الفلتر المعالج هنا
                vector_store=self.vector_store,
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