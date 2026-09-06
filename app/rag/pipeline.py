from typing import Any

from langchain_core.documents import Document

from app.rag.llm import LLMGenerator
from app.rag.prompt import build_prompt, extract_sources
from app.rag.retriever import DocuMindRetriever


class RAGPipeline:
    """
    Orchestrates the complete RAG pipeline by coordinating retrieval,
    prompt building, generation, and source extraction.
    """

    def __init__(self, retriever: DocuMindRetriever, generator: LLMGenerator) -> None:
        """
        Initialize the pipeline with injected dependencies.

        Args:
            retriever: Injected retriever instance for fetching context documents.
            generator: Injected LLM generator instance for producing answers.
        """
        self.retriever = retriever
        self.generator = generator

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