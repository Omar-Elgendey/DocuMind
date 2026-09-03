import pytest
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from app.rag.vector_store import (
    get_vector_store,
    _prepare_document_ids,
    add_documents,
    similarity_search,
    delete_documents_by_id,
    upsert_document,
)


@pytest.fixture
def mock_embedding_model():
    return MagicMock()


@pytest.fixture
def mock_vector_store():
    store = MagicMock()

    store.add_documents.return_value = None

    store.similarity_search_with_score.return_value = [
        (
            Document(
                page_content="chunk 1",
                metadata={"document_id": "doc1"},
            ),
            0.12,
        ),
        (
            Document(
                page_content="chunk 2",
                metadata={"document_id": "doc1"},
            ),
            0.34,
        ),
    ]

    store.get.return_value = {
        "ids": ["doc1_0", "doc1_1"]
    }

    store.delete.return_value = None

    return store


@pytest.fixture
def sample_documents():
    return [
        Document(
            page_content="First chunk",
            metadata={"document_id": "doc1"},
        ),
        Document(
            page_content="Second chunk",
            metadata={"document_id": "doc1"},
        ),
    ]


@patch("app.rag.vector_store.Chroma")
def test_get_vector_store_success(
    mock_chroma_cls,
    mock_embedding_model,
):
    mock_chroma_cls.return_value = MagicMock()

    store = get_vector_store(
        embedding_model=mock_embedding_model,
        persist_directory="./test_dir",
        collection_name="test_collection",
    )

    mock_chroma_cls.assert_called_once_with(
        collection_name="test_collection",
        embedding_function=mock_embedding_model,
        persist_directory="./test_dir",
    )

    assert store is not None


@patch("app.rag.vector_store.get_embedding_model")
@patch("app.rag.vector_store.Chroma")
def test_get_vector_store_uses_default_embedding_model(
    mock_chroma_cls,
    mock_get_embedding_model,
    mock_embedding_model,
):
    mock_get_embedding_model.return_value = mock_embedding_model

    get_vector_store()

    mock_get_embedding_model.assert_called_once()


@patch("app.rag.vector_store.Chroma")
def test_get_vector_store_failure_raises_runtime_error(
    mock_chroma_cls,
    mock_embedding_model,
):
    mock_chroma_cls.side_effect = Exception("Connection error")

    with pytest.raises(
        RuntimeError,
        match="Failed to initialize vector store collection",
    ):
        get_vector_store(
            embedding_model=mock_embedding_model,
        )


def test_prepare_document_ids_valid(sample_documents):
    documents, ids = _prepare_document_ids(sample_documents)

    assert documents == sample_documents
    assert ids == ["doc1_0", "doc1_1"]


def test_prepare_document_ids_empty_list_raises_value_error():
    with pytest.raises(
        ValueError,
        match="Cannot add an empty document list",
    ):
        _prepare_document_ids([])


def test_prepare_document_ids_missing_document_id_raises_value_error():
    documents = [
        Document(
            page_content="No ID here",
            metadata={},
        )
    ]

    with pytest.raises(
        ValueError,
        match="non-empty 'document_id'",
    ):
        _prepare_document_ids(documents)


def test_prepare_document_ids_whitespace_document_id_raises_value_error():
    documents = [
        Document(
            page_content="Blank ID",
            metadata={"document_id": "   "},
        )
    ]

    with pytest.raises(
        ValueError,
        match="non-empty 'document_id'",
    ):
        _prepare_document_ids(documents)


def test_add_documents_success(
    sample_documents,
    mock_vector_store,
):
    ids = add_documents(
        sample_documents,
        vector_store=mock_vector_store,
    )

    assert ids == ["doc1_0", "doc1_1"]

    mock_vector_store.add_documents.assert_called_once_with(
        documents=sample_documents,
        ids=["doc1_0", "doc1_1"],
    )


def test_add_documents_empty_list_raises_value_error(
    mock_vector_store,
):
    with pytest.raises(
        ValueError,
        match="Cannot add an empty document list",
    ):
        add_documents(
            [],
            vector_store=mock_vector_store,
        )


def test_add_documents_missing_document_id_raises_value_error(
    mock_vector_store,
):
    documents = [
        Document(
            page_content="Missing ID",
            metadata={},
        )
    ]

    with pytest.raises(
        ValueError,
        match="non-empty 'document_id'",
    ):
        add_documents(
            documents,
            vector_store=mock_vector_store,
        )


def test_add_documents_failure_raises_runtime_error(
    sample_documents,
    mock_vector_store,
):
    mock_vector_store.add_documents.side_effect = Exception(
        "Chroma write error"
    )

    with pytest.raises(
        RuntimeError,
        match="Failed to add documents to the vector store",
    ):
        add_documents(
            sample_documents,
            vector_store=mock_vector_store,
        )


@patch("app.rag.vector_store.get_vector_store")
def test_add_documents_creates_vector_store_when_not_provided(
    mock_get_vector_store,
    sample_documents,
    mock_vector_store,
):
    mock_get_vector_store.return_value = mock_vector_store

    add_documents(sample_documents)

    mock_get_vector_store.assert_called_once()


def test_similarity_search_success(mock_vector_store):
    results = similarity_search(
        "What are the profits?",
        top_k=2,
        vector_store=mock_vector_store,
    )

    assert len(results) == 2

    mock_vector_store.similarity_search_with_score.assert_called_once_with(
        query="What are the profits?",
        k=2,
        filter=None,
    )


@pytest.mark.parametrize(
    "invalid_query",
    ["", "   ", "\n\t"],
)
def test_similarity_search_empty_query_raises_value_error(
    invalid_query,
    mock_vector_store,
):
    with pytest.raises(
        ValueError,
        match="Search query cannot be empty",
    ):
        similarity_search(
            invalid_query,
            vector_store=mock_vector_store,
        )


@pytest.mark.parametrize(
    "invalid_top_k",
    [0, -1, True, "5", 1.5],
)
def test_similarity_search_invalid_top_k_raises_value_error(
    invalid_top_k,
    mock_vector_store,
):
    with pytest.raises(
        ValueError,
        match="top_k must be a positive integer",
    ):
        similarity_search(
            "query",
            top_k=invalid_top_k,
            vector_store=mock_vector_store,
        )


def test_similarity_search_failure_raises_runtime_error(
    mock_vector_store,
):
    mock_vector_store.similarity_search_with_score.side_effect = Exception(
        "Search backend error"
    )

    with pytest.raises(
        RuntimeError,
        match="Failed to execute similarity search",
    ):
        similarity_search(
            "query",
            vector_store=mock_vector_store,
        )


def test_similarity_search_passes_metadata_filter(
    mock_vector_store,
):
    similarity_search(
        "query",
        top_k=3,
        filter={"document_id": "doc1"},
        vector_store=mock_vector_store,
    )

    mock_vector_store.similarity_search_with_score.assert_called_once_with(
        query="query",
        k=3,
        filter={"document_id": "doc1"},
    )


def test_delete_documents_by_id_success(mock_vector_store):
    delete_documents_by_id(
        "doc1",
        vector_store=mock_vector_store,
    )

    mock_vector_store.get.assert_called_once_with(
        where={"document_id": "doc1"}
    )

    mock_vector_store.delete.assert_called_once_with(
        ids=["doc1_0", "doc1_1"]
    )


def test_delete_documents_by_id_no_matches_skips_delete(
    mock_vector_store,
):
    mock_vector_store.get.return_value = {
        "ids": []
    }

    delete_documents_by_id(
        "doc_not_found",
        vector_store=mock_vector_store,
    )

    mock_vector_store.delete.assert_not_called()


@pytest.mark.parametrize(
    "invalid_id",
    ["", "   "],
)
def test_delete_documents_by_id_empty_id_raises_value_error(
    invalid_id,
    mock_vector_store,
):
    with pytest.raises(
        ValueError,
        match="document_id cannot be empty",
    ):
        delete_documents_by_id(
            invalid_id,
            vector_store=mock_vector_store,
        )


def test_delete_documents_by_id_failure_raises_runtime_error(
    mock_vector_store,
):
    mock_vector_store.get.side_effect = Exception(
        "Chroma get error"
    )

    with pytest.raises(
        RuntimeError,
        match="Failed to delete documents for document_id",
    ):
        delete_documents_by_id(
            "doc1",
            vector_store=mock_vector_store,
        )


def test_upsert_document_success(
    sample_documents,
    mock_vector_store,
):
    ids = upsert_document(
        "doc1",
        sample_documents,
        vector_store=mock_vector_store,
    )

    mock_vector_store.get.assert_called_once_with(
        where={"document_id": "doc1"}
    )

    mock_vector_store.delete.assert_called_once_with(
        ids=["doc1_0", "doc1_1"]
    )

    mock_vector_store.add_documents.assert_called_once()

    assert ids == ["doc1_0", "doc1_1"]


@pytest.mark.parametrize(
    "invalid_id",
    ["", "   "],
)
def test_upsert_document_empty_id_raises_value_error(
    invalid_id,
    sample_documents,
    mock_vector_store,
):
    with pytest.raises(
        ValueError,
        match="document_id cannot be empty",
    ):
        upsert_document(
            invalid_id,
            sample_documents,
            vector_store=mock_vector_store,
        )


def test_upsert_document_empty_documents_raises_value_error(
    mock_vector_store,
):
    with pytest.raises(
        ValueError,
        match="Cannot upsert an empty document list",
    ):
        upsert_document(
            "doc1",
            [],
            vector_store=mock_vector_store,
        )


def test_upsert_document_mismatched_document_id_raises_value_error(
    mock_vector_store,
):
    documents = [
        Document(
            page_content="Wrong owner",
            metadata={"document_id": "other_doc"},
        )
    ]

    with pytest.raises(
        ValueError,
        match="must have a 'document_id' matching",
    ):
        upsert_document(
            "doc1",
            documents,
            vector_store=mock_vector_store,
        )


def test_upsert_document_replaces_old_chunks(
    sample_documents,
    mock_vector_store,
):
    mock_vector_store.get.return_value = {
        "ids": ["doc1_0"]
    }

    upsert_document(
        "doc1",
        sample_documents,
        vector_store=mock_vector_store,
    )

    assert mock_vector_store.delete.call_count == 1
    assert mock_vector_store.add_documents.call_count == 1