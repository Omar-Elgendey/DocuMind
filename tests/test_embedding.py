import pytest
from unittest.mock import MagicMock

from app.rag.embedding import embed_documents, embed_query


@pytest.fixture
def mock_embedding_model():
    """Provide a mock embedding model for unit tests."""
    mock_model = MagicMock()

    mock_model.embed_documents.side_effect = (
        lambda texts: [[float(i)] * 3 for i in range(len(texts))]
    )

    mock_model.embed_query.return_value = [0.1, 0.2, 0.3]

    return mock_model


def test_embed_documents_valid_and_order(mock_embedding_model):
    """Test document embeddings preserve input count and order."""
    chunks = [
        "First chunk",
        "Second chunk",
        "Third chunk",
    ]

    embeddings = embed_documents(
        chunks,
        model=mock_embedding_model,
    )

    assert isinstance(embeddings, list)
    assert len(embeddings) == len(chunks)

    assert embeddings == [
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0],
    ]

    mock_embedding_model.embed_documents.assert_called_once_with(chunks)


def test_embed_documents_empty_list():
    """Test that an empty document list returns an empty list."""
    embeddings = embed_documents([])

    assert embeddings == []


def test_embed_query_valid(mock_embedding_model):
    """Test generating an embedding for a valid query."""
    query = "What is DocuMind?"

    embedding = embed_query(
        query,
        model=mock_embedding_model,
    )

    assert isinstance(embedding, list)
    assert embedding == [0.1, 0.2, 0.3]

    mock_embedding_model.embed_query.assert_called_once_with(query)


@pytest.mark.parametrize(
    "invalid_query",
    [
        "",
        "   ",
        "\n\t",
    ],
)
def test_embed_query_empty_raises_value_error(invalid_query):
    """Test that empty or whitespace-only queries raise ValueError."""
    with pytest.raises(
        ValueError,
        match="Query text cannot be empty or whitespace-only.",
    ):
        embed_query(invalid_query)


def test_embed_documents_failure_handling(mock_embedding_model):
    """Test that document embedding failures raise RuntimeError."""
    mock_embedding_model.embed_documents.side_effect = Exception(
        "Model provider error"
    )

    with pytest.raises(
        RuntimeError,
        match="Failed to generate document embeddings.",
    ):
        embed_documents(
            ["Test chunk"],
            model=mock_embedding_model,
        )


def test_embed_query_failure_handling(mock_embedding_model):
    """Test that query embedding failures raise RuntimeError."""
    mock_embedding_model.embed_query.side_effect = Exception(
        "Model inference error"
    )

    with pytest.raises(
        RuntimeError,
        match="Failed to generate query embedding.",
    ):
        embed_query(
            "Test query",
            model=mock_embedding_model,
        )