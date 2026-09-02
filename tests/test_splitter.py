import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import pytest

from langchain_core.documents import Document
from app.rag.splitter import split_documents


def test_splitter_returns_documents():
    document = Document(page_content="This is a test document.")

    chunks = split_documents([document])

    assert isinstance(chunks, list)
    assert all(isinstance(chunk, Document) for chunk in chunks)


def test_long_document_is_split():
    text = "This is a sentence. " * 300
    document = Document(page_content=text)

    chunks = split_documents(
        [document],
        chunk_size=1000,
        chunk_overlap=200
    )

    assert len(chunks) > 1


def test_chunks_are_not_empty():
    text = "This is a sentence. " * 300
    document = Document(page_content=text)

    chunks = split_documents([document])

    assert all(chunk.page_content.strip() for chunk in chunks)


def test_chunks_do_not_exceed_chunk_size():
    text = "This is a sentence. " * 300
    document = Document(page_content=text)

    chunk_size = 1000

    chunks = split_documents(
        [document],
        chunk_size=chunk_size,
        chunk_overlap=200
    )

    assert all(
        len(chunk.page_content) <= chunk_size
        for chunk in chunks
    )


def test_metadata_is_preserved():
    document = Document(
        page_content="This is a sentence. " * 100,
        metadata={
            "source": "test.pdf",
            "page": 1
        }
    )

    chunks = split_documents([document])

    assert len(chunks) > 1

    for chunk in chunks:
        assert chunk.metadata["source"] == "test.pdf"
        assert chunk.metadata["page"] == 1


def test_short_document_stays_one_chunk():
    document = Document(
        page_content="This is a short document."
    )

    chunks = split_documents(
        [document],
        chunk_size=1000,
        chunk_overlap=200
    )

    assert len(chunks) == 1


def test_empty_input_returns_empty_list():
    chunks = split_documents([])

    assert chunks == []


def test_invalid_chunk_size():
    document = Document(page_content="Some text.")

    with pytest.raises(ValueError):
        split_documents(
            [document],
            chunk_size=0
        )


def test_invalid_chunk_overlap():
    document = Document(page_content="Some text.")

    with pytest.raises(ValueError):
        split_documents(
            [document],
            chunk_size=100,
            chunk_overlap=100
        )


def test_chunk_overlap():
    text = "a" * 1000

    chunks = split_documents(
        [Document(page_content=text)],
        chunk_size=200,
        chunk_overlap=50
    )

    assert len(chunks) > 1

    assert (
        chunks[0].page_content[-50:]
        == chunks[1].page_content[:50]
    )
    
    
    
if __name__ == "__main__":
    document = Document(
        page_content="This is a test document. " * 100
    )

    chunks = split_documents(
        [document],
        chunk_size=200,
        chunk_overlap=50
    )

    print(f"Number of chunks: {len(chunks)}")

    for i, chunk in enumerate(chunks, start=1):
        print("\n" + "=" * 60)
        print(f"CHUNK {i}")
        print(f"Length: {len(chunk.page_content)}")
        print(chunk.page_content)