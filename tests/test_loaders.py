from pathlib import Path

from app.rag.loaders import load_pdf_documents


def test_load_pdf_documents():
    pdf_path = Path("data/uploads/Omar_Mohamed_Elgandey_CV.pdf")

    documents = load_pdf_documents(str(pdf_path))

    assert isinstance(documents, list)

    assert len(documents) > 0

    for document in documents:
        assert document.page_content
        assert isinstance(document.metadata, dict)

    print(f"\nSuccessfully loaded {len(documents)} pages.")
    print(f"First page characters: {len(documents[0].page_content)}")
    print(f"Metadata: {documents[0].metadata}")