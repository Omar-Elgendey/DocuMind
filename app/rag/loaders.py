import logging
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader


# Configure logging for clean execution tracking
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def load_pdf_documents(path: str) -> list[Document]:
    """
    Load PDF documents from a file or a directory containing PDF files.

    Args:
        path (str): Path to a PDF file or a directory containing PDF files.

    Returns:
        list[Document]: A list of loaded PDF page documents.

    Raises:
        FileNotFoundError: If the specified path does not exist.
        ValueError: If the specified file is not a PDF.
        RuntimeError: If an unexpected error occurs during PDF loading.
    """

    documents: list[Document] = []
    path_obj = Path(path)

    try:
        # Check if the provided path exists
        if not path_obj.exists():
            raise FileNotFoundError(
                f"The specified path does not exist: {path}"
            )

        # Handle a single PDF file
        if path_obj.is_file():

            if path_obj.suffix.lower() != ".pdf":
                raise ValueError(
                    f"The file is not a PDF: {path}"
                )

            logger.info("Loading single PDF file: %s", path)

            loader = PyMuPDFLoader(str(path_obj))
            documents.extend(loader.load())

        # Handle a directory containing PDF files
        elif path_obj.is_dir():

            pdf_files = sorted(
                file
                for file in path_obj.iterdir()
                if file.is_file() and file.suffix.lower() == ".pdf"
            )

            if not pdf_files:
                logger.warning(
                    "No PDF files found in directory: %s",
                    path
                )
                return documents

            # Load each PDF independently
            for pdf_file in pdf_files:

                try:
                    logger.info(
                        "Loading PDF file: %s",
                        pdf_file.name
                    )

                    loader = PyMuPDFLoader(str(pdf_file))
                    documents.extend(loader.load())

                except Exception as e:
                    # Log the error and continue with the remaining files
                    logger.error(
                        "Failed to load file %s: %s",
                        pdf_file.name,
                        e
                    )

    except FileNotFoundError:
        logger.exception("Path not found: %s", path)
        raise

    except ValueError:
        logger.exception("Invalid PDF file: %s", path)
        raise

    except Exception as e:
        logger.exception("Unexpected error during PDF loading")
        raise RuntimeError(
            f"PDF loading failed: {e}"
        ) from e

    logger.info(
        "Successfully loaded a total of %d document page(s).",
        len(documents)
    )

    return documents