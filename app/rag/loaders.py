import logging
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    Docx2txtLoader,
    UnstructuredPowerPointLoader,
    TextLoader,
)


logger = logging.getLogger(__name__)


class UniversalLoader:
    """
    Load supported document formats and return a unified
    list of LangChain Document objects.
    """

    SUPPORTED_EXTENSIONS = {
        ".pdf": PyMuPDFLoader,
        ".docx": Docx2txtLoader,
        ".pptx": UnstructuredPowerPointLoader,
        ".txt": TextLoader,
    }

    def __init__(self, path: str):
        self.path = Path(path)

    def load(self) -> list[Document]:
        """
        Load documents from a file or directory.

        Returns:
            list[Document]: Loaded LangChain documents.

        Raises:
            FileNotFoundError: If the path does not exist.
            ValueError: If the file format is not supported.
            RuntimeError: If an unexpected loading error occurs.
        """

        if not self.path.exists():
            raise FileNotFoundError(
                f"The specified path does not exist: {self.path}"
            )

        try:
            if self.path.is_file():
                return self._load_file(self.path)

            if self.path.is_dir():
                return self._load_directory()

            raise ValueError(
                f"The specified path is neither a file nor a directory: "
                f"{self.path}"
            )

        except (FileNotFoundError, ValueError):
            raise

        except Exception as e:
            logger.exception(
                "Unexpected error while loading: %s",
                self.path
            )
            raise RuntimeError(
                f"Document loading failed: {e}"
            ) from e

    def _load_file(self, file_path: Path) -> list[Document]:
        """Load a single supported file."""

        extension = file_path.suffix.lower()

        loader_class = self.SUPPORTED_EXTENSIONS.get(extension)

        if loader_class is None:
            raise ValueError(
                f"Unsupported file format: {extension}"
            )

        logger.info("Loading file: %s", file_path.name)

        loader = loader_class(str(file_path))
        documents = loader.load()

        logger.info(
            "Loaded %d document(s) from %s",
            len(documents),
            file_path.name,
        )

        return documents

    def _load_directory(self) -> list[Document]:
        """Load all supported files from a directory."""

        documents: list[Document] = []

        files = sorted(
            file
            for file in self.path.iterdir()
            if file.is_file()
            and file.suffix.lower() in self.SUPPORTED_EXTENSIONS
        )

        if not files:
            logger.warning(
                "No supported files found in directory: %s",
                self.path,
            )
            return documents

        for file_path in files:
            try:
                documents.extend(
                    self._load_file(file_path)
                )

            except Exception as e:
                logger.error(
                    "Failed to load file %s: %s",
                    file_path.name,
                    e,
                )

        logger.info(
            "Successfully loaded %d document(s) from directory: %s",
            len(documents),
            self.path,
        )

        return documents