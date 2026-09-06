from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from app.rag.loaders import UniversalLoader


MODULE_PATH = "app.rag.loaders"


def _make_fake_loader(return_docs):
    mock_loader_instance = MagicMock()
    mock_loader_instance.load.return_value = return_docs

    mock_loader_class = MagicMock(return_value=mock_loader_instance)
    return mock_loader_class


def _fake_documents(n=1, source="dummy"):
    return [
        Document(page_content=f"content {i}", metadata={"source": source})
        for i in range(n)
    ]


def _patched_extensions(**overrides):
    patched = dict(UniversalLoader.SUPPORTED_EXTENSIONS)
    patched.update(overrides)
    return patch.object(UniversalLoader, "SUPPORTED_EXTENSIONS", patched)


class TestLoadInvalidInput:

    def test_path_does_not_exist_raises_file_not_found(self, tmp_path):
        missing_path = tmp_path / "does_not_exist.pdf"

        loader = UniversalLoader(str(missing_path))

        with pytest.raises(FileNotFoundError):
            loader.load()

    def test_unsupported_extension_raises_value_error(self, tmp_path):
        file_path = tmp_path / "notes.xyz"
        file_path.write_text("some content")

        loader = UniversalLoader(str(file_path))

        with pytest.raises(ValueError, match="Unsupported file format"):
            loader.load()

    def test_neither_file_nor_directory_raises_value_error(self, tmp_path):
        real_path = tmp_path / "something.pdf"
        real_path.write_text("placeholder")

        loader = UniversalLoader(str(real_path))

        with patch.object(type(loader.path), "is_file", return_value=False), \
             patch.object(type(loader.path), "is_dir", return_value=False):
            with pytest.raises(ValueError, match="neither a file nor a directory"):
                loader.load()


class TestLoadHappyPathByType:

    def test_load_pdf_success(self, tmp_path):
        file_path = tmp_path / "doc.pdf"
        file_path.write_bytes(b"%PDF-1.4 fake content")
        fake_docs = _fake_documents(2, source="doc.pdf")
        fake_loader_class = _make_fake_loader(fake_docs)

        with _patched_extensions(**{".pdf": fake_loader_class}):
            result = UniversalLoader(str(file_path)).load()

        fake_loader_class.assert_called_once_with(str(file_path))
        assert result == fake_docs

    def test_load_docx_success(self, tmp_path):
        file_path = tmp_path / "doc.docx"
        file_path.write_bytes(b"fake docx content")
        fake_docs = _fake_documents(1, source="doc.docx")
        fake_loader_class = _make_fake_loader(fake_docs)

        with _patched_extensions(**{".docx": fake_loader_class}):
            result = UniversalLoader(str(file_path)).load()

        fake_loader_class.assert_called_once_with(str(file_path))
        assert result == fake_docs

    def test_load_pptx_success(self, tmp_path):
        file_path = tmp_path / "doc.pptx"
        file_path.write_bytes(b"fake pptx content")
        fake_docs = _fake_documents(3, source="doc.pptx")
        fake_loader_class = _make_fake_loader(fake_docs)

        with _patched_extensions(**{".pptx": fake_loader_class}):
            result = UniversalLoader(str(file_path)).load()

        fake_loader_class.assert_called_once_with(str(file_path))
        assert result == fake_docs

    def test_load_txt_success(self, tmp_path):
        file_path = tmp_path / "doc.txt"
        file_path.write_text("plain text content")
        fake_docs = _fake_documents(1, source="doc.txt")
        fake_loader_class = _make_fake_loader(fake_docs)

        with _patched_extensions(**{".txt": fake_loader_class}):
            result = UniversalLoader(str(file_path)).load()

        fake_loader_class.assert_called_once_with(str(file_path))
        assert result == fake_docs

    @pytest.mark.parametrize("extension", [".pdf", ".docx", ".pptx", ".txt"])
    def test_extension_matching_is_case_insensitive(self, tmp_path, extension):
        file_path = tmp_path / f"doc{extension.upper()}"
        file_path.write_text("content")
        fake_docs = _fake_documents(1)
        fake_loader_class = _make_fake_loader(fake_docs)

        with _patched_extensions(**{extension: fake_loader_class}):
            result = UniversalLoader(str(file_path)).load()

        assert result == fake_docs


class TestLoadOperationFailure:

    def test_loader_exception_is_wrapped_in_runtime_error(self, tmp_path):
        file_path = tmp_path / "corrupt.pdf"
        file_path.write_bytes(b"broken")

        broken_loader_instance = MagicMock()
        broken_loader_instance.load.side_effect = Exception("parsing exploded")
        broken_loader_class = MagicMock(return_value=broken_loader_instance)

        with _patched_extensions(**{".pdf": broken_loader_class}):
            loader = UniversalLoader(str(file_path))

            with pytest.raises(RuntimeError, match="Document loading failed"):
                loader.load()

    def test_runtime_error_is_chained_from_original_exception(self, tmp_path):
        file_path = tmp_path / "corrupt.docx"
        file_path.write_bytes(b"broken")

        broken_loader_instance = MagicMock()
        original_exc = OSError("malformed zip")
        broken_loader_instance.load.side_effect = original_exc
        broken_loader_class = MagicMock(return_value=broken_loader_instance)

        with _patched_extensions(**{".docx": broken_loader_class}):
            loader = UniversalLoader(str(file_path))

            with pytest.raises(RuntimeError) as exc_info:
                loader.load()

        assert exc_info.value.__cause__ is original_exc

    def test_value_error_and_file_not_found_are_not_wrapped(self, tmp_path):
        file_path = tmp_path / "notes.xyz"
        file_path.write_text("content")

        loader = UniversalLoader(str(file_path))

        with pytest.raises(ValueError):
            loader.load()


class TestLoadDirectory:

    def test_directory_with_mixed_supported_files(self, tmp_path):
        (tmp_path / "a.pdf").write_bytes(b"pdf content")
        (tmp_path / "b.txt").write_text("txt content")

        pdf_docs = _fake_documents(1, source="a.pdf")
        txt_docs = _fake_documents(1, source="b.txt")

        with _patched_extensions(
            **{".pdf": _make_fake_loader(pdf_docs), ".txt": _make_fake_loader(txt_docs)}
        ):
            result = UniversalLoader(str(tmp_path)).load()

        assert len(result) == 2
        assert pdf_docs[0] in result
        assert txt_docs[0] in result

    def test_directory_ignores_unsupported_files(self, tmp_path):
        (tmp_path / "a.pdf").write_bytes(b"pdf content")
        (tmp_path / "ignore_me.xyz").write_text("irrelevant")

        pdf_docs = _fake_documents(1, source="a.pdf")

        with _patched_extensions(**{".pdf": _make_fake_loader(pdf_docs)}):
            result = UniversalLoader(str(tmp_path)).load()

        assert result == pdf_docs

    def test_directory_with_no_supported_files_returns_empty_list(self, tmp_path):
        (tmp_path / "ignore_me.xyz").write_text("irrelevant")

        result = UniversalLoader(str(tmp_path)).load()

        assert result == []

    def test_empty_directory_returns_empty_list(self, tmp_path):
        result = UniversalLoader(str(tmp_path)).load()

        assert result == []

    def test_directory_partial_failure_does_not_stop_other_files(self, tmp_path):
        (tmp_path / "good.pdf").write_bytes(b"good content")
        (tmp_path / "bad.txt").write_text("bad content")

        good_docs = _fake_documents(1, source="good.pdf")

        broken_txt_instance = MagicMock()
        broken_txt_instance.load.side_effect = Exception("txt parsing failed")
        broken_txt_class = MagicMock(return_value=broken_txt_instance)

        with _patched_extensions(
            **{".pdf": _make_fake_loader(good_docs), ".txt": broken_txt_class}
        ):
            result = UniversalLoader(str(tmp_path)).load()

        assert result == good_docs

    def test_directory_all_files_fail_returns_empty_list(self, tmp_path):
        (tmp_path / "bad.pdf").write_bytes(b"bad content")

        broken_instance = MagicMock()
        broken_instance.load.side_effect = Exception("boom")
        broken_class = MagicMock(return_value=broken_instance)

        with _patched_extensions(**{".pdf": broken_class}):
            result = UniversalLoader(str(tmp_path)).load()

        assert result == []