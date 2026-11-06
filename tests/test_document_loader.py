
import pymupdf
import pytest

from app.services.document_loader import extract_pdf_text


def test_missing_pdf_raises_error():
    with pytest.raises(FileNotFoundError):
        extract_pdf_text("this_file_does_not_exist.pdf")


def test_non_pdf_file_raises_error(tmp_path):
    text_file = tmp_path / "sample.txt"
    text_file.write_text("This is not a PDF.", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported file type"):
        extract_pdf_text(text_file)


def test_extracts_text_and_preserves_page_numbers(tmp_path):
    pdf_path = tmp_path / "sample.pdf"

    with pymupdf.open() as document:
        page_one = document.new_page()
        page_one.insert_text(
            (72, 72),
            "Natural language processing helps computers understand text.",
        )

        page_two = document.new_page()
        page_two.insert_text(
            (72, 72),
            "Question answering retrieves answers from documents.",
        )

        document.save(pdf_path)

    pages = extract_pdf_text(pdf_path)

    assert len(pages) == 2
    assert pages[0]["page_number"] == 1
    assert "natural language processing" in pages[0]["text"].lower()
    assert pages[1]["page_number"] == 2
    assert "question answering" in pages[1]["text"].lower()


def test_rejects_pdf_without_readable_text(tmp_path):
    pdf_path = tmp_path / "empty.pdf"

    with pymupdf.open() as document:
        document.new_page()
        document.save(pdf_path)

    with pytest.raises(ValueError, match="No readable text"):
        extract_pdf_text(pdf_path)
