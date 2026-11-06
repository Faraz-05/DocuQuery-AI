
from pathlib import Path

import pymupdf


def extract_pdf_text(file_path: str | Path) -> list[dict]:
    """
    Extract text from a PDF while preserving page numbers.

    Returns:
        A list of dictionaries containing each page's number and text.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError("Unsupported file type. Please provide a PDF.")

    pages = []

    with pymupdf.open(path) as document:
        if document.page_count == 0:
            raise ValueError("The PDF contains no pages.")

        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text").strip()

            pages.append(
                {
                    "page_number": page_number,
                    "text": text,
                }
            )

    if not any(page["text"] for page in pages):
        raise ValueError(
            "No readable text was found. The PDF may contain scanned images "
            "and require OCR."
        )

    return pages
