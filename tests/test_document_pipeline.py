
import pymupdf
import pytest

from app.services.document_pipeline import DocumentPipeline


@pytest.fixture
def sample_pdf(tmp_path):
    pdf_path = tmp_path / "contract.pdf"

    with pymupdf.open() as document:
        page_one = document.new_page()
        page_one.insert_text(
            (72, 72),
            "Invoices must be paid within thirty days of receipt.",
        )

        page_two = document.new_page()
        page_two.insert_text(
            (72, 72),
            "Either party may terminate this contract with sixty days written notice.",
        )

        document.save(pdf_path)

    return pdf_path


def test_process_pdf_returns_document_statistics(sample_pdf):
    pipeline = DocumentPipeline()

    stats = pipeline.process_pdf(sample_pdf)

    assert stats["document_name"] == "contract.pdf"
    assert stats["page_count"] == 2
    assert stats["chunk_count"] >= 2
    assert stats["word_count"] > 0
    assert pipeline.is_ready is True


def test_search_returns_source_page(sample_pdf):
    pipeline = DocumentPipeline()
    pipeline.process_pdf(sample_pdf)

    results = pipeline.search(
        "How much notice is needed to end the contract?",
        method="semantic",
        top_k=2,
    )

    assert results
    assert results[0]["page_number"] == 2
    assert results[0]["relevance_score"] > 0


def test_search_before_processing_raises_error():
    pipeline = DocumentPipeline()

    with pytest.raises(RuntimeError, match="Process a PDF"):
        pipeline.search("payment terms")


def test_unknown_retrieval_method_raises_error(sample_pdf):
    pipeline = DocumentPipeline()
    pipeline.process_pdf(sample_pdf)

    with pytest.raises(ValueError, match="method"):
        pipeline.search("payment terms", method="unknown")
