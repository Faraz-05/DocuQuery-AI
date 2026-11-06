
import pytest

from app.qa.qa_service import QAService
from app.services.document_pipeline import DocumentPipeline


@pytest.fixture
def sample_pdf(tmp_path):
    """Create a small PDF with known facts on separate pages."""
    import fitz

    pdf_path = tmp_path / "evaluation_lease.pdf"

    with fitz.open() as pdf:
        page_one = pdf.new_page()
        page_one.insert_text(
            (72, 72),
            "Section 4.1. The monthly base rent is Rs. 1,35,000.",
        )

        page_two = pdf.new_page()
        page_two.insert_text(
            (72, 72),
            "Section 4.2. Rent escalation is 8% per annum.",
        )

        page_three = pdf.new_page()
        page_three.insert_text(
            (72, 72),
            "Section 5. The security deposit is Rs. 8,10,000.",
        )

        pdf.save(pdf_path)

    return pdf_path


@pytest.fixture
def qa_service():
    """Create the QA service without loading the transformer model."""
    return QAService()


@pytest.fixture
def lease_context():
    """Representative facts from the sample lease document."""
    return """
    Section 4.1: The monthly base rent is Rs. 1,35,000.
    Section 4.2: Rent escalation shall be 8% per annum,
    effective on each anniversary of the commencement date.
    Section 4.4: The total estimated property tax is
    Rs. 1,80,000 per annum.
    Section 5: The security deposit is Rs. 8,10,000,
    is interest-free, and is refundable.
    """


# --------------------------------------------------
# End-to-end document retrieval tests
# --------------------------------------------------

@pytest.mark.parametrize("method", ["semantic", "tfidf"])
def test_pipeline_retrieves_relevant_fact(sample_pdf, method):
    pipeline = DocumentPipeline()
    stats = pipeline.process_pdf(sample_pdf)

    results = pipeline.search(
        "What is the monthly base rent?",
        method=method,
        top_k=3,
    )

    assert stats["page_count"] == 3
    assert stats["chunk_count"] > 0
    assert results, f"{method} retrieval returned no results"

    combined_text = " ".join(
        result["text"] for result in results
    ).lower()

    assert "1,35,000" in combined_text, (
        f"{method} did not retrieve the expected rent amount"
    )


@pytest.mark.parametrize("method", ["semantic", "tfidf"])
def test_pipeline_preserves_relevant_source_page(sample_pdf, method):
    pipeline = DocumentPipeline()
    pipeline.process_pdf(sample_pdf)

    results = pipeline.search(
        "What is the security deposit?",
        method=method,
        top_k=3,
    )

    assert results, "No search results were returned"

    matching_results = [
        result
        for result in results
        if "8,10,000" in result["text"]
    ]

    assert matching_results, (
        f"{method} did not retrieve the security deposit passage"
    )

    assert any(
        result.get("page_number") == 3
        for result in matching_results
    ), "The matching passage should identify PDF page 3"


# --------------------------------------------------
# QA answer-quality tests
# --------------------------------------------------

def test_monthly_rent_answer_is_supported(qa_service, lease_context):
    result = qa_service.answer(
        "What is the monthly rent amount?",
        lease_context,
    )

    assert result["method"] == "source_match"
    assert "1,35,000" in result["answer"]


def test_rent_escalation_answer_is_supported(qa_service, lease_context):
    result = qa_service.answer(
        "What is the annual rent escalation percentage?",
        lease_context,
    )

    assert result["method"] == "source_match"
    assert "8%" in result["answer"]


def test_property_tax_answer_is_supported(qa_service, lease_context):
    result = qa_service.answer(
        "What is the total property tax?",
        lease_context,
    )

    assert result["method"] == "source_match"
    assert "1,80,000" in result["answer"]


def test_security_deposit_answer_is_supported(qa_service, lease_context):
    result = qa_service.answer(
        "What is the security deposit amount?",
        lease_context,
    )

    assert result["method"] == "source_match"
    assert "8,10,000" in result["answer"]


def test_missing_rent_escalation_evidence_abstains(qa_service):
    context = """
    The tenant must maintain the premises in good condition.
    The tenant must keep the premises clean.
    """

    result = qa_service.answer(
        "What is the annual rent escalation percentage?",
        context,
    )

    assert result["method"] == "abstain"


def test_answer_is_present_in_source_context(qa_service, lease_context):
    questions = [
        "What is the monthly rent amount?",
        "What is the annual rent escalation percentage?",
        "What is the total property tax?",
        "What is the security deposit amount?",
    ]

    for question in questions:
        result = qa_service.answer(question, lease_context)

        assert result["answer"] in lease_context
