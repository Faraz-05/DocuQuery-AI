
from unittest.mock import patch

import pytest

from app.qa.qa_service import QAService


def test_rejects_empty_question():
    service = QAService()
    with pytest.raises(ValueError, match="Question cannot be empty"):
        service.answer("", "The contract lasts one year.")


def test_rejects_empty_context():
    service = QAService()
    with pytest.raises(ValueError, match="Context cannot be empty"):
        service.answer("How long is the contract?", "  ")



@patch("app.qa.qa_service.load_qa_pipeline")
def test_returns_answer_and_confidence(mock_loader):
    mock_qa_model = mock_loader.return_value
    mock_qa_model.return_value = {
        "answer": "one year",
        "confidence": 0.95,
        "start": 20,
        "end": 28,
    }

    result = QAService().answer(
        "How long is the contract?",
        "The agreement lasts one year.",
    )

    assert result["answer"] == "one year"
    assert result["confidence"] == pytest.approx(0.95)
    assert result["start"] == 20
    assert result["end"] == 28



@patch("app.qa.qa_service.load_qa_pipeline")
def test_strips_question_and_context(mock_loader):
    mock_qa_model = mock_loader.return_value
    mock_qa_model.return_value = {
        "answer": "30 days",
        "score": 0.9,
        "start": 0,
        "end": 7,
    }

    QAService().answer("  When?  ", "  30 days  ")

    mock_loader.assert_called_once()
    mock_qa_model.assert_called_once_with(
        question="When?",
        context="30 days",
    )
