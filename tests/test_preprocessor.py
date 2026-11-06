
import pytest

from app.nlp.preprocessor import (
    create_chunks,
    normalize_text,
    split_into_sentences,
)


def test_normalize_text_cleans_whitespace():
    text = "  Payment   is due \n\n\n within 30 days.  "

    result = normalize_text(text)

    assert result == "Payment is due \n\n within 30 days."


def test_normalize_text_rejects_non_string():
    with pytest.raises(TypeError):
        normalize_text(123)


def test_sentence_tokenization():
    text = (
        "The agreement starts today. "
        "Payment is due within 30 days."
    )

    sentences = split_into_sentences(text)

    assert len(sentences) == 2
    assert sentences[0] == "The agreement starts today."
    assert sentences[1] == "Payment is due within 30 days."


def test_create_chunks_preserves_page_number_and_overlap():
    words = [f"word{i}" for i in range(1, 16)]
    text = " ".join(words)

    chunks = create_chunks(
        [{"page_number": 3, "text": text}],
        chunk_size=10,
        overlap=3,
    )

    assert len(chunks) == 2
    assert chunks[0]["page_number"] == 3
    assert chunks[0]["word_count"] == 10
    assert chunks[1]["word_count"] == 8

    first_words = chunks[0]["text"].split()
    second_words = chunks[1]["text"].split()

    assert first_words[-3:] == second_words[:3]


def test_create_chunks_rejects_invalid_overlap():
    with pytest.raises(ValueError):
        create_chunks([], chunk_size=10, overlap=10)


def test_create_chunks_skips_empty_page_text():
    chunks = create_chunks(
        [{"page_number": 1, "text": "   "}]
    )

    assert chunks == []
