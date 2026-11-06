
import re
from typing import Any

import nltk
from nltk.tokenize import sent_tokenize


def normalize_text(text: str) -> str:
    """
    Normalize whitespace while preserving words, punctuation, and numbers.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def split_into_sentences(text: str) -> list[str]:
    """
    Split text into sentences using NLTK sentence tokenization.
    """
    cleaned_text = normalize_text(text)

    if not cleaned_text:
        return []

    try:
        return sent_tokenize(cleaned_text)
    except LookupError:
        # Support NLTK versions that request this tokenizer resource.
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)
        return sent_tokenize(cleaned_text)


def create_chunks(
    pages: list[dict[str, Any]],
    chunk_size: int = 120,
    overlap: int = 30,
) -> list[dict[str, Any]]:
    """
    Split page text into overlapping word-based chunks.

    Each chunk retains its source page number.

    Args:
        pages: List of dictionaries with page_number and text.
        chunk_size: Maximum number of words in each chunk.
        overlap: Number of words shared between consecutive chunks.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    chunks = []
    chunk_id = 1
    step = chunk_size - overlap

    for page in pages:
        if "page_number" not in page or "text" not in page:
            raise ValueError("Each page must contain page_number and text")

        text = normalize_text(page["text"])
        words = text.split()

        for start in range(0, len(words), step):
            chunk_words = words[start : start + chunk_size]

            if not chunk_words:
                break

            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "page_number": page["page_number"],
                    "text": " ".join(chunk_words),
                    "word_count": len(chunk_words),
                }
            )
            chunk_id += 1

            if start + chunk_size >= len(words):
                break

    return chunks
